"""Spatial index of shapes (RFC 4: tiled and multiscale shapes), experimental.

The shapes are sorted by the tile of their centroid on a grid of their own (as in RFC 3, a single level), the geometry
is written as GeoArrow, the largest overhang of a shape outside its tile (the margin) is recorded in
``spatial_index``, and the table annotating the shapes can be reordered to the same row order, with the expression
matrix as CSR (``X``) and CSC (``layers["X_csc"]``). Layout as in RFC 3::

    <element>/spatial/shapes.parquet/        rows sorted by tile, GeoArrow
    <element>/spatial/tile_offsets/0         tile index
    <element>/spatial_index.parquet.json     part files in order, first rows
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import zarr
from anndata import AnnData
from scipy import sparse

from spatialdata._io.spatial_index._build import _part_name, _row_groups, _write_offsets
from spatialdata._io.spatial_index._metadata import LevelGrid, SpatialIndexMetadata
from spatialdata._io.spatial_index._read import _ParquetTable

if TYPE_CHECKING:
    from spatialdata import SpatialData

CSC_LAYER = "X_csc"
MIN_FRACTION_WITHIN_TILE = 0.5


def _centroids_and_bboxes(shapes: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geoms = shapes.geometry.to_numpy()
    if "radius" in shapes.columns:  # circles: the center, bbox from the radius
        c = shapely.get_coordinates(geoms)
        r = shapes["radius"].to_numpy(dtype=np.float64)[:, None]
        return c, c - r, c + r
    b = shapely.bounds(geoms)
    c = shapely.get_coordinates(shapely.centroid(geoms))
    return c, b[:, :2], b[:, 2:]


def fraction_within_tile(c: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, grid: LevelGrid) -> float:
    """Fraction of the shapes that lie entirely within the tile of their centroid (RFC 4, "When to use it")."""
    t = grid.tile_coords(c)
    lo = np.asarray(grid.origin) + t * np.asarray(grid.tile_size)
    hi = lo + np.asarray(grid.tile_size)
    return float(np.mean(np.all((bmin >= lo) & (bmax <= hi), axis=1)))


def _grid(origin: np.ndarray, s: float, c: np.ndarray) -> LevelGrid:
    n = np.floor((c.max(axis=0) - origin) / s).astype(np.int64) + 1
    return LevelGrid(tuple(origin.tolist()), (float(s),) * len(origin), tuple(int(v) for v in n), 1.0)


def choose_shapes_grid(
    c: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
    nest_with: LevelGrid | None = None,
    min_fraction: float = MIN_FRACTION_WITHIN_TILE,
) -> tuple[LevelGrid, float]:
    """The grid of the shapes: the smallest tile size such that ``min_fraction`` of them lie within their tile.

    With ``nest_with`` (the level-0 grid of the points of the same dataset), the tile size is that of a level of the
    points' merge rule (a power of 2 times its tile size) and the origin is aligned with it, so that each tile of the
    shapes is a union of tiles of the points (RFC 4, "Spatial order"); the grid of the shapes covers its own bounding
    box (the origin is shifted by whole tiles of the points if needed).
    """
    if nest_with is not None:
        s0 = float(nest_with.tile_size[0])
        o_p = np.asarray(nest_with.origin[: c.shape[1]], dtype=np.float64)
        for k in range(0, 16):
            s = s0 * 2**k
            shift = np.maximum(np.ceil((o_p - c.min(axis=0)) / s), 0)
            o = o_p - shift * s
            grid = _grid(o, s, c)
            f = fraction_within_tile(c, bmin, bmax, grid)
            if f >= min_fraction:
                return grid, f
        return grid, f
    o = np.floor(c.min(axis=0))
    size = np.median(np.max(bmax - bmin, axis=1))
    s = float(size)
    for _ in range(64):
        grid = _grid(o, s, c)
        f = fraction_within_tile(c, bmin, bmax, grid)
        if f >= min_fraction:
            return grid, f
        s *= 1.25
    return grid, f


def write_shapes_with_spatial_index(
    shapes: gpd.GeoDataFrame,
    path: str | Path,
    *,
    grid: LevelGrid | None = None,
    nest_with: LevelGrid | None = None,
    geometry_encoding: str = "geoarrow",
    row_group_min_rows: int = 65_536,
    row_group_max_rows: int = 262_144,
    overwrite: bool = False,
    table: str | None = None,
) -> tuple[SpatialIndexMetadata, np.ndarray]:
    """Write a shapes element sorted by the tile of its centroids (RFC 4) to a new element group at ``path``.

    Parameters
    ----------
    grid
        Grid to use; default: chosen by :func:`choose_shapes_grid` (rule of thumb of RFC 4, nested with ``nest_with``
        if given).
    nest_with
        Level-0 grid of the points of the same dataset, to nest the grid of the shapes with it.
    table
        Name of the table whose rows are (or will be) in the order of the shapes, recorded in ``spatial_index``.

    Returns
    -------
    The metadata and the permutation applied to the rows (new row i is old row ``order[i]``).
    """
    from spatialdata._io._utils import _write_metadata, overwrite_coordinate_transformations_non_raster
    from spatialdata._io.format import ShapesFormatV04
    from spatialdata._io.spatial_index import _ensure_parent_group
    from spatialdata._io.spatial_index._sidecar import write_spatial_index
    from spatialdata.models import get_axes_names
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations._utils import _get_transformations

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True.")
        shutil.rmtree(path)
    c, bmin, bmax = _centroids_and_bboxes(shapes)
    if grid is None:
        grid, fraction = choose_shapes_grid(c, bmin, bmax, nest_with)
    else:
        fraction = fraction_within_tile(c, bmin, bmax, grid)
    if fraction < MIN_FRACTION_WITHIN_TILE:
        warnings.warn(
            f"Only {fraction:.0%} of the shapes lie within the tile of their centroid; RFC 4 recommends the index only "
            "for small shapes (more than 50% within a tile).",
            UserWarning,
            stacklevel=2,
        )
    margin = np.maximum((c - bmin).max(axis=0), (bmax - c).max(axis=0))
    md = SpatialIndexMetadata(
        axes=("x", "y"),
        levels=(grid,),
        feature_keys=(),
        margin=tuple(float(v) for v in margin),
        table=table,
        extra={"fraction_within_tile": fraction},
    )
    md.validate()
    tile = grid.tile_ids(c)
    order = np.argsort(tile, kind="stable")
    sorted_shapes = shapes.iloc[order]
    tile_sorted = tile[order]

    _ensure_parent_group(path)
    group = zarr.open_group(path, mode="w", zarr_format=3)
    spatial = group.require_group("spatial")
    table_dir = path / "spatial" / "shapes.parquet"
    table_dir.mkdir(parents=True)
    transforms = sorted_shapes.attrs.get(TRANSFORM_KEY)
    gdf = gpd.GeoDataFrame(sorted_shapes, geometry="geometry")
    gdf.attrs = {}
    if geometry_encoding == "geoarrow" and (gdf.geom_type == "MultiPolygon").any():
        # RFC 4: a column that mixes Polygon and MultiPolygon is written as MultiPolygon
        gdf["geometry"] = gdf.geometry.apply(
            lambda g: g if g.geom_type == "MultiPolygon" else shapely.MultiPolygon([g])
        )
    from geopandas.io.arrow import _geopandas_to_arrow

    arrow = _geopandas_to_arrow(gdf, index=None, geometry_encoding=geometry_encoding)
    # the geometry first, then the other columns (RFC 4, "Geometry")
    names = ["geometry"] + [n for n in arrow.schema.names if n != "geometry"]
    arrow = arrow.select(names)
    boundaries = np.flatnonzero(np.diff(tile_sorted, prepend=-1))
    groups = _row_groups(boundaries, len(gdf), row_group_min_rows, row_group_max_rows)
    name = _part_name(0)
    with pq.ParquetWriter(table_dir / name, arrow.schema, compression="zstd") as w:
        for a, b in groups:
            w.write_table(arrow.slice(a, b - a), row_group_size=b - a)
    offs = np.concatenate([[0], np.cumsum(np.bincount(tile_sorted, minlength=grid.n_tiles))])
    _write_offsets(spatial.require_group("tile_offsets"), "0", offs, np.int32 if len(gdf) < 2**31 else np.int64)

    fmt = ShapesFormatV04()
    _write_metadata(
        group,
        group_type="ngff:shapes",
        axes=list(get_axes_names(shapes)),
        attrs={"version": fmt.spatialdata_format_version},
    )
    if transforms is None:
        transforms = _get_transformations(shapes)
    overwrite_coordinate_transformations_non_raster(group=group, axes=get_axes_names(shapes), transformations=transforms)
    sidecar = {"tables": {"spatial": {"files": [f"spatial/shapes.parquet/{name}"], "first_row": [0, len(gdf)]}}}
    write_spatial_index(path, md, sidecar)
    return md, order


def reorder_table_to_shapes(table: AnnData, shapes_index: pd.Index, add_csc: bool = True) -> AnnData:
    """The rows of ``table``, which annotates the shapes, in the order of ``shapes_index`` (one row per shape).

    With ``add_csc``, ``X`` is CSR and ``layers["X_csc"]`` is CSC with the same values (RFC 4).
    """
    from spatialdata.models import TableModel

    attrs = table.uns[TableModel.ATTRS_KEY]
    pos = pd.Index(table.obs[attrs["instance_key"]].to_numpy()).get_indexer(shapes_index.to_numpy())
    if np.any(pos < 0):
        raise ValueError("Some shapes are not annotated by the table.")
    t = table[pos].copy()
    if add_csc:
        X = t.X if sparse.issparse(t.X) else sparse.csr_matrix(t.X)
        t.X = sparse.csr_matrix(X)
        t.layers[CSC_LAYER] = sparse.csc_matrix(X)
    return t


def build_shapes_spatial_index(
    sdata: SpatialData,
    shapes_name: str,
    *,
    table_name: str | None = None,
    nest_with_points: str | None = None,
    grid: LevelGrid | None = None,
    add_csc: bool = True,
) -> ShapesSpatialIndex:
    """Build the spatial index of RFC 4 for a shapes element of a disk-backed SpatialData object (in place).

    Parameters
    ----------
    table_name
        Table annotating the shapes (its ``region`` must be the shapes element; making sure of it is the user's
        responsibility, RFC 4). Its rows are reordered to the order of the shapes, so that the tile offsets also
        slice the table, and, with ``add_csc``, a CSC copy of ``X`` is added.
    nest_with_points
        An indexed points element of the same dataset: the grid of the shapes is nested with its grid (RFC 4,
        "Spatial order"). Otherwise the grid is chosen for the shapes alone.
    grid
        Grid to use instead.
    """
    from spatialdata._io.io_shapes import _read_shapes
    from spatialdata._io.spatial_index import PointsSpatialIndex
    from spatialdata.models import TableModel

    if sdata.path is None:
        raise ValueError("The SpatialData object must be backed by a Zarr store.")
    root = Path(sdata.path)
    shapes = sdata.shapes[shapes_name]
    if table_name is not None:
        region = sdata.tables[table_name].uns[TableModel.ATTRS_KEY]["region"]
        regions = [region] if isinstance(region, str) else list(region)
        if shapes_name not in regions:
            raise ValueError(
                f"The table {table_name!r} annotates {regions}, not {shapes_name!r}: RFC 4 applies only to a table "
                "that annotates the shapes."
            )
    nest_grid = None
    if nest_with_points is not None:
        nest_grid = PointsSpatialIndex(root / "points" / nest_with_points).metadata.levels[0]
    tmp = root / "shapes" / f"{shapes_name}.__spatial_index_tmp__"
    write_shapes_with_spatial_index(shapes, tmp, grid=grid, nest_with=nest_grid, overwrite=True, table=table_name)
    final = root / "shapes" / shapes_name
    shutil.rmtree(final)
    tmp.rename(final)
    sdata.shapes[shapes_name] = _read_shapes(final)

    if table_name is not None:
        table = sdata.tables[table_name]
        new_table = reorder_table_to_shapes(table, sdata.shapes[shapes_name].index, add_csc=add_csc)
        new_table.uns[TableModel.ATTRS_KEY] = dict(table.uns[TableModel.ATTRS_KEY])
        del sdata.tables[table_name]
        sdata.tables[table_name] = TableModel.parse(new_table)
        sdata.delete_element_from_disk(table_name)
        sdata.write_element(table_name)
    return ShapesSpatialIndex(final, root / "tables" / table_name if table_name else None)


class ShapesSpatialIndex:
    """Read shapes (and the rows of their table) by region through the spatial index of RFC 4."""

    def __init__(self, path: str | Path, table_path: str | Path | None = None) -> None:
        from spatialdata._io.spatial_index._sidecar import read_metadata

        self.path = Path(path)
        g = zarr.open_group(self.path, mode="r")
        self.metadata, sidecar = read_metadata(self.path)
        self.margin = np.asarray(self.metadata.margin or (0.0, 0.0), dtype=np.float64)
        self.offsets = g["spatial"]["tile_offsets"]["0"]
        assert sidecar is not None
        entry = sidecar["tables"]["spatial"]
        self.table = _ParquetTable([self.path / f for f in entry["files"]], np.asarray(entry["first_row"]))
        if table_path is None and self.metadata.table is not None:
            candidate = self.path.parent.parent / "tables" / self.metadata.table
            table_path = candidate if candidate.exists() else None
        self.table_path = Path(table_path) if table_path is not None else None
        self._var_names: list[str] | None = None

    def row_ranges(self, box_min: Sequence[float], box_max: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(box_min, np.float64) - self.margin
        hi = np.asarray(box_max, np.float64) + self.margin
        tiles = self.metadata.levels[0].tiles_in_box(lo, hi)
        if len(tiles) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        o = self.offsets.get_coordinate_selection(np.concatenate([tiles, tiles + 1])).astype(np.int64)
        return o[: len(tiles)], o[len(tiles) :]

    def count(self, box_min: Sequence[float], box_max: Sequence[float]) -> int:
        s, e = self.row_ranges(box_min, box_max)
        return int(np.sum(e - s))

    def query(self, box_min: Sequence[float], box_max: Sequence[float], exact: bool = True) -> gpd.GeoDataFrame:
        """Shapes that intersect the box; the column ``__row__`` is their row in the element (and in the table)."""
        from spatialdata._io.spatial_index._read import _merge_ranges

        s, e = self.row_ranges(box_min, box_max)
        s, e = _merge_ranges(s, e)
        t = self.table.read_ranges(s, e, None)
        rows = np.concatenate([np.arange(a, b) for a, b in zip(s, e, strict=True)]) if len(s) else np.empty(0, int)
        gdf = gpd.GeoDataFrame.from_arrow(t) if t.num_rows else gpd.GeoDataFrame(geometry=[])
        if "__index_level_0__" in gdf.columns:
            gdf = gdf.set_index("__index_level_0__")
        gdf["__row__"] = rows
        if exact and len(gdf):
            box = shapely.box(*box_min, *box_max)
            gdf = gdf[gdf.intersects(box)]
        return gdf

    # --- table ----------------------------------------------------------------------------------------------------
    @property
    def var_names(self) -> list[str]:
        if self._var_names is None:
            assert self.table_path is not None
            import anndata

            var = anndata.io.read_elem(zarr.open_group(self.table_path, mode="r")["var"])
            self._var_names = list(var.index)
        return self._var_names

    def gene_values(self, gene: str, rows: np.ndarray | None = None, layer: str = CSC_LAYER) -> np.ndarray:
        """Values of one gene for the given rows (default: all), from the CSC layer: one column read."""
        assert self.table_path is not None
        g = zarr.open_group(self.table_path, mode="r")
        m = g["layers"][layer]
        n_obs = int(m.attrs["shape"][0])
        j = self.var_names.index(gene)
        a, b = (int(v) for v in m["indptr"][j : j + 2])
        idx = m["indices"][a:b]
        val = m["data"][a:b]
        dense = np.zeros(n_obs, dtype=val.dtype)
        dense[idx] = val
        return dense if rows is None else dense[rows]

    def table_rows(self, rows: np.ndarray, layer: str | None = None) -> sparse.csr_matrix:
        """Rows of the CSR matrix X (or a CSR layer) for the given rows of the shapes (table aligned to shapes)."""
        assert self.table_path is not None
        g = zarr.open_group(self.table_path, mode="r")
        m = g["X"] if layer is None else g["layers"][layer]
        indptr = m["indptr"][:]
        n_vars = int(m.attrs["shape"][1])
        out_indptr = [0]
        idx, val = [], []
        for r in rows:
            a, b = int(indptr[r]), int(indptr[r + 1])
            idx.append(m["indices"][a:b])
            val.append(m["data"][a:b])
            out_indptr.append(out_indptr[-1] + b - a)
        return sparse.csr_matrix(
            (np.concatenate(val) if val else [], np.concatenate(idx) if idx else [], out_indptr),
            shape=(len(rows), n_vars),
        )


__all__ = [
    "ShapesSpatialIndex",
    "build_shapes_spatial_index",
    "choose_shapes_grid",
    "fraction_within_tile",
    "reorder_table_to_shapes",
    "write_shapes_with_spatial_index",
]

