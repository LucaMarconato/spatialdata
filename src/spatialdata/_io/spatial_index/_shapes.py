"""Spatial index of shapes (RFC 4: tiled and multiscale shapes), experimental.

The shapes are sorted by the tile of their centroid on a grid as in RFC 3, the geometry is written as GeoArrow, and
the table annotating them can be reordered to the same row order, with the expression matrix as CSR (``X``) and CSC
(``layers["X_csc"]``).
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import zarr
from anndata import AnnData
from scipy import sparse

from spatialdata._io.spatial_index._build import _row_groups, write_spatial_index_attrs
from spatialdata._io.spatial_index._metadata import LevelGrid, SpatialIndexMetadata, merge_rule
from spatialdata._io.spatial_index._read import _ParquetTable

if TYPE_CHECKING:
    from spatialdata import SpatialData

CSC_LAYER = "X_csc"


def _centroids_and_bboxes(shapes: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geoms = shapes.geometry.to_numpy()
    if "radius" in shapes.columns:  # circles: the center, bbox from the radius
        c = shapely.get_coordinates(geoms)
        r = shapes["radius"].to_numpy(dtype=np.float64)[:, None]
        return c, c - r, c + r
    b = shapely.bounds(geoms)
    c = shapely.get_coordinates(shapely.centroid(geoms))
    return c, b[:, :2], b[:, 2:]


def _fit_grid(grid: LevelGrid, c: np.ndarray) -> LevelGrid:
    """Extend a grid (whole tiles, same alignment) so that it covers all the centroids."""
    o = np.asarray(grid.origin, np.float64)
    s = np.asarray(grid.tile_size, np.float64)
    shift = np.maximum(np.ceil((o - c.min(axis=0)) / s), 0)
    o2 = o - shift * s
    n = np.floor((c.max(axis=0) - o2) / s).astype(np.int64) + 1
    n = np.maximum(n, np.asarray(grid.grid_shape) + shift.astype(np.int64))
    return LevelGrid(tuple(o2.tolist()), grid.tile_size, tuple(int(v) for v in n), 1.0)


def _choose_grid(c: np.ndarray, target: int) -> LevelGrid:
    o = np.floor(c.min(axis=0))
    extent = np.maximum(c.max(axis=0) - o, 1e-9)
    d = c.shape[1]
    s = (float(np.prod(extent)) * target / max(len(c), 1)) ** (1.0 / d)
    for _ in range(20):
        lv = LevelGrid(tuple(o), (s,) * d, tuple(int(v) for v in np.floor(extent / s) + 1), 1.0)
        densest = np.bincount(lv.tile_ids(c)).max()
        if densest <= target:
            return lv
        s *= 0.9 * (target / densest) ** (1.0 / d)
    return lv


def write_shapes_with_spatial_index(
    shapes: gpd.GeoDataFrame,
    path: str | Path,
    *,
    grid: LevelGrid | None = None,
    target_shapes_per_tile: int = 1_000,
    n_levels: int | None = 1,
    geometry_encoding: str = "geoarrow",
    row_group_min_rows: int | None = None,
    row_group_max_rows: int = 65_536,
    overwrite: bool = False,
) -> tuple[SpatialIndexMetadata, np.ndarray]:
    """Write a shapes element sorted by the tile of its centroids (RFC 4) to a new element group at ``path``.

    Parameters
    ----------
    grid
        Grid of level 0 to use, e.g. the one of the points of the same dataset (it is extended by whole tiles if it
        does not cover all the centroids); default: chosen for ``target_shapes_per_tile``.
    n_levels
        Number of levels of the grid (RFC 4 has no levels yet: the default is 1, the tiles of level 0 only).

    Returns
    -------
    The metadata and the permutation applied to the rows (new row i is old row ``order[i]``).
    """
    from spatialdata._io._utils import _write_metadata, overwrite_coordinate_transformations_non_raster
    from spatialdata._io.format import ShapesFormatV04
    from spatialdata.models import get_axes_names
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations._utils import _get_transformations

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True.")
        shutil.rmtree(path)
    c, bmin, bmax = _centroids_and_bboxes(shapes)
    level0 = _fit_grid(grid, c) if grid is not None else _choose_grid(c, target_shapes_per_tile)
    levels = merge_rule(level0, n_levels)
    margin = np.maximum((c - bmin).max(axis=0), (bmax - c).max(axis=0))
    md = SpatialIndexMetadata(
        axes=("x", "y"),
        levels=tuple(levels),
        feature_keys=(),
        extra={"margin": [float(v) for v in margin]},
    )
    md.validate()
    tile = level0.tile_ids(c)
    order = np.argsort(tile, kind="stable")
    sorted_shapes = shapes.iloc[order]
    tile_sorted = tile[order]

    from spatialdata._io.spatial_index import _ensure_parent_group

    _ensure_parent_group(path)
    group = zarr.open_group(path, mode="w", zarr_format=3)
    table_dir = path / "shapes.parquet"
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

    table = _geopandas_to_arrow(gdf, index=None, geometry_encoding=geometry_encoding)
    boundaries = np.flatnonzero(np.diff(tile_sorted, prepend=-1))
    groups = _row_groups(boundaries, len(gdf), row_group_min_rows or target_shapes_per_tile, row_group_max_rows)
    with pq.ParquetWriter(table_dir / "part.0.parquet", table.schema, compression="zstd") as w:
        for a, b in groups:
            w.write_table(table.slice(a, b - a), row_group_size=b - a)
    # tile index (level 0 only, as in RFC 3: tile_offsets/<l>)
    offs = np.concatenate([[0], np.cumsum(np.bincount(tile_sorted, minlength=level0.n_tiles))]).astype(np.int64)
    og = group.require_group("tile_offsets")
    og.create_array("0", data=offs, chunks=(min(len(offs), 65536),))

    fmt = ShapesFormatV04()
    _write_metadata(group, group_type="ngff:shapes", axes=list(get_axes_names(shapes)), attrs={"version": "0.4"})
    assert fmt.spatialdata_format_version == "0.4"
    if transforms is None:
        transforms = _get_transformations(shapes)
    overwrite_coordinate_transformations_non_raster(group=group, axes=get_axes_names(shapes), transformations=transforms)
    write_spatial_index_attrs(group, md)
    return md, order


def reorder_table_to_shapes(table: AnnData, shapes_index: pd.Index, add_csc: bool = True) -> AnnData:
    """The rows of ``table`` annotating ``shapes_index`` (one row per shape, in the order of the shapes).

    With ``add_csc``, ``X`` is CSR and ``layers["X_csc"]`` is CSC with the same values (RFC 4).
    """
    from spatialdata.models import TableModel

    attrs = table.uns[TableModel.ATTRS_KEY]
    instance_key = attrs["instance_key"]
    pos = pd.Index(table.obs[instance_key].to_numpy()).get_indexer(shapes_index.to_numpy())
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
    grid_from_points: str | None = None,
    target_shapes_per_tile: int = 1_000,
    add_csc: bool = True,
    centroids_points_name: str | None = None,
) -> ShapesSpatialIndex:
    """Build the spatial index of RFC 4 for a shapes element of a disk-backed SpatialData object (in place).

    Parameters
    ----------
    table_name
        Table annotating the shapes; its rows are reordered to the order of the shapes (so that the tile offsets
        also slice the table) and, with ``add_csc``, a CSC copy of ``X`` is added.
    grid_from_points
        Use the level-0 grid of this indexed points element (RFC 4: "shapes and points of the same dataset should use
        the same grid").
    centroids_points_name
        Also write the centroids as a points element with the spatial index of RFC 3 (levels for the zoomed-out
        views, one of the options of RFC 4 "Levels").
    """
    from spatialdata._io.io_shapes import _read_shapes
    from spatialdata._io.spatial_index import PointsSpatialIndex, write_points_with_spatial_index
    from spatialdata.models import PointsModel, TableModel
    from spatialdata.transformations import get_transformation

    if sdata.path is None:
        raise ValueError("The SpatialData object must be backed by a Zarr store.")
    root = Path(sdata.path)
    shapes = sdata.shapes[shapes_name]
    grid = None
    if grid_from_points is not None:
        grid = PointsSpatialIndex(root / "points" / grid_from_points).metadata.levels[0]
    tmp = root / "shapes" / f"{shapes_name}.__spatial_index_tmp__"
    md, order = write_shapes_with_spatial_index(
        shapes, tmp, grid=grid, target_shapes_per_tile=target_shapes_per_tile, overwrite=True
    )
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
        g = zarr.open_group(final, mode="a")
        si = g.attrs["spatial_index"]
        si.setdefault("spatialdata_extra", {})["table"] = {"name": table_name, "row_order": "same_as_shapes"}
        g.attrs["spatial_index"] = si

    if centroids_points_name is not None:
        c = shapely.get_coordinates(shapely.centroid(sdata.shapes[shapes_name].geometry.to_numpy()))
        df = pd.DataFrame({"x": c[:, 0], "y": c[:, 1], "instance_id": sdata.shapes[shapes_name].index.to_numpy()})
        t = get_transformation(sdata.shapes[shapes_name], get_all=True)
        pts = PointsModel.parse(df, instance_key="instance_id", transformations=t)
        write_points_with_spatial_index(
            pts,
            root / "points" / centroids_points_name,
            overwrite=True,
            feature_keys=[],
            tile_size=md.levels[0].tile_size,
            origin=md.levels[0].origin,
        )
        sdata.write_consolidated_metadata()
    return ShapesSpatialIndex(final, root / "tables" / table_name if table_name else None)


class ShapesSpatialIndex:
    """Read shapes (and the rows of their table) by region through the spatial index of RFC 4."""

    def __init__(self, path: str | Path, table_path: str | Path | None = None) -> None:
        self.path = Path(path)
        g = zarr.open_group(self.path, mode="r")
        self.metadata = SpatialIndexMetadata.from_dict(g.attrs["spatial_index"])
        self.margin = np.asarray(self.metadata.extra.get("margin", [0.0, 0.0]), dtype=np.float64)
        self.offsets = g["tile_offsets"]["0"]
        self.table = _ParquetTable(self.path / "shapes.parquet")
        self.table_path = Path(table_path) if table_path is not None else None
        self._var_names: list[str] | None = None

    def row_ranges(self, box_min: Sequence[float], box_max: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(box_min, np.float64) - self.margin
        hi = np.asarray(box_max, np.float64) + self.margin
        tiles = self.metadata.levels[0].tiles_in_box(lo, hi)
        if len(tiles) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        o = self.offsets.get_coordinate_selection(np.concatenate([tiles, tiles + 1]))
        return o[: len(tiles)], o[len(tiles) :]

    def count(self, box_min: Sequence[float], box_max: Sequence[float]) -> int:
        s, e = self.row_ranges(box_min, box_max)
        return int(np.sum(e - s))

    def query(self, box_min: Sequence[float], box_max: Sequence[float], exact: bool = True) -> gpd.GeoDataFrame:
        """Shapes that intersect the box; the column ``__row__`` is their row in the element (and in the table)."""
        s, e = self.row_ranges(box_min, box_max)
        from spatialdata._io.spatial_index._read import _merge_ranges

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
    "reorder_table_to_shapes",
    "write_shapes_with_spatial_index",
]
