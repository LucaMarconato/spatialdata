"""Spatial index of points (RFC 3: tiled and multiscale points), experimental."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
import zarr

from spatialdata._io.spatial_index._build import write_points_spatial_index, write_spatial_index_attrs
from spatialdata._io.spatial_index._metadata import (
    P_COLUMN,
    FeatureIndexLayout,
    LevelGrid,
    SpatialIndexMetadata,
    Storage,
    merge_rule,
)
from spatialdata._io.spatial_index._read import PointsSpatialIndex, read_points_spatial_index
from spatialdata._io.spatial_index._zarr_df import read_zarr_df, zarr_df_columns, zarr_df_n_rows

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = [
    "P_COLUMN",
    "LevelGrid",
    "PointsSpatialIndex",
    "SpatialIndexMetadata",
    "build_points_spatial_index",
    "merge_rule",
    "read_points_spatial_index",
    "read_zarr_df_as_dask",
    "write_points_with_spatial_index",
]


def read_zarr_df_as_dask(group: zarr.Group, rows_per_partition: int = 10_000_000) -> dd.DataFrame:
    """A lazy dask dataframe over a ``df:`` Zarr group."""
    n = zarr_df_n_rows(group)
    columns = zarr_df_columns(group)
    bounds = list(range(0, n, rows_per_partition)) + [n]
    if len(bounds) == 1:
        bounds = [0, 0]
    meta = read_zarr_df(group, 0, 0, columns)
    store_path = Path(group.store_path.store.root) / group.path  # type: ignore[attr-defined]

    def load(bound: tuple[int, int]) -> pd.DataFrame:
        g = zarr.open_group(store_path, mode="r")
        df = read_zarr_df(g, bound[0], bound[1], columns)
        df.index = pd.RangeIndex(bound[0], bound[1])
        return df

    pairs = list(zip(bounds[:-1], bounds[1:], strict=True))
    return dd.from_map(load, pairs, meta=meta, divisions=[*bounds[:-1], bounds[-1] - 1] if n else None)


def write_points_with_spatial_index(
    points: dd.DataFrame,
    path: str | Path,
    *,
    storage: Storage = "parquet",
    overwrite: bool = False,
    **kwargs: Any,
) -> tuple[SpatialIndexMetadata, dict[str, Any]]:
    """Write a points element with the spatial index of RFC 3 to a new element group at ``path``.

    ``path`` is the element group, e.g. ``<store>.zarr/points/<name>``; the element metadata (axes, transformations,
    ``spatialdata_attrs``) is taken from ``points``, as in :meth:`SpatialData.write_element`. The other keyword
    arguments are passed to the builder, see :func:`spatialdata._io.spatial_index._build.write_points_spatial_index`.
    """
    from spatialdata._io._utils import _write_metadata, overwrite_coordinate_transformations_non_raster
    from spatialdata._io.format import PointsFormatV03
    from spatialdata.models import get_axes_names
    from spatialdata.transformations._utils import _get_transformations

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True.")
        shutil.rmtree(path)
    group = zarr.open_group(path, mode="w", zarr_format=3)
    md, stats = write_points_spatial_index(points, group, storage=storage, **kwargs)
    fmt = PointsFormatV03()
    attrs: dict[str, Any] = dict(fmt.attrs_to_dict(points.attrs))
    attrs["version"] = fmt.spatialdata_format_version
    axes = get_axes_names(points)
    _write_metadata(group, group_type="ngff:points", axes=list(axes), attrs=attrs)
    transformations = _get_transformations(points)
    assert transformations is not None
    overwrite_coordinate_transformations_non_raster(group=group, axes=axes, transformations=transformations)
    write_spatial_index_attrs(group, md)
    return md, stats


def build_points_spatial_index(
    sdata: SpatialData,
    element_name: str,
    *,
    new_element_name: str | None = None,
    storage: Storage = "parquet",
    axes: Sequence[str] | None = None,
    feature_keys: Sequence[str] | None = None,
    tile_size: float | Sequence[float] | None = None,
    target_points_per_tile: int = 10_000,
    n_levels: int | None = None,
    feature_index: FeatureIndexLayout = "dense",
    seed: int = 0,
    **kwargs: Any,
) -> PointsSpatialIndex:
    """Build the spatial index of RFC 3 for a points element of a disk-backed SpatialData object.

    The element is written again, sorted and indexed, in the points format 0.3, either in place (default) or as a
    new element ``new_element_name``; the in-memory element is replaced by a lazy view of the new one.

    Parameters
    ----------
    sdata
        A SpatialData object backed by a local Zarr store (``sdata.path``).
    element_name
        Name of the points element.
    new_element_name
        Write the indexed element under this name and keep the original.
    storage
        ``"parquet"`` (default in RFC 3) or ``"zarr"`` (experimental in RFC 3).
    axes
        Axes of the grid; default ``("x", "y")``: 2-D tiles also for 3-D points.
    feature_keys
        Columns with a feature-major table; default: the element's feature key, if any. Pass ``[]`` for none.
    tile_size
        Tile size of level 0; default: chosen so that the densest tile holds about ``target_points_per_tile``.
    target_points_per_tile
        See "Choosing the tile size and the rates" in RFC 3.
    n_levels
        Number of levels; default: merge until a single tile.
    feature_index
        ``"dense"`` (as in RFC 3: ``int64[n_features, N_tiles + 1]`` per band) or ``"sparse"`` (proposed fix: only
        the non-empty tiles of each feature, CSR over features).
    seed
        Seed of the random priority p (ignored if the element already has ``__spatialdata_p__``).
    kwargs
        Passed to the builder (``rows_per_bucket``, ``tmp_dir``, ``n_workers``, ``zarr_chunk_length``, ...).

    Returns
    -------
    The reader of the new index.
    """
    from spatialdata._io.io_points import _read_points

    if sdata.path is None:
        raise ValueError("The SpatialData object must be backed by a Zarr store; call sdata.write() first.")
    points = sdata.points[element_name]
    target_name = new_element_name or element_name
    points_dir = Path(sdata.path) / "points"
    final_path = points_dir / target_name
    tmp_path = points_dir / f"{target_name}.__spatial_index_tmp__"
    write_points_with_spatial_index(
        points,
        tmp_path,
        storage=storage,
        overwrite=True,
        axes=axes,
        feature_keys=feature_keys,
        tile_size=tile_size,
        target_points_per_tile=target_points_per_tile,
        n_levels=n_levels,
        feature_index=feature_index,
        seed=seed,
        **kwargs,
    )
    if final_path.exists():
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)
    sdata.points[target_name] = _read_points(final_path)
    if new_element_name is not None and target_name != element_name:
        # register the new element in the root metadata if needed
        sdata.write_consolidated_metadata()
    return PointsSpatialIndex(final_path)

