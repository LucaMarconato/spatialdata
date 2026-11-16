"""Reading points through the spatial index of RFC 3."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from spatialdata._io.spatial_index._metadata import P_COLUMN, SPATIAL_INDEX_KEY, SpatialIndexMetadata
from spatialdata._io.spatial_index._zarr_df import read_zarr_df, zarr_df_columns, zarr_df_n_rows


def _merge_ranges(starts: np.ndarray, stops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and merge adjacent/overlapping [start, stop) ranges, dropping empty ones."""
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]
    if len(starts) == 0:
        return starts, stops
    order = np.argsort(starts, kind="stable")
    starts, stops = starts[order], stops[order]
    new_run = np.ones(len(starts), dtype=bool)
    new_run[1:] = starts[1:] > np.maximum.accumulate(stops)[:-1]
    run_id = np.cumsum(new_run) - 1
    out_starts = starts[new_run]
    out_stops = np.zeros(len(out_starts), dtype=starts.dtype)
    np.maximum.at(out_stops, run_id, stops)
    return out_starts, out_stops


def _natural_key(p: Path) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


class _ParquetTable:
    """Random access to rows of a (multi-file) Parquet table through its row groups."""

    def __init__(self, path: Path) -> None:
        files = sorted(path.glob("*.parquet"), key=_natural_key) if path.is_dir() else [path]
        self.files = [pq.ParquetFile(f) for f in files]
        rg_rows = []
        self.rg_file: list[int] = []
        self.rg_local: list[int] = []
        for fi, f in enumerate(self.files):
            for ri in range(f.metadata.num_row_groups):
                rg_rows.append(f.metadata.row_group(ri).num_rows)
                self.rg_file.append(fi)
                self.rg_local.append(ri)
        self.rg_start = np.concatenate([[0], np.cumsum(rg_rows)]).astype(np.int64)
        self.n_rows = int(self.rg_start[-1])
        self.bytes_read = 0
        self.row_groups_read = 0

    @property
    def columns(self) -> list[str]:
        return list(self.files[0].schema_arrow.names)

    def read_ranges(self, starts: np.ndarray, stops: np.ndarray, columns: Sequence[str] | None) -> pa.Table:
        starts, stops = _merge_ranges(np.asarray(starts, np.int64), np.asarray(stops, np.int64))
        if len(starts) == 0:
            return self.files[0].schema_arrow.empty_table().select(columns or self.columns)
        first_rg = np.searchsorted(self.rg_start, starts, side="right") - 1
        last_rg = np.searchsorted(self.rg_start, stops - 1, side="right") - 1
        pieces = []
        cache: dict[int, pa.Table] = {}
        for a, b, r0, r1 in zip(starts, stops, first_rg, last_rg, strict=True):
            for r in range(int(r0), int(r1) + 1):
                if r not in cache:
                    f = self.files[self.rg_file[r]]
                    cache[r] = f.read_row_group(self.rg_local[r], columns=list(columns) if columns else None)
                    self.row_groups_read += 1
                    self.bytes_read += sum(
                        f.metadata.row_group(self.rg_local[r]).column(i).total_compressed_size
                        for i in range(f.metadata.num_columns)
                        if columns is None or f.metadata.schema.column(i).name in columns
                    )
                lo = max(int(a), int(self.rg_start[r])) - int(self.rg_start[r])
                hi = min(int(b), int(self.rg_start[r + 1])) - int(self.rg_start[r])
                pieces.append(cache[r].slice(lo, hi - lo))
            # free row groups that later ranges cannot need (ranges are sorted)
            for r in [r for r in cache if r < r1]:
                del cache[r]
        return pa.concat_tables(pieces)

    def categories(self, column: str) -> list[str]:
        col = self.files[0].read_row_group(0, columns=[column]).column(column)
        if not pa.types.is_dictionary(col.type):
            raise ValueError(f"Column {column!r} is not dictionary-encoded.")
        return [str(v) for v in col.chunk(0).dictionary.to_pylist()]


class _ZarrTable:
    def __init__(self, group: zarr.Group) -> None:
        self.group = group
        self.n_rows = zarr_df_n_rows(group)
        self.bytes_read = 0
        self.row_groups_read = 0

    @property
    def columns(self) -> list[str]:
        return zarr_df_columns(self.group)

    def read_ranges(self, starts: np.ndarray, stops: np.ndarray, columns: Sequence[str] | None) -> pd.DataFrame:
        starts, stops = _merge_ranges(np.asarray(starts, np.int64), np.asarray(stops, np.int64))
        if len(starts) == 1:
            return read_zarr_df(self.group, int(starts[0]), int(stops[0]), columns)
        if len(starts) == 0:
            return read_zarr_df(self.group, 0, 0, columns)
        rows = np.concatenate([np.arange(a, b) for a, b in zip(starts, stops, strict=True)])
        return read_zarr_df(self.group, columns=columns, rows=rows)

    def categories(self, column: str) -> list[str]:
        names = self.columns
        node = self.group[f"column_{names.index(column)}"]
        return [str(v) for v in node["enum_values"][:]]


class PointsSpatialIndex:
    """Read the points of an element with a spatial index (RFC 3), by region, level, budget and feature.

    Parameters
    ----------
    path
        Path of the points element group (``.../points/<name>``).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.group = zarr.open_group(self.path, mode="r")
        attrs = self.group.attrs.asdict()
        if SPATIAL_INDEX_KEY not in attrs:
            raise ValueError(f"{self.path} has no {SPATIAL_INDEX_KEY} metadata.")
        self.metadata = SpatialIndexMetadata.from_dict(attrs[SPATIAL_INDEX_KEY])
        self.storage: Literal["parquet", "zarr"] = "zarr" if "spatial" in self.group else "parquet"
        self._tables: dict[str | None, _ParquetTable | _ZarrTable] = {}
        self._offsets_cache: dict[tuple[str | None, int], Any] = {}
        self._categories: dict[str, list[str]] = {}

    # --- tables and offsets ---------------------------------------------------------------------------------------
    def _order_group(self, feature_key: str | None) -> zarr.Group:
        if feature_key is None:
            return self.group if self.storage == "parquet" else self.group["spatial"]
        return self.group["feature"][feature_key]

    def table(self, feature_key: str | None = None) -> _ParquetTable | _ZarrTable:
        if feature_key not in self._tables:
            og = self._order_group(feature_key)
            if self.storage == "parquet":
                rel = "points.parquet" if feature_key is None else f"feature/{feature_key}/points.parquet"
                self._tables[feature_key] = _ParquetTable(self.path / rel)
            else:
                self._tables[feature_key] = _ZarrTable(og["points"])
        return self._tables[feature_key]

    def _offsets(self, feature_key: str | None, level: int) -> Any:
        key = (feature_key, level)
        if key not in self._offsets_cache:
            self._offsets_cache[key] = self._order_group(feature_key)["tile_offsets"][str(level)]
        return self._offsets_cache[key]

    @cached_property
    def n_rows(self) -> int:
        return self.table(None).n_rows

    def categories(self, feature_key: str) -> list[str]:
        """The global categories of a feature column (the codes of the feature index index this list)."""
        if feature_key not in self._categories:
            fg = self.group.get("feature")
            if fg is not None and feature_key in fg and "categories" in fg[feature_key]:
                self._categories[feature_key] = [str(v) for v in fg[feature_key]["categories"][:]]
            else:
                t = self.table(feature_key if feature_key in self.metadata.feature_keys else None)
                self._categories[feature_key] = t.categories(feature_key)
        return self._categories[feature_key]

    @property
    def feature_index_layout(self) -> str:
        return str(self.metadata.extra.get("feature_index_layout", "dense"))

    # --- row ranges -----------------------------------------------------------------------------------------------
    def _box(self, box_min: Sequence[float] | None, box_max: Sequence[float] | None) -> tuple[np.ndarray, np.ndarray]:
        lv0 = self.metadata.levels[0]
        lo = np.asarray(lv0.origin, np.float64) if box_min is None else np.asarray(box_min, np.float64)
        hi = (
            np.asarray(lv0.origin) + np.asarray(lv0.tile_size) * np.asarray(lv0.grid_shape)
            if box_max is None
            else np.asarray(box_max, np.float64)
        )
        return lo, hi

    def tile_ranges(
        self, box_min: Sequence[float] | None = None, box_max: Sequence[float] | None = None, level: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row ranges of the tile-major table for the tiles in view, bands L..level."""
        lo, hi = self._box(box_min, box_max)
        starts, stops = [], []
        for band in range(self.metadata.L, level - 1, -1):
            tiles = self.metadata.levels[band].tiles_in_box(lo, hi)
            if len(tiles) == 0:
                continue
            O = self._offsets(None, band)
            o = O.get_coordinate_selection(np.concatenate([tiles, tiles + 1]))
            starts.append(o[: len(tiles)])
            stops.append(o[len(tiles) :])
        if not starts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(starts), np.concatenate(stops)

    def feature_ranges(
        self,
        feature_key: str,
        codes: Sequence[int],
        box_min: Sequence[float] | None = None,
        box_max: Sequence[float] | None = None,
        level: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row ranges of the feature-major table of ``feature_key`` for the given feature codes and tiles in view."""
        lo, hi = self._box(box_min, box_max)
        codes = np.asarray(sorted(set(int(c) for c in codes)), dtype=np.int64)
        starts, stops = [], []
        full_view = box_min is None and box_max is None
        for band in range(self.metadata.L, level - 1, -1):
            O = self._offsets(feature_key, band)
            if full_view:
                # whole (feature, band) ranges
                if isinstance(O, zarr.Array):
                    n_t = O.shape[1] - 1
                    a = O.get_coordinate_selection((codes, np.zeros_like(codes)))
                    b = O.get_coordinate_selection((codes, np.full_like(codes, n_t)))
                else:
                    ptr = O["feature_ptr"]
                    ro = O["row_offsets"]
                    p0 = ptr.get_coordinate_selection(codes)
                    p1 = ptr.get_coordinate_selection(codes + 1)
                    a = ro.get_coordinate_selection(p0 + codes)
                    b = ro.get_coordinate_selection(p1 + codes)
                starts.append(a)
                stops.append(b)
                continue
            tiles = self.metadata.levels[band].tiles_in_box(lo, hi)
            if len(tiles) == 0:
                continue
            if isinstance(O, zarr.Array):
                rows = np.repeat(codes, len(tiles))
                cols = np.tile(tiles, len(codes))
                starts.append(O.get_coordinate_selection((rows, cols)))
                stops.append(O.get_coordinate_selection((rows, cols + 1)))
            else:
                ptr = O["feature_ptr"]
                tile_ids = O["tile_ids"]
                ro = O["row_offsets"]
                p0 = ptr.get_coordinate_selection(codes)
                p1 = ptr.get_coordinate_selection(codes + 1)
                for c, a, b in zip(codes, p0, p1, strict=True):
                    if b == a:
                        continue
                    t = tile_ids[int(a) : int(b)]
                    hit = np.flatnonzero(np.isin(t, tiles, assume_unique=True))
                    if len(hit) == 0:
                        continue
                    offs = ro[int(a + c) : int(b + c) + 1]
                    starts.append(offs[hit])
                    stops.append(offs[hit + 1])
        if not starts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(starts), np.concatenate(stops)

    # --- queries --------------------------------------------------------------------------------------------------
    def query(
        self,
        box_min: Sequence[float] | None = None,
        box_max: Sequence[float] | None = None,
        *,
        level: int = 0,
        budget: float | None = None,
        features: Sequence[str] | None = None,
        feature_key: str | None = None,
        columns: Sequence[str] | None = None,
        use_feature_index: bool | None = None,
        exact: bool = True,
    ) -> pd.DataFrame:
        """Points in the box ``[box_min, box_max]`` (grid axes order) at a level, optionally with a budget.

        Parameters
        ----------
        level
            Level to read (0 = all the points). Ignored if ``budget`` is given.
        budget
            Read only the points with ``p < budget`` (see "Reading with a budget").
        features
            Keep only these values of ``feature_key``.
        feature_key
            Feature column; default: the first of ``feature_keys`` or, if none, the element's feature key.
        use_feature_index
            Use the feature-major table (if any); default: when ``features`` is given.
        exact
            Drop the points of the tiles in view that are outside the box.
        """
        md = self.metadata
        if budget is not None:
            level = md.coarsest_level_for_budget(budget)
        if feature_key is None and md.feature_keys:
            feature_key = md.feature_keys[0]
        table_cols = self.table(None).columns
        cols = list(table_cols) if columns is None else list(columns)
        need = list(dict.fromkeys(cols + list(md.axes) + ([P_COLUMN] if budget is not None else [])))
        if features is not None:
            assert feature_key is not None
            need = list(dict.fromkeys(need + [feature_key]))
        if use_feature_index is None:
            use_feature_index = features is not None and feature_key in md.feature_keys
        if use_feature_index:
            assert features is not None and feature_key is not None
            cats = self.categories(feature_key)
            lut = {c: i for i, c in enumerate(cats)}
            codes = [lut[f] for f in features if f in lut]
            starts, stops = self.feature_ranges(feature_key, codes, box_min, box_max, level)
            data = self.table(feature_key).read_ranges(starts, stops, need)
        else:
            starts, stops = self.tile_ranges(box_min, box_max, level)
            data = self.table(None).read_ranges(starts, stops, need)
        df = data.to_pandas() if isinstance(data, pa.Table) else data
        mask = np.ones(len(df), dtype=bool)
        if budget is not None:
            mask &= df[P_COLUMN].to_numpy() < budget
        if features is not None and not use_feature_index:
            mask &= df[feature_key].isin(list(features)).to_numpy()
        if exact and (box_min is not None or box_max is not None):
            lo, hi = self._box(box_min, box_max)
            c = df[list(md.axes)].to_numpy()
            mask &= np.all((c >= lo) & (c <= hi), axis=1)
        if not mask.all():
            df = df[mask]
        return df[cols].reset_index(drop=True)


def read_points_spatial_index(path: str | Path) -> PointsSpatialIndex:
    """Open the spatial index (RFC 3) of a points element stored at ``path``."""
    return PointsSpatialIndex(path)
