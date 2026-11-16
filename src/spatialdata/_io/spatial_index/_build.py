"""Build the spatial index of RFC 3 (tiled and multiscale points) out of core.

The points are read partition by partition (a dask dataframe), so the input never has to fit in memory:

1. pass A: bounding box and a fine histogram of the coordinates, to choose the grid (origin, tile size, levels);
2. pass B: the priority p of each point, its band and its tile; the number of points per (band, tile), which gives
   the tile index, and per (feature, band), which plans the feature-major order;
3. pass C: each partition is scattered to buckets, contiguous ranges of rows of each order, in temporary Parquet files;
4. pass D: each bucket is sorted in memory and written to the final table (Parquet or Zarr), with its offsets.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from spatialdata._io.spatial_index._metadata import (
    P_BITS,
    P_COLUMN,
    SPATIAL_INDEX_CONVENTION,
    SPATIAL_INDEX_KEY,
    FeatureIndexLayout,
    LevelGrid,
    SpatialIndexMetadata,
    Storage,
    merge_rule,
)
from spatialdata._io.spatial_index._zarr_df import ZarrDataFrameWriter

logger = logging.getLogger(__name__)

_G = "__sd_g__"  # temporary column: global (band, tile) id
_PB = "__sd_pbits__"  # temporary column: bits of p (float32 viewed as uint32), sorts like p

# above this number of entries the dense feature index is refused unless forced (8 GB as int64)
DENSE_FEATURE_INDEX_MAX_ENTRIES = 1_000_000_000


@dataclass
class _Plan:
    metadata: SpatialIndexMetadata
    grid_axes: list[str]
    feature_keys: list[str]
    categories: dict[str, list[str]]  # global categories of every categorical column
    n_rows: int
    band_base: np.ndarray  # first global (band, tile) id of each band, indexed by rank = L - band (coarsest first)
    counts_g: np.ndarray  # number of points per global (band, tile) id
    counts_fr: dict[str, np.ndarray]  # per feature key: points per (feature code, rank), shape (n_features, L + 1)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def L(self) -> int:
        return self.metadata.L


def _ensure_dask(points: Any) -> dd.DataFrame:
    if isinstance(points, pd.DataFrame):
        return dd.from_pandas(points, npartitions=max(1, len(points) // 5_000_000))
    if not isinstance(points, dd.DataFrame):
        raise TypeError(f"Expected a dask or pandas DataFrame, got {type(points).__name__}.")
    return points


def _partition(points: dd.DataFrame, i: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
    part = points.get_partition(i)
    if columns is not None:
        part = part[list(columns)]
    df = part.compute(scheduler="sync")
    assert isinstance(df, pd.DataFrame)
    return df.reset_index(drop=True)


def _p_bits(points_part: pd.DataFrame, seed: int, part_index: int) -> np.ndarray:
    """The bits of the float32 p of each point (uint32, sorts like p), reused if the column exists."""
    if P_COLUMN in points_part.columns:
        p = points_part[P_COLUMN].to_numpy(dtype=np.float32)
        if np.any(p < 0) or np.any(p >= 1):
            raise ValueError(f"Column {P_COLUMN} must be in [0, 1).")
        return p.view(np.uint32)
    rng = np.random.default_rng([seed, part_index])
    p_int = rng.integers(0, 2**P_BITS, size=len(points_part), dtype=np.uint32)
    return (p_int.astype(np.float32) * np.float32(2.0**-P_BITS)).view(np.uint32)


def _bands_and_tiles(
    md: SpatialIndexMetadata, coords: np.ndarray, p_bits: np.ndarray, band_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rank (L - band, coarsest first) and global (band, tile) id of each point."""
    p = p_bits.view(np.float32)
    band = md.band_of(p)
    g = np.empty(len(p), dtype=np.int64)
    for level in range(md.n_levels):
        mask = band == level
        if mask.any():
            rank = md.L - level
            g[mask] = band_base[rank] + md.levels[level].tile_ids(coords[mask])
    return md.L - band, g


def _choose_level0(
    points: dd.DataFrame,
    grid_axes: list[str],
    tile_size: float | Sequence[float] | None,
    origin: Sequence[float] | None,
    target_points_per_tile: int,
    n_workers: int,
) -> tuple[LevelGrid, int, dict[str, float]]:
    """Pass A: bounding box, then (if no tile size is given) a fine histogram to pick an isotropic tile size."""
    d = len(grid_axes)
    nparts = points.npartitions

    def bbox(i: int) -> tuple[np.ndarray, np.ndarray, int]:
        c = _partition(points, i, grid_axes).to_numpy(dtype=np.float64)
        if len(c) == 0:
            return np.full(d, np.inf), np.full(d, -np.inf), 0
        return c.min(axis=0), c.max(axis=0), len(c)

    with ThreadPoolExecutor(n_workers) as ex:
        res = list(ex.map(bbox, range(nparts)))
    lo = np.min([r[0] for r in res], axis=0)
    hi = np.max([r[1] for r in res], axis=0)
    n = int(sum(r[2] for r in res))
    o = np.floor(lo) if origin is None else np.asarray(origin, dtype=np.float64)
    extent = np.maximum(hi - o, 1e-9)
    info: dict[str, float] = {"n_points": n}

    if tile_size is None:
        # a fine histogram with bins h = (uniform-density tile size) / 16, then the smallest multiple k of h whose
        # densest tile holds at most the target
        s_uniform = (float(np.prod(extent)) * target_points_per_tile / max(n, 1)) ** (1.0 / d)
        h = s_uniform / 16.0
        bins = np.maximum(np.ceil(extent / h).astype(np.int64), 1)
        while np.prod(bins) > 2e8:  # keep the histogram small
            h *= 2
            bins = np.maximum(np.ceil(extent / h).astype(np.int64), 1)

        def hist(i: int) -> np.ndarray:
            c = _partition(points, i, grid_axes).to_numpy(dtype=np.float64)
            t = np.clip(np.floor((c - o) / h).astype(np.int64), 0, bins - 1)
            return np.bincount(np.ravel_multi_index(tuple(t.T), tuple(bins)), minlength=int(np.prod(bins)))

        H = np.zeros(int(np.prod(bins)), dtype=np.int64)
        with ThreadPoolExecutor(n_workers) as ex:
            for hh in ex.map(hist, range(nparts)):
                H += hh
        H = H.reshape(tuple(bins))
        def densest(k: int) -> int:
            Hp = np.pad(H, [(0, (-b) % k) for b in bins])
            shape: list[int] = []
            for b in Hp.shape:
                shape += [b // k, k]
            return int(Hp.reshape(shape).sum(axis=tuple(range(1, 2 * d, 2))).max())

        best_k, best_max = 1, densest(1)
        for k in range(2, 4 * 16 + 1):
            m = densest(k)
            if m > target_points_per_tile:
                break
            best_k, best_max = k, m
        s = np.full(d, best_k * h)
        info["max_points_per_tile_estimate"] = float(best_max)
    else:
        s = np.broadcast_to(np.asarray(tile_size, dtype=np.float64), (d,)).copy()
    grid_shape = tuple(int(v) for v in np.floor(extent / s).astype(np.int64) + 1)
    return LevelGrid(tuple(o.tolist()), tuple(s.tolist()), grid_shape, 1.0), n, info


def _global_categories(points: dd.DataFrame, columns: Sequence[str]) -> tuple[dd.DataFrame, dict[str, list[str]]]:
    cats: dict[str, list[str]] = {}
    for c in columns:
        dtype = points[c].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            if not points[c].cat.known or len(dtype.categories) == 0:
                points[c] = points[c].cat.as_unknown().cat.as_known()
            cats[c] = [str(v) for v in points[c].cat.categories]
        else:
            uniques = points[c].dropna().unique().compute()
            cats[c] = sorted(str(v) for v in uniques)
            points[c] = points[c].astype(pd.CategoricalDtype(cats[c]))
    for c in points.columns:
        if c not in cats and isinstance(points[c].dtype, pd.CategoricalDtype):
            if not points[c].cat.known or len(points[c].dtype.categories) == 0:
                points[c] = points[c].cat.as_unknown().cat.as_known()
            cats[c] = [str(v) for v in points[c].cat.categories]
    return points, cats


def _codes(df: pd.DataFrame, column: str, categories: list[str]) -> np.ndarray:
    s = df[column]
    if not isinstance(s.dtype, pd.CategoricalDtype) or list(s.cat.categories) != categories:
        s = s.astype(str).astype(pd.CategoricalDtype(categories))
    codes = s.cat.codes.to_numpy()
    if np.any(codes < 0):
        raise ValueError(f"Missing values in the feature column {column!r} are not supported.")
    return codes.astype(np.int64)


def _plan(
    points: dd.DataFrame,
    grid_axes: list[str],
    feature_keys: list[str],
    tile_size: float | Sequence[float] | None,
    origin: Sequence[float] | None,
    target_points_per_tile: int,
    n_levels: int | None,
    P: Sequence[float] | None,
    seed: int,
    n_workers: int,
) -> tuple[dd.DataFrame, _Plan]:
    t0 = time.perf_counter()
    points, categories = _global_categories(points, feature_keys)
    t1 = time.perf_counter()
    level0, n, info = _choose_level0(points, grid_axes, tile_size, origin, target_points_per_tile, n_workers)
    levels = merge_rule(level0, n_levels)
    if P is not None:
        if len(P) != len(levels):
            raise ValueError(f"P must have one entry per level ({len(levels)}), got {len(P)}.")
        levels = [LevelGrid(lv.origin, lv.tile_size, lv.grid_shape, float(p)) for lv, p in zip(levels, P, strict=True)]
    md = SpatialIndexMetadata(
        axes=tuple(grid_axes), levels=tuple(levels), feature_keys=tuple(feature_keys), extra={"seed": seed}
    )
    md.validate()
    t2 = time.perf_counter()

    L = md.L
    # rank r = L - band; global id of (band, tile) = band_base[r] + tile id, so that ids sort coarsest band first
    n_tiles_by_rank = [md.levels[L - r].n_tiles for r in range(L + 1)]
    band_base = np.concatenate([[0], np.cumsum(n_tiles_by_rank)]).astype(np.int64)
    total = int(band_base[-1])
    columns = list(dict.fromkeys(list(grid_axes) + feature_keys + ([P_COLUMN] if P_COLUMN in points.columns else [])))

    def count(i: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        df = _partition(points, i, columns)
        coords = df[grid_axes].to_numpy(dtype=np.float64)
        pb = _p_bits(df, seed, i)
        rank, g = _bands_and_tiles(md, coords, pb, band_base)
        cg = np.bincount(g, minlength=total)
        cfr = {}
        for fk in feature_keys:
            codes = _codes(df, fk, categories[fk])
            cfr[fk] = np.bincount(codes * (L + 1) + rank, minlength=len(categories[fk]) * (L + 1))
        return cg, cfr

    counts_g = np.zeros(total, dtype=np.int64)
    counts_fr = {fk: np.zeros(len(categories[fk]) * (L + 1), dtype=np.int64) for fk in feature_keys}
    with ThreadPoolExecutor(n_workers) as ex:
        for cg, cfr in ex.map(count, range(points.npartitions)):
            counts_g += cg
            for fk in feature_keys:
                counts_fr[fk] += cfr[fk]
    t3 = time.perf_counter()
    plan = _Plan(
        metadata=md,
        grid_axes=grid_axes,
        feature_keys=feature_keys,
        categories=categories,
        n_rows=int(counts_g.sum()),
        band_base=band_base,
        counts_g=counts_g,
        counts_fr={fk: c.reshape(len(categories[fk]), L + 1) for fk, c in counts_fr.items()},
        timings={"categories": t1 - t0, "grid": t2 - t1, "counts": t3 - t2, **{f"grid_{k}": v for k, v in info.items()}},
    )
    assert plan.n_rows == n
    return points, plan


def _buckets(sizes: np.ndarray, rows_per_bucket: int) -> np.ndarray:
    """Split consecutive units with the given sizes into buckets of about rows_per_bucket rows.

    Returns the bucket of each unit (non-decreasing).
    """
    csum = np.cumsum(sizes)
    start = csum - sizes
    return (start // max(rows_per_bucket, 1)).astype(np.int64)


def _row_groups(boundaries: np.ndarray, n_rows: int, min_rows: int, max_rows: int) -> list[tuple[int, int]]:
    """Row groups following the given unit boundaries (sorted starts of units, first is 0).

    Small units share a row group until it holds at least min_rows rows; a unit larger than max_rows is split.
    """
    starts = list(boundaries.tolist()) + [n_rows]
    groups: list[tuple[int, int]] = []
    cur = 0
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        if b - a > max_rows:
            if a > cur:
                groups.append((cur, a))
            for s in range(a, b, max_rows):
                groups.append((s, min(b, s + max_rows)))
            cur = b
        elif b - cur >= min_rows:
            groups.append((cur, b))
            cur = b
    if cur < n_rows:
        groups.append((cur, n_rows))
    return groups


class _OrderWriter:
    """Writes one order (the tile-major table or one feature-major table) bucket by bucket."""

    def __init__(
        self,
        root: zarr.Group,
        table_dir: Path,
        storage: Storage,
        schema: pa.Schema,
        plan: _Plan,
        feature_key: str | None,
        feature_index: FeatureIndexLayout,
        row_group_min_rows: int,
        row_group_max_rows: int,
        zarr_chunk_length: int,
        zarr_shard_length: int | None,
        parquet_compression: str,
    ) -> None:
        self.storage = storage
        self.schema = schema
        self.plan = plan
        self.feature_key = feature_key
        self.feature_index = feature_index
        self.row_group_min_rows = row_group_min_rows
        self.row_group_max_rows = row_group_max_rows
        self.parquet_compression = parquet_compression
        self.root = root
        self.table_dir = table_dir
        self.compact_dictionaries = True
        self._lock = threading.Lock()
        self.offsets_group = root.require_group("tile_offsets")
        md = plan.metadata
        if storage == "zarr":
            self.df_writer: ZarrDataFrameWriter | None = ZarrDataFrameWriter(
                root.require_group("points"),
                schema,
                plan.n_rows,
                zarr_chunk_length,
                zarr_shard_length,
                plan.categories,
            )
        else:
            self.df_writer = None
            table_dir.mkdir(parents=True, exist_ok=True)
        # the sparse feature index, accumulated per band
        self._sparse: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {lv: [] for lv in range(md.n_levels)}
        self._dense: dict[int, zarr.Array] = {}
        if feature_key is not None and feature_index == "dense":
            n_features = len(plan.categories[feature_key])
            for lv in range(md.n_levels):
                n_t = md.levels[lv].n_tiles
                self._dense[lv] = self.offsets_group.create_array(
                    str(lv),
                    shape=(n_features, n_t + 1),
                    dtype=np.int64,
                    chunks=(1, min(n_t + 1, 65536)),
                    compressors=_offset_compressors(),
                    filters=_offset_filters(),
                )

    def write_tile_index(self) -> None:
        """Tile-major: offsets of every band follow from the counts."""
        md = self.plan.metadata
        L = md.L
        row = 0
        for r in range(L + 1):
            level = L - r
            base = self.plan.band_base[r]
            c = self.plan.counts_g[base : base + md.levels[level].n_tiles]
            offs = row + np.concatenate([[0], np.cumsum(c)]).astype(np.int64)
            n_t = md.levels[level].n_tiles
            self.offsets_group.create_array(
                str(level),
                data=offs,
                chunks=(min(n_t + 1, 65536),),
                compressors=_offset_compressors(),
                filters=_offset_filters(),
            )
            row = int(offs[-1])

    def write_bucket(self, start: int, table: pa.Table, g: np.ndarray, codes: np.ndarray | None) -> None:
        """Write a sorted bucket that starts at global row ``start``; g and codes are per row (sorted)."""
        n = table.num_rows
        if n == 0:
            return
        unit = g
        if codes is not None:
            # the row groups of the feature-major table follow (feature, band)
            rank = np.searchsorted(self.plan.band_base, g, side="right") - 1
            unit = codes * (self.plan.L + 1) + rank
        boundaries = np.flatnonzero(np.diff(unit, prepend=unit[0] - 1))
        if self.storage == "parquet":
            groups = _row_groups(boundaries, n, self.row_group_min_rows, self.row_group_max_rows)
            path = self.table_dir / f"part.{start:012d}.parquet"
            with pq.ParquetWriter(path, self.schema, compression=self.parquet_compression) as w:
                for a, b in groups:
                    piece = table.slice(a, b - a)
                    if self.compact_dictionaries:
                        piece = _compact_dictionaries(piece)
                    w.write_table(piece, row_group_size=b - a)
        else:
            assert self.df_writer is not None
            # buckets may share a (sharded) chunk: a partial write is a read-modify-write, so serialize the writes
            with self._lock:
                self.df_writer.write(start, table)
        if codes is not None:
            self._feature_index(start, g, codes)

    def _feature_index(self, start: int, g: np.ndarray, codes: np.ndarray) -> None:
        md = self.plan.metadata
        L = md.L
        rank = np.searchsorted(self.plan.band_base, g, side="right") - 1
        fr = codes * (L + 1) + rank
        # one entry per (feature, rank, tile) present
        key_change = np.flatnonzero(np.diff(g, prepend=-1) | np.diff(fr, prepend=-1))
        ent_fr = fr[key_change]
        ent_tile = g[key_change] - self.plan.band_base[rank[key_change]]
        ent_start = start + key_change
        ent_end = np.append(ent_start[1:], start + len(g))
        # ent_fr is non-decreasing (rows sorted by feature, then band): one group of entries per (feature, band)
        last_of_group = np.append(ent_fr[1:] != ent_fr[:-1], True)
        first_of_group = np.append(True, ent_fr[1:] != ent_fr[:-1])
        feature_all, r_all = np.divmod(ent_fr, L + 1)
        level_all = L - r_all
        if self.feature_index == "sparse":
            for level in np.unique(level_all):
                sel = level_all == level
                self._sparse[int(level)].append((feature_all[sel], ent_tile[sel], ent_start[sel]))
                # the end of each (feature, band) range, marked with tile -1
                end = sel & last_of_group
                self._sparse[int(level)].append(
                    (feature_all[end], np.full(int(end.sum()), -1, dtype=np.int64), ent_end[end])
                )
            return
        group_starts = np.flatnonzero(first_of_group)
        group_ends = np.append(group_starts[1:], len(ent_fr))
        for a, b in zip(group_starts, group_ends, strict=True):
            feature, level = int(feature_all[a]), int(level_all[a])
            n_t = md.levels[level].n_tiles
            counts = np.zeros(n_t, dtype=np.int64)
            counts[ent_tile[a:b]] = ent_end[a:b] - ent_start[a:b]
            row = ent_start[a] + np.concatenate([[0], np.cumsum(counts)])
            self._dense[level][feature, :] = row
            self._dense_written.add((feature, level))

    _dense_written: set[tuple[int, int]]

    def finalize_feature_index(self) -> None:
        md = self.plan.metadata
        L = md.L
        fk = self.feature_key
        assert fk is not None
        n_features = len(self.plan.categories[fk])
        # start row of every (feature, rank) range, also the empty ones
        cfr = self.plan.counts_fr[fk]
        fr_start = (np.cumsum(cfr.ravel()) - cfr.ravel()).reshape(cfr.shape)
        if self.feature_index == "dense":
            for level in range(md.n_levels):
                r = L - level
                n_t = md.levels[level].n_tiles
                for feature in range(n_features):
                    if (feature, level) not in self._dense_written:
                        self._dense[level][feature, :] = np.full(n_t + 1, fr_start[feature, r], dtype=np.int64)
            return
        # sparse (proposed fix, not in RFC 3): per band, CSR over features of the non-empty tiles
        for level in range(md.n_levels):
            r = L - level
            parts = self._sparse[level]
            if parts:
                feats = np.concatenate([p[0] for p in parts])
                tiles = np.concatenate([p[1] for p in parts])
                rows = np.concatenate([p[2] for p in parts])
            else:
                feats = tiles = rows = np.empty(0, dtype=np.int64)
            # order by feature, then the end markers (-1) last, then tile
            order = np.lexsort((tiles, tiles < 0, feats))
            feats, tiles, rows = feats[order], tiles[order], rows[order]
            is_end = tiles < 0
            nnz_per_feature = np.bincount(feats[~is_end], minlength=n_features)
            feature_ptr = np.concatenate([[0], np.cumsum(nnz_per_feature)]).astype(np.int64)
            # row_offsets: for feature f, entries feature_ptr[f] + f ... feature_ptr[f + 1] + f (inclusive): the
            # start of each non-empty tile, then the end of the (feature, band) range
            row_offsets = np.empty(len(tiles) - int(is_end.sum()) + n_features, dtype=np.int64)
            has_any = np.zeros(n_features, dtype=bool)
            has_any[feats[is_end]] = True
            # features without points in this band: a single entry, the start of their (empty) range
            pos_end = feature_ptr[1:] + np.arange(n_features)
            row_offsets[pos_end] = fr_start[:, r]
            row_offsets[pos_end[feats[is_end]]] = rows[is_end]
            mask = np.ones(len(row_offsets), dtype=bool)
            mask[pos_end] = False
            row_offsets[mask] = rows[~is_end]
            g = self.offsets_group.require_group(str(level))
            g.attrs["layout"] = "sparse-csr"
            for name, data in (
                ("feature_ptr", feature_ptr),
                ("tile_ids", tiles[~is_end]),
                ("row_offsets", row_offsets),
            ):
                g.create_array(
                    name,
                    data=data,
                    chunks=(max(1, min(len(data), 65536)),),
                    compressors=_offset_compressors(),
                    filters=_offset_filters(),
                )


def _compact_dictionaries(table: pa.Table) -> pa.Table:
    """Keep in the dictionary of each categorical column only the values used in this row group.

    Parquet stores the dictionary of a column once per row group: with small row groups (about one tile) and
    tens of thousands of features, a full dictionary per row group would dominate the size of the file. The global
    categories (whose codes the feature index uses) are stored once, in ``feature/<key>/categories``.
    """
    cols = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type) and table.num_rows:
            arr = col.combine_chunks()
            idx = arr.indices.to_numpy(zero_copy_only=False)
            used, remap = np.unique(idx, return_inverse=True)
            arr = pa.DictionaryArray.from_arrays(
                pa.array(remap.astype(idx.dtype), type=field.type.index_type), arr.dictionary.take(pa.array(used))
            )
            col = pa.chunked_array([arr], type=field.type)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=table.schema)


def _offset_compressors() -> Any:
    from zarr.codecs import BloscCodec

    return [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")]


def _offset_filters() -> Any:
    from zarr.codecs.numcodecs import Delta

    return [Delta(dtype="<i8")]


def _output_schema(points: dd.DataFrame, plan: _Plan, drop: Sequence[str]) -> pa.Schema:
    meta = points._meta.reset_index(drop=True)
    meta = meta[[c for c in meta.columns if c not in drop and c != P_COLUMN]]
    for c, cats in plan.categories.items():
        if c in meta.columns:
            meta[c] = meta[c].astype(pd.CategoricalDtype(cats))
    meta[P_COLUMN] = pd.Series([], dtype=np.float32)
    first = [*plan.grid_axes]
    first += [c for c in ("x", "y", "z") if c in meta.columns and c not in first]
    first += [*plan.feature_keys, P_COLUMN]
    order = list(dict.fromkeys(first + [n for n in meta.columns if n not in first]))
    # keep the pandas metadata, so that readers such as dask restore the categories
    return pa.Schema.from_pandas(meta[order], preserve_index=False)


def _to_table(df: pd.DataFrame, schema: pa.Schema, plan: _Plan, p_bits: np.ndarray) -> pa.Table:
    df = df.copy()
    for c, cats in plan.categories.items():
        if c in df.columns:
            if not isinstance(df[c].dtype, pd.CategoricalDtype) or list(df[c].cat.categories) != cats:
                df[c] = df[c].astype(str).astype(pd.CategoricalDtype(cats))
    df[P_COLUMN] = p_bits.view(np.float32)
    t = pa.Table.from_pandas(df[schema.names], preserve_index=False)
    return t.cast(schema)


def write_points_spatial_index(
    points: dd.DataFrame | pd.DataFrame,
    group: zarr.Group,
    *,
    storage: Storage = "parquet",
    axes: Sequence[str] | None = None,
    feature_keys: Sequence[str] | None = None,
    tile_size: float | Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    target_points_per_tile: int = 10_000,
    n_levels: int | None = None,
    P: Sequence[float] | None = None,
    seed: int = 0,
    feature_index: FeatureIndexLayout = "dense",
    force_dense_feature_index: bool = False,
    rows_per_bucket: int = 20_000_000,
    row_group_min_rows: int | None = None,
    row_group_max_rows: int = 1_048_576,
    zarr_chunk_length: int | None = None,
    zarr_shard_length: int | None = 2_097_152,
    parquet_compression: str = "zstd",
    tmp_dir: str | Path | None = None,
    n_workers: int | None = None,
    drop_columns: Sequence[str] = ("__null_dask_index__",),
    progress: Callable[[str], None] | None = None,
) -> tuple[SpatialIndexMetadata, dict[str, Any]]:
    """Write the tables and offsets of the spatial index of RFC 3 into an (empty) element group.

    Only the data of the index is written (``points.parquet`` / ``spatial/``, ``tile_offsets``, ``feature/``); the
    caller writes the element metadata (transformations, ``spatialdata_attrs``) and the ``spatial_index`` attributes,
    which this function returns.

    Returns
    -------
    The metadata of the index and a dictionary of statistics (timings, sizes).
    """
    log = progress or (lambda msg: logger.info(msg))
    points = _ensure_dask(points)
    n_workers = n_workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    if axes is None:
        axes = [a for a in ("x", "y") if a in points.columns]
    grid_axes = list(axes)
    if feature_keys is None:
        fk = points.attrs.get("spatialdata_attrs", {}).get("feature_key")
        feature_keys = [fk] if fk is not None else []
    feature_keys = list(feature_keys)
    if row_group_min_rows is None:
        row_group_min_rows = target_points_per_tile
    if zarr_chunk_length is None:
        zarr_chunk_length = target_points_per_tile

    t_start = time.perf_counter()
    log("planning the grid and counting the points per tile")
    points, plan = _plan(
        points, grid_axes, feature_keys, tile_size, origin, target_points_per_tile, n_levels, P, seed, n_workers
    )
    md = plan.metadata
    stats: dict[str, Any] = {"timings": dict(plan.timings), "n_rows": plan.n_rows, "n_levels": md.n_levels}
    for fk in feature_keys:
        n_entries = len(plan.categories[fk]) * sum(lv.n_tiles + 1 for lv in md.levels)
        stats[f"dense_feature_index_entries[{fk}]"] = n_entries
        if feature_index == "dense" and n_entries > DENSE_FEATURE_INDEX_MAX_ENTRIES and not force_dense_feature_index:
            raise ValueError(
                f"The dense feature index of {fk!r} would have {n_entries:,} entries "
                f"({len(plan.categories[fk]):,} features x {sum(lv.n_tiles for lv in md.levels):,} tiles); use "
                "feature_index='sparse' or force_dense_feature_index=True."
            )
    log(
        f"{plan.n_rows:,} points, {md.n_levels} levels, level 0 grid {md.levels[0].grid_shape} of tile size "
        f"{md.levels[0].tile_size}"
    )

    schema = _output_schema(points, plan, drop_columns)
    in_columns = [n for n in schema.names if n != P_COLUMN] + ([P_COLUMN] if P_COLUMN in points.columns else [])

    # bucket plans: tile-major by global (band, tile) id; feature-major by (feature, band)
    bucket_of_g = _buckets(plan.counts_g, rows_per_bucket)
    n_buckets_t = int(bucket_of_g.max()) + 1 if len(bucket_of_g) else 1
    bucket_start_t = np.zeros(n_buckets_t + 1, dtype=np.int64)
    np.add.at(bucket_start_t, bucket_of_g + 1, plan.counts_g)
    bucket_start_t = np.cumsum(bucket_start_t)
    bucket_of_fr: dict[str, np.ndarray] = {}
    bucket_start_f: dict[str, np.ndarray] = {}
    for fk in feature_keys:
        sizes = plan.counts_fr[fk].ravel()
        b = _buckets(sizes, rows_per_bucket)
        bucket_of_fr[fk] = b
        nb = int(b.max()) + 1
        s = np.zeros(nb + 1, dtype=np.int64)
        np.add.at(s, b + 1, sizes)
        bucket_start_f[fk] = np.cumsum(s)

    tmp_root = Path(tempfile.mkdtemp(prefix="spatial_index_", dir=tmp_dir))
    stats["tmp_dir"] = str(tmp_root)
    try:
        # pass C: scatter
        t0 = time.perf_counter()
        log(f"scattering {points.npartitions} partitions to buckets in {tmp_root}")
        orders = ["tile"] + [f"feature/{fk}" for fk in feature_keys]

        def scatter(i: int) -> int:
            df = _partition(points, i, in_columns)
            pb = _p_bits(df, seed, i)
            coords = df[grid_axes].to_numpy(dtype=np.float64)
            rank, g = _bands_and_tiles(md, coords, pb, plan.band_base)
            table = _to_table(df.drop(columns=[P_COLUMN], errors="ignore"), schema, plan, pb)
            table = table.append_column(_G, pa.array(g)).append_column(_PB, pa.array(pb))
            for order in orders:
                if order == "tile":
                    bucket = bucket_of_g[g]
                else:
                    fk = order.split("/", 1)[1]
                    codes = _codes(df, fk, plan.categories[fk])
                    bucket = bucket_of_fr[fk][codes * (md.L + 1) + rank]
                sort = np.argsort(bucket, kind="stable")
                b_sorted = bucket[sort]
                cuts = np.flatnonzero(np.diff(b_sorted, prepend=-1))
                ends = np.append(cuts[1:], len(b_sorted))
                tsorted = table.take(pa.array(sort))
                for a, e in zip(cuts, ends, strict=True):
                    d = tmp_root / order / f"bucket_{int(b_sorted[a]):06d}"
                    d.mkdir(parents=True, exist_ok=True)
                    pq.write_table(tsorted.slice(a, e - a), d / f"part_{i:06d}.parquet", compression="lz4")
            return len(df)

        with ThreadPoolExecutor(n_workers) as ex:
            n_scattered = sum(ex.map(scatter, range(points.npartitions)))
        assert n_scattered == plan.n_rows
        stats["timings"]["scatter"] = time.perf_counter() - t0
        stats["tmp_bytes"] = sum(f.stat().st_size for f in tmp_root.rglob("*.parquet"))

        # pass D: sort each bucket and write
        root_tile = group if storage == "parquet" else group.require_group("spatial")
        writers: dict[str, _OrderWriter] = {}
        for order in orders:
            fk_or_none = None if order == "tile" else order.split("/", 1)[1]
            if storage == "parquet":
                og = group if order == "tile" else group.require_group(order)
            else:
                og = root_tile if order == "tile" else group.require_group(order)
            table_dir = _local_path(og) / "points.parquet"
            w = _OrderWriter(
                og,
                table_dir,
                storage,
                schema,
                plan,
                fk_or_none,
                feature_index,
                row_group_min_rows,
                row_group_max_rows,
                zarr_chunk_length,
                zarr_shard_length,
                parquet_compression,
            )
            w._dense_written = set()
            writers[order] = w
        writers["tile"].write_tile_index()
        # the global category list of each feature column, whose codes the feature index uses
        for fk in feature_keys:
            fg = group.require_group("feature").require_group(fk)
            cats = np.asarray(plan.categories[fk], dtype=str)
            fg.create_array("categories", data=cats, chunks=(max(1, len(cats)),), overwrite=True)

        t0 = time.perf_counter()
        for order in orders:
            fk = None if order == "tile" else order.split("/", 1)[1]
            starts = bucket_start_t if fk is None else bucket_start_f[fk]
            n_b = len(starts) - 1
            log(f"sorting and writing {n_b} buckets of the {order} order")

            def sort_write(b: int, order: str = order, fk: str | None = fk, starts: np.ndarray = starts) -> None:
                d = tmp_root / order / f"bucket_{b:06d}"
                if not d.exists():
                    return
                files = sorted(d.glob("*.parquet"))
                t = pa.concat_tables([pq.read_table(f) for f in files])
                g = t.column(_G).to_numpy()
                pb = t.column(_PB).to_numpy()
                if fk is None:
                    idx = np.lexsort((pb, g))
                    codes = None
                else:
                    codes_all = t.column(fk).combine_chunks().indices.to_numpy().astype(np.int64)
                    idx = np.lexsort((pb, g, codes_all))
                    codes = codes_all[idx]
                t = t.take(pa.array(idx)).drop_columns([_G, _PB]).cast(schema)
                assert t.num_rows == starts[b + 1] - starts[b]
                writers[order].write_bucket(int(starts[b]), t, g[idx], codes)
                shutil.rmtree(d)

            # dense feature indices write rows of a shared zarr array: one bucket at a time is safest
            workers = 1 if (fk is not None and feature_index == "dense") else max(1, n_workers // 2)
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(sort_write, range(n_b)))
            if fk is not None:
                writers[order].finalize_feature_index()
        stats["timings"]["sort_write"] = time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    stats["timings"]["total"] = time.perf_counter() - t_start
    if feature_index == "sparse" and feature_keys:
        md = SpatialIndexMetadata(
            md.axes, md.levels, md.feature_keys, md.version, {**md.extra, "feature_index_layout": "sparse-csr"}
        )
    return md, stats


def _local_path(group: zarr.Group) -> Path:
    from zarr.storage import LocalStore

    store = group.store_path.store
    if not isinstance(store, LocalStore):
        raise TypeError(f"Writing a spatial index requires a local zarr store, got {type(store).__name__}.")
    return Path(store.root) / group.path


def write_spatial_index_attrs(group: zarr.Group, md: SpatialIndexMetadata) -> None:
    attrs = group.attrs.asdict()
    conventions = [c for c in attrs.get("zarr_conventions", []) if c.get("name") != SPATIAL_INDEX_KEY]
    conventions.append(SPATIAL_INDEX_CONVENTION)
    group.attrs.update({"zarr_conventions": conventions, SPATIAL_INDEX_KEY: md.to_dict()})


def summarize(md: SpatialIndexMetadata) -> str:
    return json.dumps(md.to_dict(), indent=2)


__all__ = ["write_points_spatial_index", "write_spatial_index_attrs", "math"]
