"""Build the spatial index of RFC 3 (tiled and multiscale points) out of core.

The points are read partition by partition (a dask dataframe), so the input never has to fit in memory:

1. pass A: bounding box and a fine histogram of the coordinates, to choose the grid (origin, tile size, levels);
2. pass B: the priority p = k / 2**p_bits of each point, its band and its tile; the number of points per (band,
   tile), which gives the tile index, and per (feature, band), which plans the feature-major order;
3. pass C: each partition is scattered to buckets, contiguous ranges of rows of each order, in temporary Parquet files;
4. pass D: each bucket is sorted in memory and written to the final table (Parquet or Zarr), with its offsets.

Layout (the same in both formats)::

    <element>/spatial/points.parquet/ | spatial/points/      tile-major table
    <element>/spatial/tile_offsets/<l>                       tile index
    <element>/feature/<key>/points.parquet/ | points/        feature-major table
    <element>/feature/<key>/tile_offsets/<l>/{feature_ptr, tile_ids, row_offsets}   feature index (sparse, CSR)
    <element>/spatial_index.parquet.json                     Parquet only: part files, first rows, categories
"""

from __future__ import annotations

import logging
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
    DEFAULT_P_BITS,
    P_COLUMN,
    LevelGrid,
    SpatialIndexMetadata,
    Storage,
    merge_rule,
    min_p_bits,
    p_dtype,
)
from spatialdata._io.spatial_index._zarr_df import ZarrDataFrameWriter

logger = logging.getLogger(__name__)

_G = "__sd_g__"  # temporary column: global (band, tile) id
_K = "__sd_k__"  # temporary column: k of p = k / 2**p_bits (sorts like p)

SPATIAL = "spatial"


@dataclass
class _Plan:
    metadata: SpatialIndexMetadata
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

    @property
    def offset_dtype(self) -> Any:
        # the smallest integer type that holds the row numbers (RFC 3, "Tile index")
        return np.int32 if self.n_rows < 2**31 else np.int64


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


def _k(points_part: pd.DataFrame, seed: int, part_index: int, p_bits: int, source_p_bits: int | None) -> np.ndarray:
    """k of p = k / 2**p_bits for each point (uint32 in memory); an existing p is kept (RFC 3, "Priority").

    An existing column of floats p is converted with k = floor(p * 2**p_bits); an existing column of integer codes
    with ``source_p_bits`` bits is shifted to ``p_bits`` bits. Both keep the order of p, and the bands of the default
    rates (powers of 2) as long as p_bits is large enough for them.
    """
    if P_COLUMN in points_part.columns:
        col = points_part[P_COLUMN].to_numpy()
        if np.issubdtype(col.dtype, np.floating):
            p = col.astype(np.float64)
            if np.any(p < 0) or np.any(p >= 1):
                raise ValueError(f"Column {P_COLUMN} must be in [0, 1).")
            return np.floor(p * 2.0**p_bits).astype(np.uint32)
        src = source_p_bits or (16 if col.dtype == np.uint16 else 24)
        k = col.astype(np.uint64)
        k = k >> np.uint64(src - p_bits) if src >= p_bits else k << np.uint64(p_bits - src)
        return k.astype(np.uint32)
    rng = np.random.default_rng([seed, part_index])
    return rng.integers(0, 2**p_bits, size=len(points_part), dtype=np.uint64).astype(np.uint32)


def _bands_and_tiles(
    md: SpatialIndexMetadata, coords: np.ndarray, k: np.ndarray, band_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rank (L - band, coarsest first) and global (band, tile) id of each point."""
    band = md.band_of_k(k)
    g = np.empty(len(k), dtype=np.int64)
    for level in range(md.n_levels):
        mask = band == level
        if mask.any():
            rank = md.L - level
            g[mask] = band_base[rank] + md.levels[level].tile_ids(coords[mask])
    return md.L - band, g


def _choose_level0(
    points: dd.DataFrame,
    axes: list[str],
    tiled_axes: list[str],
    tile_size: float | Sequence[float] | None,
    origin: Sequence[float] | None,
    target_points_per_tile: int,
    n_workers: int,
) -> tuple[LevelGrid, int, dict[str, float]]:
    """Pass A: bounding box, then (if no tile size is given) a fine histogram to pick the tile size.

    All the ``axes`` are in the grid; the axes not in ``tiled_axes`` get a single tile. The tile size along the tiled
    axes is isotropic and chosen so that the median number of points of the non-empty tiles of level 0 is about
    ``target_points_per_tile`` (RFC 3, "Choosing the tile size and the rates").
    """
    nparts = points.npartitions
    d = len(axes)

    def bbox(i: int) -> tuple[np.ndarray, np.ndarray, int]:
        c = _partition(points, i, axes).to_numpy(dtype=np.float64)
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
    ti = [axes.index(a) for a in tiled_axes]
    dt = len(ti)

    s = np.empty(d, dtype=np.float64)
    # untiled axes: a single tile covering the extent
    for i in range(d):
        if i not in ti:
            s[i] = float(np.floor(extent[i]) + 1.0)
    if tile_size is not None:
        given = np.broadcast_to(np.asarray(tile_size, dtype=np.float64), (dt,))
        s[ti] = given
    else:
        ext_t = extent[ti]
        s_uniform = (float(np.prod(ext_t)) * target_points_per_tile / max(n, 1)) ** (1.0 / dt)
        h = s_uniform / 16.0
        bins = np.maximum(np.ceil(ext_t / h).astype(np.int64), 1)
        while np.prod(bins) > 2e8:  # keep the histogram small
            h *= 2
            bins = np.maximum(np.ceil(ext_t / h).astype(np.int64), 1)
        cols = [axes[i] for i in ti]

        def hist(i: int) -> np.ndarray:
            c = _partition(points, i, cols).to_numpy(dtype=np.float64)
            t = np.clip(np.floor((c - o[ti]) / h).astype(np.int64), 0, bins - 1)
            return np.bincount(np.ravel_multi_index(tuple(t.T), tuple(bins)), minlength=int(np.prod(bins)))

        H = np.zeros(int(np.prod(bins)), dtype=np.int64)
        with ThreadPoolExecutor(n_workers) as ex:
            for hh in ex.map(hist, range(nparts)):
                H += hh
        H = H.reshape(tuple(bins))

        def blocks(k: int) -> np.ndarray:
            Hp = np.pad(H, [(0, (-b) % k) for b in bins])
            shape: list[int] = []
            for b in Hp.shape:
                shape += [b // k, k]
            return Hp.reshape(shape).sum(axis=tuple(range(1, 2 * dt, 2)))

        best_k = 1
        for k in range(1, 16 * 16 + 1):
            B = blocks(k)
            nonempty = B[B > 0]
            best_k = k
            if len(nonempty) and np.median(nonempty) >= target_points_per_tile:
                break
        B = blocks(best_k)
        nonempty = B[B > 0]
        s[ti] = best_k * h
        info.update(
            {
                "median_points_per_tile_estimate": float(np.median(nonempty)),
                "p90_points_per_tile_estimate": float(np.percentile(nonempty, 90)),
                "max_points_per_tile_estimate": float(nonempty.max()),
            }
        )
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
    axes: list[str],
    tiled_axes: list[str],
    feature_keys: list[str],
    tile_size: float | Sequence[float] | None,
    origin: Sequence[float] | None,
    target_points_per_tile: int,
    n_levels: int | None,
    P: Sequence[float] | None,
    p_bits: int | None,
    source_p_bits: int | None,
    seed: int,
    n_workers: int,
) -> tuple[dd.DataFrame, _Plan]:
    t0 = time.perf_counter()
    points, categories = _global_categories(points, feature_keys)
    t1 = time.perf_counter()
    level0, n, info = _choose_level0(points, axes, tiled_axes, tile_size, origin, target_points_per_tile, n_workers)
    levels = merge_rule(level0, n_levels)
    if P is not None:
        if len(P) != len(levels):
            raise ValueError(f"P must have one entry per level ({len(levels)}), got {len(P)}.")
        levels = [LevelGrid(lv.origin, lv.tile_size, lv.grid_shape, float(p)) for lv, p in zip(levels, P, strict=True)]
    if p_bits is None:
        if n_levels is None and P is None:
            # default: keep p in 16 bits and stop the merge rule at the coarsest level that 16 bits can represent
            # (P(L) >= 2**(4 - 16)); the remaining levels hold a few tiles each, enough for an overview
            levels = [lv for lv in levels if lv.P >= 2.0 ** (4 - DEFAULT_P_BITS)]
            p_bits = DEFAULT_P_BITS
        else:
            p_bits = max(DEFAULT_P_BITS, min_p_bits(levels[-1].P))
    md = SpatialIndexMetadata(
        axes=tuple(axes),
        levels=tuple(levels),
        feature_keys=tuple(feature_keys),
        p_bits=p_bits,
        extra={"seed": seed},
    )
    md.validate()
    t2 = time.perf_counter()

    L = md.L
    # rank r = L - band; global id of (band, tile) = band_base[r] + tile id, so that ids sort coarsest band first
    n_tiles_by_rank = [md.levels[L - r].n_tiles for r in range(L + 1)]
    band_base = np.concatenate([[0], np.cumsum(n_tiles_by_rank)]).astype(np.int64)
    total = int(band_base[-1])
    columns = list(dict.fromkeys(list(axes) + feature_keys + ([P_COLUMN] if P_COLUMN in points.columns else [])))

    def count(i: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        df = _partition(points, i, columns)
        coords = df[list(axes)].to_numpy(dtype=np.float64)
        k = _k(df, seed, i, p_bits, source_p_bits)
        rank, g = _bands_and_tiles(md, coords, k, band_base)
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

    Returns the bucket of each unit (non-decreasing). A bucket becomes one part file, so files start at units.
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


def _offset_compressors() -> Any:
    from zarr.codecs import BloscCodec

    return [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")]


def _offset_filters(dtype: Any) -> Any:
    from zarr.codecs.numcodecs import Delta

    return [Delta(dtype=np.dtype(dtype).str)]


def _write_offsets(group: zarr.Group, name: str, data: np.ndarray, dtype: Any) -> None:
    data = np.asarray(data).astype(dtype)
    group.create_array(
        name,
        data=data,
        chunks=(max(1, min(len(data), 65536)),),
        compressors=_offset_compressors(),
        filters=_offset_filters(dtype),
        overwrite=True,
    )


class _OrderWriter:
    """Writes one order (the tile-major table or one feature-major table) bucket by bucket."""

    def __init__(
        self,
        root: zarr.Group,
        root_path: Path,
        storage: Storage,
        schema: pa.Schema,
        plan: _Plan,
        feature_key: str | None,
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
        self.row_group_min_rows = row_group_min_rows
        self.row_group_max_rows = row_group_max_rows
        self.parquet_compression = parquet_compression
        self.root = root
        self.table_dir = root_path / "points.parquet"
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
            self.table_dir.mkdir(parents=True, exist_ok=True)
        # the sparse feature index, accumulated per band
        self._sparse: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {lv: [] for lv in range(md.n_levels)}

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
            _write_offsets(self.offsets_group, str(level), offs, self.plan.offset_dtype)
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
            path = self.table_dir / _part_name(start)
            with pq.ParquetWriter(path, self.schema, compression=self.parquet_compression) as w:
                for a, b in groups:
                    w.write_table(_compact_dictionaries(table.slice(a, b - a)), row_group_size=b - a)
        else:
            assert self.df_writer is not None
            # buckets may share a (sharded) chunk: a partial write is a read-modify-write, so serialize the writes
            with self._lock:
                self.df_writer.write(start, table)
        if codes is not None:
            self._feature_index(start, g, codes)

    def _feature_index(self, start: int, g: np.ndarray, codes: np.ndarray) -> None:
        L = self.plan.L
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
        feature_all, r_all = np.divmod(ent_fr, L + 1)
        level_all = L - r_all
        with self._lock:
            for level in np.unique(level_all):
                sel = level_all == level
                self._sparse[int(level)].append((feature_all[sel], ent_tile[sel], ent_start[sel]))
                # the end of each (feature, band) range, marked with tile -1
                end = sel & last_of_group
                self._sparse[int(level)].append(
                    (feature_all[end], np.full(int(end.sum()), -1, dtype=np.int64), ent_end[end])
                )

    def finalize_feature_index(self) -> None:
        """Per band, CSR over the features of their non-empty tiles (RFC 3, "Feature index")."""
        md = self.plan.metadata
        L = md.L
        fk = self.feature_key
        assert fk is not None
        n_features = len(self.plan.categories[fk])
        # start row of every (feature, rank) range, also the empty ones
        cfr = self.plan.counts_fr[fk]
        fr_start = (np.cumsum(cfr.ravel()) - cfr.ravel()).reshape(cfr.shape)
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
            # features without points in this band: a single entry, the row where their (empty) range would start
            pos_end = feature_ptr[1:] + np.arange(n_features)
            row_offsets[pos_end] = fr_start[:, r]
            row_offsets[pos_end[feats[is_end]]] = rows[is_end]
            mask = np.ones(len(row_offsets), dtype=bool)
            mask[pos_end] = False
            row_offsets[mask] = rows[~is_end]
            g = self.offsets_group.require_group(str(level))
            dtype = self.plan.offset_dtype
            tile_dtype = np.int32 if md.levels[level].n_tiles < 2**31 else np.int64
            _write_offsets(g, "feature_ptr", feature_ptr, dtype)
            _write_offsets(g, "tile_ids", tiles[~is_end], tile_dtype)
            _write_offsets(g, "row_offsets", row_offsets, dtype)


def _compact_dictionaries(table: pa.Table) -> pa.Table:
    """Keep in the dictionary of each categorical column only the values used in this row group.

    Parquet stores the dictionary of a column once per row group: with tens of thousands of features, a full dictionary
    per row group would dominate the size of the file. The global categories (whose codes the feature index uses) are
    stored once, in the Parquet sidecar.
    """
    cols = []
    for i, f in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(f.type) and table.num_rows:
            arr = col.combine_chunks()
            idx = arr.indices.to_numpy(zero_copy_only=False)
            used, remap = np.unique(idx, return_inverse=True)
            arr = pa.DictionaryArray.from_arrays(
                pa.array(remap.astype(idx.dtype), type=f.type.index_type), arr.dictionary.take(pa.array(used))
            )
            col = pa.chunked_array([arr], type=f.type)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=table.schema)


def _output_schema(points: dd.DataFrame, plan: _Plan, drop: Sequence[str]) -> pa.Schema:
    meta = points._meta.reset_index(drop=True)
    meta = meta[[c for c in meta.columns if c not in drop and c != P_COLUMN]]
    for c, cats in plan.categories.items():
        if c in meta.columns:
            meta[c] = meta[c].astype(pd.CategoricalDtype(cats))
    meta[P_COLUMN] = pd.Series([], dtype=p_dtype(plan.metadata.p_bits))
    # the commonly projected columns first (RFC 3, "Order of the columns")
    first = [*plan.metadata.axes]
    first += [c for c in ("x", "y", "z") if c in meta.columns and c not in first]
    first += [*plan.feature_keys, P_COLUMN]
    order = list(dict.fromkeys(first + [n for n in meta.columns if n not in first]))
    # keep the pandas metadata, so that readers such as dask restore the categories
    return pa.Schema.from_pandas(meta[order], preserve_index=False)


def _to_table(df: pd.DataFrame, schema: pa.Schema, plan: _Plan, k: np.ndarray) -> pa.Table:
    df = df.copy()
    for c, cats in plan.categories.items():
        if c in df.columns:
            if not isinstance(df[c].dtype, pd.CategoricalDtype) or list(df[c].cat.categories) != cats:
                df[c] = df[c].astype(str).astype(pd.CategoricalDtype(cats))
    df[P_COLUMN] = k.astype(p_dtype(plan.metadata.p_bits))
    t = pa.Table.from_pandas(df[schema.names], preserve_index=False)
    return t.cast(schema)


def write_points_spatial_index(
    points: dd.DataFrame | pd.DataFrame,
    group: zarr.Group,
    *,
    storage: Storage = "parquet",
    axes: Sequence[str] | None = None,
    tiled_axes: Sequence[str] | None = None,
    feature_keys: Sequence[str] | None = None,
    tile_size: float | Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    target_points_per_tile: int = 10_000,
    n_levels: int | None = None,
    P: Sequence[float] | None = None,
    p_bits: int | None = None,
    source_p_bits: int | None = None,
    seed: int = 0,
    rows_per_bucket: int = 20_000_000,
    row_group_min_rows: int = 250_000,
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

    Only the data of the index is written (``spatial/``, ``feature/``); the caller writes the element metadata
    (transformations, ``spatialdata_attrs``), the ``spatial_index`` attributes and, for Parquet, the sidecar, from the
    returned metadata and statistics (``stats["parquet_sidecar"]``).

    Parameters
    ----------
    axes
        All the spatial axes of the element (default: ``x``, ``y`` and ``z`` if present).
    tiled_axes
        The axes with more than one tile (default: ``x`` and ``y``); the others get a single tile.
    target_points_per_tile
        Target for the median number of points of the non-empty tiles of level 0.
    p_bits
        Precision of p (16, 24 or 32); default: 16, or more if the coarsest level needs it.
    source_p_bits
        Precision of an existing integer ``__spatialdata_p__`` column (default: 16 for uint16, 24 for uint32).
    row_group_min_rows, row_group_max_rows
        Parquet row groups hold several tiles (at least ``row_group_min_rows`` rows) and split larger tiles.

    Returns
    -------
    The metadata of the index and a dictionary of statistics (timings, sizes, and the Parquet sidecar).
    """
    log = progress or (lambda msg: logger.info(msg))
    points = _ensure_dask(points)
    n_workers = n_workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    if axes is None:
        axes = [a for a in ("x", "y", "z") if a in points.columns]
    axes = list(axes)
    if tiled_axes is None:
        tiled_axes = [a for a in ("x", "y") if a in axes]
    tiled_axes = list(tiled_axes)
    if feature_keys is None:
        fk = points.attrs.get("spatialdata_attrs", {}).get("feature_key")
        feature_keys = [fk] if fk is not None else []
    feature_keys = list(feature_keys)
    if zarr_chunk_length is None:
        zarr_chunk_length = target_points_per_tile

    t_start = time.perf_counter()
    log("planning the grid and counting the points per tile")
    points, plan = _plan(
        points,
        axes,
        tiled_axes,
        feature_keys,
        tile_size,
        origin,
        target_points_per_tile,
        n_levels,
        P,
        p_bits,
        source_p_bits,
        seed,
        n_workers,
    )
    md = plan.metadata
    stats: dict[str, Any] = {"timings": dict(plan.timings), "n_rows": plan.n_rows, "n_levels": md.n_levels}
    log(
        f"{plan.n_rows:,} points, {md.n_levels} levels, level 0 grid {md.levels[0].grid_shape} of tile size "
        f"{md.levels[0].tile_size}, p_bits {md.p_bits}"
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

    root_path = _local_path(group)
    tmp_root = Path(tempfile.mkdtemp(prefix="spatial_index_", dir=tmp_dir))
    stats["tmp_dir"] = str(tmp_root)
    orders = [SPATIAL] + [f"feature/{fk}" for fk in feature_keys]
    try:
        # pass C: scatter
        t0 = time.perf_counter()
        log(f"scattering {points.npartitions} partitions to buckets in {tmp_root}")

        def scatter(i: int) -> int:
            df = _partition(points, i, in_columns)
            k = _k(df, seed, i, md.p_bits, source_p_bits)
            coords = df[axes].to_numpy(dtype=np.float64)
            rank, g = _bands_and_tiles(md, coords, k, plan.band_base)
            table = _to_table(df.drop(columns=[P_COLUMN], errors="ignore"), schema, plan, k)
            table = table.append_column(_G, pa.array(g)).append_column(_K, pa.array(k))
            for order in orders:
                if order == SPATIAL:
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
        writers: dict[str, _OrderWriter] = {}
        for order in orders:
            fk_or_none = None if order == SPATIAL else order.split("/", 1)[1]
            og = group.require_group(order)
            writers[order] = _OrderWriter(
                og,
                root_path / order,
                storage,
                schema,
                plan,
                fk_or_none,
                row_group_min_rows,
                row_group_max_rows,
                zarr_chunk_length,
                zarr_shard_length,
                parquet_compression,
            )
        writers[SPATIAL].write_tile_index()
        sidecar: dict[str, Any] = {"tables": {}}

        t0 = time.perf_counter()
        for order in orders:
            fk = None if order == SPATIAL else order.split("/", 1)[1]
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
                kk = t.column(_K).to_numpy()
                if fk is None:
                    idx = np.lexsort((kk, g))
                    codes = None
                else:
                    codes_all = t.column(fk).combine_chunks().indices.to_numpy().astype(np.int64)
                    idx = np.lexsort((kk, g, codes_all))
                    codes = codes_all[idx]
                t = t.take(pa.array(idx)).drop_columns([_G, _K]).cast(schema)
                assert t.num_rows == starts[b + 1] - starts[b]
                writers[order].write_bucket(int(starts[b]), t, g[idx], codes)
                shutil.rmtree(d)

            with ThreadPoolExecutor(max(1, n_workers // 2)) as ex:
                list(ex.map(sort_write, range(n_b)))
            if fk is not None:
                writers[order].finalize_feature_index()
            if storage == "parquet":
                # the part files in order and the first row of each, so that a reader neither lists the directory
                # nor opens the footers it does not need (the Parquet sidecar)
                nonempty = [int(starts[b]) for b in range(n_b) if starts[b + 1] > starts[b]]
                entry: dict[str, Any] = {
                    "files": [f"{order}/points.parquet/{_part_name(r)}" for r in nonempty],
                    "first_row": nonempty + [int(starts[-1])],
                }
                if fk is not None:
                    entry["categories"] = list(plan.categories[fk])
                sidecar["tables"][order] = entry
        stats["timings"]["sort_write"] = time.perf_counter() - t0
        if storage == "parquet":
            stats["parquet_sidecar"] = sidecar
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    stats["timings"]["total"] = time.perf_counter() - t_start
    return md, stats


def _part_name(first_row: int) -> str:
    """Zero-padded, so that tools that list and sort the directory (dask, pyarrow) read the files in order."""
    return f"part.{first_row:012d}.parquet"


def _local_path(group: zarr.Group) -> Path:
    from zarr.storage import LocalStore

    store = group.store_path.store
    if not isinstance(store, LocalStore):
        raise TypeError(f"Writing a spatial index requires a local zarr store, got {type(store).__name__}.")
    return Path(store.root) / group.path


__all__ = ["write_points_spatial_index"]
