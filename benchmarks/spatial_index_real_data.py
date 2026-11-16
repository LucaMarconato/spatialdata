"""Benchmark the spatial index of points (RFC 3) on real data, against reading without an index.

Run from the repository root:

    python benchmarks/spatial_index_real_data.py --original <store.zarr> --element transcripts \
        --indexed <indexed_store.zarr> [<indexed_store.zarr> ...] --out results.json

Each indexed store holds the same points written with ``write_points_with_spatial_index`` (Parquet or Zarr storage,
dense or sparse feature index). Each query is run ``--repeats`` times with a fresh reader (the OS page cache is warm
after the first run; the first run is reported separately as ``first``).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd


def _queries(md: Any, categories: list[str], counts: np.ndarray) -> list[dict[str, Any]]:
    lv0 = md.levels[0]
    origin = np.asarray(lv0.origin)
    extent = np.asarray(lv0.tile_size) * np.asarray(lv0.grid_shape)
    center = origin + extent / 2
    order = np.argsort(counts)
    nonzero = order[counts[order] > 0]
    rare = categories[int(nonzero[len(nonzero) // 10])]  # 10th percentile of the expressed features
    common = [categories[int(i)] for i in order[-3:]]
    medium = categories[int(nonzero[len(nonzero) // 2])]

    def box(size: float) -> tuple[np.ndarray, np.ndarray]:
        return center - size / 2, center + size / 2

    return [
        {"name": "small box 200 um, all features, level 0", "box": box(200.0)},
        {"name": "medium box 1000 um, all features, level 0", "box": box(1000.0)},
        {"name": "medium box 1000 um, budget 100k points", "box": box(1000.0), "max_points": 100_000},
        {"name": "whole dataset, budget 100k points", "box": None, "max_points": 100_000},
        {"name": f"whole dataset, rare feature {rare}", "box": None, "features": [rare]},
        {"name": f"whole dataset, median feature {medium}", "box": None, "features": [medium]},
        {"name": "medium box 1000 um, 3 common features", "box": box(1000.0), "features": common},
        {"name": "small box 200 um, 3 common features", "box": box(200.0), "features": common},
    ]


def _baseline(original: Path, element: str, feature_key: str) -> Callable[[dict[str, Any]], int]:
    """Without an index: dask read_parquet of the element, with filters on the box and the features."""
    path = original / "points" / element / "points.parquet"

    def run(q: dict[str, Any]) -> int:
        filters = []
        if q.get("box") is not None:
            lo, hi = q["box"]
            filters += [("x", ">=", lo[0]), ("x", "<=", hi[0]), ("y", ">=", lo[1]), ("y", "<=", hi[1])]
        if q.get("features"):
            filters += [(feature_key, "in", list(q["features"]))]
        df = dd.read_parquet(path, columns=["x", "y", feature_key], filters=filters or None)
        n = len(df)
        if q.get("max_points") and n > q["max_points"]:
            # the client has to read everything and then subsample
            pass
        return n

    return run


def _indexed(path: Path, element: str, use_feature_index: bool) -> Callable[[dict[str, Any]], int]:
    from spatialdata._io.spatial_index import PointsSpatialIndex

    def run(q: dict[str, Any]) -> int:
        index = PointsSpatialIndex(path / "points" / element)
        md = index.metadata
        fk = md.feature_keys[0] if md.feature_keys else None
        lo, hi = q["box"] if q.get("box") is not None else (None, None)
        budget = None
        if q.get("max_points"):
            s, e = index.tile_ranges(lo, hi, 0)
            n_in_view = int(np.sum(e - s))
            budget = min(1.0, q["max_points"] / max(n_in_view, 1))
        if q.get("features") and use_feature_index and fk is None:
            raise ValueError("no feature index")
        df = index.query(
            lo,
            hi,
            budget=budget,
            features=q.get("features"),
            feature_key=fk,
            columns=["x", "y", fk] if fk else ["x", "y"],
            use_feature_index=bool(q.get("features")) and use_feature_index,
        )
        table = index.table(fk if (q.get("features") and use_feature_index) else None)
        run.last_bytes = table.bytes_read  # type: ignore[attr-defined]
        run.last_row_groups = table.row_groups_read  # type: ignore[attr-defined]
        return len(df)

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--element", default="transcripts")
    ap.add_argument("--indexed", type=Path, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("spatial_index_benchmark.json"))
    args = ap.parse_args()

    from spatialdata._io.spatial_index import PointsSpatialIndex

    ref = PointsSpatialIndex(args.indexed[0] / "points" / args.element)
    fk = ref.metadata.feature_keys[0]
    cats = ref.categories(fk)
    # number of points per feature, from the whole-dataset ranges of the feature index
    s, e = ref.feature_ranges(fk, list(range(len(cats))), None, None, 0)
    counts = np.zeros(len(cats), dtype=np.int64)
    # feature_ranges returns one range per (band, feature) in band order, features sorted
    n_b = ref.metadata.n_levels
    counts = (e - s).reshape(n_b, len(cats)).sum(axis=0)
    queries = _queries(ref.metadata, cats, counts)

    methods: dict[str, Callable[[dict[str, Any]], int]] = {"no index (dask + filters)": _baseline(args.original, args.element, fk)}
    for p in args.indexed:
        methods[f"{p.name}: tile-major"] = _indexed(p, args.element, use_feature_index=False)
        methods[f"{p.name}: feature-major"] = _indexed(p, args.element, use_feature_index=True)

    results = []
    for q in queries:
        for name, fn in methods.items():
            if name.endswith("feature-major") and not q.get("features"):
                continue
            times = []
            n = -1
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                n = fn(q)
                times.append(time.perf_counter() - t0)
            row = {
                "query": q["name"],
                "method": name,
                "n_points": n,
                "first_s": times[0],
                "median_s": float(np.median(times)),
                "bytes_read": getattr(fn, "last_bytes", None),
                "row_groups_read": getattr(fn, "last_row_groups", None),
            }
            print(json.dumps(row), flush=True)
            results.append(row)
    args.out.write_text(json.dumps(results, indent=1))
    print(pd.DataFrame(results).to_string())


if __name__ == "__main__":
    main()
