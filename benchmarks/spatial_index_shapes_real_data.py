"""Benchmark the spatial index of shapes and the CSC layer (RFC 4) on real data.

    python benchmarks/spatial_index_shapes_real_data.py --original <store.zarr> --indexed <store.zarr> \
        --shapes cell_boundaries --table table --gene C7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    import anndata
    import geopandas as gpd
    import shapely
    import zarr

    from spatialdata._io.spatial_index import ShapesSpatialIndex

    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--indexed", type=Path, required=True)
    ap.add_argument("--shapes", default="cell_boundaries")
    ap.add_argument("--table", default="table")
    ap.add_argument("--gene", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("spatial_index_shapes_benchmark.json"))
    args = ap.parse_args()

    index = ShapesSpatialIndex(args.indexed / "shapes" / args.shapes, args.indexed / "tables" / args.table)
    lv0 = index.metadata.levels[0]
    center = np.asarray(lv0.origin) + np.asarray(lv0.tile_size) * np.asarray(lv0.grid_shape) / 2
    original_shapes = args.original / "shapes" / args.shapes / "shapes.parquet"
    j = index.var_names.index(args.gene)

    def no_index_box(size: float) -> int:
        lo, hi = center - size / 2, center + size / 2
        # WKB without a bbox column: no row-group pruning is possible, read everything and filter
        gdf = gpd.read_parquet(original_shapes)
        return int(gdf.intersects(shapely.box(*lo, *hi)).sum())

    def index_box(size: float) -> int:
        lo, hi = center - size / 2, center + size / 2
        return len(ShapesSpatialIndex(index.path, index.table_path).query(lo, hi))

    def gene_csr_no_index() -> int:
        # without a CSC copy: one gene means reading the whole CSR matrix
        X = anndata.io.read_elem(zarr.open_group(args.original / "tables" / args.table, mode="r")["X"])
        return int((X[:, j] > 0).sum())

    def gene_csc_existing() -> int:
        g = zarr.open_group(args.original / "tables" / args.table, mode="r")
        if "layers" not in g or "csc_counts" not in g["layers"]:
            return -1
        m = g["layers"]["csc_counts"]
        a, b = (int(v) for v in m["indptr"][j : j + 2])
        return int((m["data"][a:b] > 0).sum())

    def gene_csc_index() -> int:
        return int((index.gene_values(args.gene) > 0).sum())

    def gene_in_view(size: float) -> int:
        lo, hi = center - size / 2, center + size / 2
        cells = index.query(lo, hi)
        return int((index.gene_values(args.gene, cells["__row__"].to_numpy()) > 0).sum())

    cases = {
        "box 100 um: no index (read all + filter)": lambda: no_index_box(100.0),
        "box 100 um: index": lambda: index_box(100.0),
        "box 1 mm: no index (read all + filter)": lambda: no_index_box(1000.0),
        "box 1 mm: index": lambda: index_box(1000.0),
        f"gene {args.gene}, all cells: CSR X (no CSC)": gene_csr_no_index,
        f"gene {args.gene}, all cells: spatialdata-db csc_counts": gene_csc_existing,
        f"gene {args.gene}, all cells: X_csc": gene_csc_index,
        f"gene {args.gene}, cells in 300 um box: index + X_csc": lambda: gene_in_view(300.0),
    }
    results = []
    for name, fn in cases.items():
        times, n = [], -1
        for _ in range(args.repeats if "CSR X" not in name else 1):
            t0 = time.perf_counter()
            n = fn()
            times.append(time.perf_counter() - t0)
        row = {"case": name, "n": n, "first_s": times[0], "median_s": float(np.median(times))}
        print(json.dumps(row), flush=True)
        results.append(row)
    args.out.write_text(json.dumps(results, indent=1))
    print(pd.DataFrame(results).to_string())


if __name__ == "__main__":
    main()
