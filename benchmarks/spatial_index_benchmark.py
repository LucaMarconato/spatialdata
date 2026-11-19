"""asv benchmarks of the spatial index of points (RFC 3) on synthetic data.

For real data (10x Atera) see ``spatial_index_real_data.py``.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import spatialdata as sd
from spatialdata._io.spatial_index import PointsSpatialIndex, write_points_with_spatial_index
from spatialdata.models import PointsModel


def _points(n: int, n_genes: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.normal([3000, 2000], 600, size=(n // 2, 2)), rng.uniform(0, 10_000, size=(n - n // 2, 2))])
    w = 1.0 / np.arange(1, n_genes + 1) ** 1.2
    genes = rng.choice(n_genes, size=n, p=w / w.sum())
    return pd.DataFrame(
        {
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
            "gene": pd.Categorical.from_codes(genes, [f"g{i}" for i in range(n_genes)]),
        }
    )


class TimeSpatialIndexQueries:
    params = (["none", "parquet", "zarr"], ["small box", "large box budget", "rare gene", "common gene in box"])
    param_names = ["storage", "query"]
    timeout = 600

    def setup_cache(self) -> str:
        tmp = Path(tempfile.mkdtemp())
        df = _points(5_000_000)
        pts = PointsModel.parse(df, feature_key="gene", npartitions=8)
        sd.SpatialData(points={"plain": pts}).write(tmp / "data.zarr")
        for storage in ("parquet", "zarr"):
            write_points_with_spatial_index(pts, tmp / "data.zarr" / "points" / storage, storage=storage)
        return str(tmp / "data.zarr")

    def setup(self, path: str, storage: str, query: str) -> None:
        self.path = Path(path)
        self.storage = storage
        self.index = None if storage == "none" else PointsSpatialIndex(self.path / "points" / storage)

    def time_query(self, path: str, storage: str, query: str) -> None:
        box = {"small box": (2900, 3100), "large box budget": (0, 6000), "common gene in box": (2500, 3500)}
        features = {"rare gene": ["g400"], "common gene in box": ["g0"]}.get(query)
        lo_hi = box.get(query)
        if self.index is None:
            import dask.dataframe as dd

            filters = []
            if lo_hi is not None:
                lo, hi = lo_hi
                filters += [("x", ">=", lo), ("x", "<=", hi), ("y", ">=", lo), ("y", "<=", hi)]
            if features:
                filters += [("gene", "in", features)]
            len(dd.read_parquet(self.path / "points" / "plain" / "points.parquet", filters=filters or None))
            return
        lo, hi = (None, None) if lo_hi is None else ([lo_hi[0]] * 2, [lo_hi[1]] * 2)
        budget = 0.02 if query == "large box budget" else None
        self.index.query(lo, hi, budget=budget, features=features, columns=["x", "y", "gene"])
