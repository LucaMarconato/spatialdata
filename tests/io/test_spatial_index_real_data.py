"""Checks of the spatial index on real data, run only locally.

Set ``SPATIALDATA_SPATIAL_INDEX_REAL_DATA`` to a directory holding stores written with
``write_points_with_spatial_index`` (``<store>.zarr/points/<name>``); each indexed points element found there is
checked against brute force on a few regions. Skipped in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = os.environ.get("SPATIALDATA_SPATIAL_INDEX_REAL_DATA")
pytestmark = pytest.mark.skipif(ROOT is None, reason="set SPATIALDATA_SPATIAL_INDEX_REAL_DATA to run")


def _elements() -> list[Path]:
    if ROOT is None:
        return []
    return sorted(p.parent for p in Path(ROOT).glob("*.zarr/points/*/zarr.json") if "spatial_index" in p.read_text())


@pytest.mark.parametrize("element", _elements(), ids=lambda p: f"{p.parents[1].name}/{p.name}")
def test_real_data_queries(element):
    from spatialdata._io.spatial_index import P_COLUMN, PointsSpatialIndex

    index = PointsSpatialIndex(element)
    md = index.metadata
    lv0 = md.levels[0]
    rng = np.random.default_rng(0)
    extent = np.asarray(lv0.tile_size) * np.asarray(lv0.grid_shape)
    fk = md.feature_keys[0] if md.feature_keys else None
    for _ in range(3):
        lo = np.asarray(lv0.origin) + rng.uniform(0.2, 0.7, 2) * extent
        hi = lo + 3 * np.asarray(lv0.tile_size)
        # superset read: the tiles in view of level 0 and all coarser bands, no exact filter
        allpts = index.query(lo - lv0.tile_size, hi + lv0.tile_size, columns=["x", "y", P_COLUMN] + ([fk] if fk else []))
        inside = allpts[(allpts.x >= lo[0]) & (allpts.x <= hi[0]) & (allpts.y >= lo[1]) & (allpts.y <= hi[1])]
        for level in range(md.n_levels):
            q = index.query(lo, hi, level=level, columns=["x", "y"])
            assert len(q) == int((inside[P_COLUMN].astype(float) < md.levels[level].P * 2**md.p_bits).sum())
        if fk:
            gene = str(inside[fk].value_counts().index[0])
            a = index.query(lo, hi, features=[gene], use_feature_index=True, columns=["x", "y"])
            b = index.query(lo, hi, features=[gene], use_feature_index=False, columns=["x", "y"])
            assert len(a) == len(b) == int((inside[fk] == gene).sum())
