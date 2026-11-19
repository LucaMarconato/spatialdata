from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

from spatialdata import SpatialData, build_points_spatial_index, read_points_spatial_index, read_zarr
from spatialdata._io.spatial_index import P_COLUMN, LevelGrid, SpatialIndexMetadata, merge_rule
from spatialdata.models import PointsModel


def _points(n: int = 20_000, n_genes: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # a dense blob and a uniform background
    blob = rng.normal(loc=[300, 200], scale=30, size=(n // 2, 2))
    bg = rng.uniform([0, 0], [1000, 600], size=(n - n // 2, 2))
    xy = np.vstack([blob, bg]).clip([0, 0], [999.9, 599.9])
    # genes with very different abundances
    w = 1.0 / np.arange(1, n_genes + 1) ** 1.5
    genes = rng.choice([f"g{i}" for i in range(n_genes)], size=n, p=w / w.sum())
    return pd.DataFrame(
        {
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
            "z": rng.uniform(0, 10, n).astype(np.float32),
            "gene": pd.Categorical(genes),
            "cell_id": rng.integers(0, 100, n).astype(str),
            "qv": rng.uniform(20, 40, n).astype(np.float32),
        }
    )


@pytest.fixture
def sdata_path(tmp_path):
    df = _points()
    points = PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "z"}, feature_key="gene", npartitions=4)
    sdata = SpatialData(points={"transcripts": points})
    path = tmp_path / "data.zarr"
    sdata.write(path)
    return path, df


def test_merge_rule_example():
    # the anisotropic example of RFC 3, axes (z, y, x)
    lv0 = LevelGrid((0.0, 0.0, 0.0), (1.0, 10.0, 100.0), (10, 10, 10), 1.0)
    levels = merge_rule(lv0)
    assert [lv.tile_size for lv in levels][-1] == (16.0, 160.0, 1600.0)
    assert [lv.grid_shape for lv in levels][4] == (1, 5, 10)
    assert len(levels) == 11
    SpatialIndexMetadata(axes=("z", "y", "x"), levels=tuple(levels)).validate()


@pytest.mark.parametrize("storage", ["parquet", "zarr"])
def test_build_and_query(sdata_path, storage):
    path, df = sdata_path
    sdata = read_zarr(path)
    index = build_points_spatial_index(
        sdata,
        "transcripts",
        storage=storage,
        target_points_per_tile=500,
        rows_per_bucket=3_000,
        row_group_min_rows=1_000,
        zarr_chunk_length=256,
    )
    md = index.metadata
    assert md.levels[0].P == 1.0
    assert md.n_levels > 1
    assert md.feature_keys == ("gene",)

    # the element reads as a normal points element, with every point once and the column p
    sdata2 = read_zarr(path)
    pts = sdata2["transcripts"].compute()
    assert len(pts) == len(df)
    assert P_COLUMN in pts.columns
    assert pts[P_COLUMN].dtype == np.uint16 and md.p_bits == 16
    assert md.axes == ("x", "y", "z") and md.levels[0].grid_shape[2] == 1  # all axes, a single tile along z
    attrs = zarr.open_group(path / "points" / "transcripts").attrs.asdict()
    assert attrs["spatialdata_attrs"]["version"] == {"parquet": "0.3", "zarr": "0.3-experimental"}[storage]
    assert "spatial_index" in attrs
    # the same layout in both formats; Parquet adds the sidecar
    element = path / "points" / "transcripts"
    assert (element / "spatial" / "tile_offsets").exists() and (element / "feature" / "gene" / "tile_offsets").exists()
    assert (element / "spatial_index.parquet.json").exists() == (storage == "parquet")
    assert ("parquet" in attrs["spatial_index"]) == (storage == "parquet")
    # offsets: the smallest integer type
    assert zarr.open_array(element / "spatial" / "tile_offsets" / "0").dtype == np.int32

    # tile-major order: sorted by band (coarsest first), tile, p
    band = md.band_of_k(pts[P_COLUMN].to_numpy())
    assert np.all(np.diff(-band) >= 0)

    # all points at level 0
    full = index.query()
    assert len(full) == len(df)

    lo, hi = np.array([250.0, 150.0]), np.array([420.0, 260.0])
    in_box = (df.x >= lo[0]) & (df.x <= hi[0]) & (df.y >= lo[1]) & (df.y <= hi[1])
    q = index.query(lo, hi)
    assert len(q) == int(in_box.sum())

    # levels are nested and follow P
    pts_box = pts[(pts.x >= lo[0]) & (pts.x <= hi[0]) & (pts.y >= lo[1]) & (pts.y <= hi[1])]
    for level in range(md.n_levels):
        ql = index.query(lo, hi, level=level)
        expected = int((pts_box[P_COLUMN] < md.levels[level].P * 2**md.p_bits).sum())
        assert len(ql) == expected

    # budget
    qb = index.query(lo, hi, budget=0.1)
    assert len(qb) == int((pts_box[P_COLUMN] < 0.1 * 2**md.p_bits).sum())

    # feature-major and tile-major give the same points
    genes = ["g0", "g7", "g29"]
    for level in (0, md.L):
        a = index.query(lo, hi, level=level, features=genes, use_feature_index=True)
        b = index.query(lo, hi, level=level, features=genes, use_feature_index=False)
        key = ["x", "y", "gene", P_COLUMN]
        a = a[key].sort_values(key).reset_index(drop=True)
        b = b[key].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_categorical=False)
        expected = int(((pts_box[P_COLUMN] < md.levels[level].P * 2**md.p_bits) & pts_box.gene.isin(genes)).sum())
        assert len(a) == expected
    # whole-dataset feature query (no box)
    a = index.query(features=["g3"])
    assert len(a) == int((df.gene == "g3").sum())


def test_rebuild_keeps_p(sdata_path):
    path, _ = sdata_path
    sdata = read_zarr(path)
    build_points_spatial_index(sdata, "transcripts", target_points_per_tile=500, feature_keys=[])
    p1 = np.sort(read_zarr(path)["transcripts"][P_COLUMN].compute().to_numpy())
    sdata = read_zarr(path)
    build_points_spatial_index(sdata, "transcripts", tile_size=37.0, storage="zarr", feature_keys=[])
    p2 = np.sort(read_zarr(path)["transcripts"][P_COLUMN].compute().to_numpy())
    np.testing.assert_array_equal(p1, p2)
    index = read_points_spatial_index(path / "points" / "transcripts")
    assert index.metadata.levels[0].tile_size[:2] == (37.0, 37.0)
