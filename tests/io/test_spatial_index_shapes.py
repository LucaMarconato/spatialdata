from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import zarr
from anndata import AnnData
from scipy import sparse

from spatialdata import SpatialData, build_points_spatial_index, build_shapes_spatial_index, read_zarr
from spatialdata._io.spatial_index import PointsSpatialIndex
from spatialdata.models import PointsModel, ShapesModel, TableModel


@pytest.fixture
def sdata_cells(tmp_path):
    rng = np.random.default_rng(0)
    n = 2_000
    c = rng.uniform(0, 1000, size=(n, 2))
    polys = [shapely.Point(x, y).buffer(rng.uniform(2, 6), quad_segs=2) for x, y in c]
    polys[5] = shapely.MultiPolygon([polys[5], shapely.Point(c[5] + 10).buffer(1)])
    ids = rng.permutation(np.arange(10_000, 10_000 + n))
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=polys, index=pd.Index(ids, name="cell_id")))
    X = sparse.random(n, 50, density=0.1, format="csr", random_state=0, dtype=np.float32)
    obs = pd.DataFrame({"cell_id": rng.permutation(ids), "region": pd.Categorical(["cells"] * n)})
    table = TableModel.parse(
        AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(50)])),
        region="cells",
        region_key="region",
        instance_key="cell_id",
    )
    df = pd.DataFrame(
        {"x": rng.uniform(0, 1000, 20_000), "y": rng.uniform(0, 1000, 20_000), "gene": pd.Categorical(["a"] * 20_000)}
    )
    points = PointsModel.parse(df, feature_key="gene")
    sdata = SpatialData(shapes={"cells": shapes}, tables={"table": table}, points={"transcripts": points})
    path = tmp_path / "data.zarr"
    sdata.write(path)
    return path


def test_build_shapes_index(sdata_cells):
    path = sdata_cells
    sdata = read_zarr(path)
    original = sdata["cells"].copy()
    original_table = sdata["table"].copy()
    build_points_spatial_index(sdata, "transcripts", target_points_per_tile=500)
    grid_pts = PointsSpatialIndex(path / "points" / "transcripts").metadata.levels[0]
    index = build_shapes_spatial_index(sdata, "cells", table_name="table", nest_with_points="transcripts")
    md = index.metadata
    # a grid nested with the points: a power of 2 times their tile size, origin aligned to their tiles
    ratio = md.levels[0].tile_size[0] / grid_pts.tile_size[0]
    assert ratio >= 1 and float(np.log2(ratio)).is_integer()
    shift = (np.asarray(grid_pts.origin[:2]) - np.asarray(md.levels[0].origin)) / grid_pts.tile_size[0]
    assert np.allclose(shift, np.round(shift))
    assert md.extra["fraction_within_tile"] >= 0.5
    assert md.margin is not None and md.table == "table"
    assert zarr.open_group(path / "shapes" / "cells").attrs["spatialdata_attrs"]["version"] == "0.4"

    sdata2 = read_zarr(path)
    cells = sdata2["cells"]
    assert len(cells) == len(original)
    assert set(cells.index) == set(original.index)
    assert (cells.geom_type == "MultiPolygon").all()  # geoarrow: mixed column written as MultiPolygon
    # rows sorted by tile of the centroid
    tiles = md.levels[0].tile_ids(shapely.get_coordinates(shapely.centroid(cells.geometry.to_numpy())))
    assert np.all(np.diff(tiles) >= 0)
    # table rows in the same order as the shapes, X CSR and X_csc CSC with the same values
    table = sdata2["table"]
    np.testing.assert_array_equal(table.obs["cell_id"].to_numpy(), cells.index.to_numpy())
    assert sparse.isspmatrix_csr(table.X) or isinstance(table.X, sparse.csr_array)
    assert (table.X != table.layers["X_csc"]).nnz == 0
    old = original_table[pd.Index(original_table.obs["cell_id"]).get_indexer(cells.index)]
    assert (table.X != old.X).nnz == 0

    # query: same as brute force (shapes intersecting the box)
    lo, hi = np.array([200.0, 300.0]), np.array([400.0, 450.0])
    q = index.query(lo, hi)
    expected = cells[cells.intersects(shapely.box(*lo, *hi))]
    assert set(q.index) == set(expected.index) if q.index.name == "cell_id" else len(q) == len(expected)
    assert len(q) == len(expected)
    # gene values from the CSC layer for the shapes in view
    v = index.gene_values("g3", q["__row__"].to_numpy())
    np.testing.assert_allclose(v, table.X[q["__row__"].to_numpy(), 3].toarray().ravel())
    m = index.table_rows(q["__row__"].to_numpy()[:10])
    assert (m != table.X[q["__row__"].to_numpy()[:10]]).nnz == 0

    # a table that does not annotate the shapes is refused
    sdata3 = read_zarr(path)
    sdata3["table"].uns["spatialdata_attrs"]["region"] = "other"
    with pytest.raises(ValueError, match="annotates"):
        build_shapes_spatial_index(sdata3, "cells", table_name="table")
