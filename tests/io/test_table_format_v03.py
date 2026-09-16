from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData

from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import TablesFormatV02, TablesFormatV03
from spatialdata.models import TableModel


def _table_v03() -> AnnData:
    obs = pd.DataFrame(
        {"cells": pd.array([1, 2, pd.NA], dtype="Int64"), "nuclei": pd.array([pd.NA, 5, 6], dtype="Int64")},
        index=["a", "b", "c"],
    )
    adata = AnnData(X=np.zeros((3, 2)), obs=obs)
    adata.uns["spatialdata_attrs"] = {"version": "0.3"}
    return TableModel.parse(adata)


def _table_v02() -> AnnData:
    obs = pd.DataFrame({"region": ["cells"] * 3, "instance_id": [1, 2, 3]}, index=["a", "b", "c"])
    adata = AnnData(X=np.zeros((3, 2)), obs=obs)
    return TableModel.parse(adata, region="cells", region_key="region", instance_key="instance_id")


class TestTableFormatV03IO:
    def test_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData(tables={"table": _table_v03()})
            sdata.write(f, sdata_formats=[TablesFormatV03()])

            group = zarr.open_group(f, mode="r")["tables/table"]
            assert group.attrs["version"] == "0.3"
            assert not {"region", "region_key", "instance_key"} & set(group.attrs.keys())

            table = read_zarr(f)["table"]
            assert table.uns["spatialdata_attrs"] == {"version": "0.3"}
            assert table.obs["cells"].dtype == "Int64"
            assert table.obs["cells"].isna().tolist() == [False, False, True]
            assert table.obs["nuclei"].tolist()[1:] == [5, 6]

    def test_v03_table_requires_v03_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sdata = SpatialData(tables={"table": _table_v03()})
            with pytest.raises(ValueError, match="can only be written with `TablesFormatV03\\(\\)`"):
                sdata.write(Path(tmpdir) / "data.zarr", sdata_formats=[TablesFormatV02()])

    def test_legacy_table_cannot_be_written_as_v03(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sdata = SpatialData(tables={"table": _table_v02()})
            with pytest.raises(ValueError, match="Convert the table with"):
                sdata.write(Path(tmpdir) / "data.zarr", sdata_formats=[TablesFormatV03()])

    def test_default_format_still_writes_legacy_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData(tables={"table": _table_v02()})
            sdata.write(f)
            group = zarr.open_group(f, mode="r")["tables/table"]
            assert group.attrs["version"] == "0.2"
            assert group.attrs["region"] == "cells"
            assert read_zarr(f)["table"].uns["spatialdata_attrs"]["region_key"] == "region"
