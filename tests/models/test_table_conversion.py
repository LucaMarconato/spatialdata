from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from spatialdata._core.links import Edge
from spatialdata.models import TableModel, convert_table_v02_to_v03, convert_table_v03_to_v02, get_table_keys


def _legacy_table(regions: list[str], ids: list[int] | list[str]) -> AnnData:
    obs = pd.DataFrame({"region": regions, "instance_id": ids, "kind": ["a", "b", "c", "d"]})
    obs.index = [f"row{i}" for i in range(len(obs))]
    adata = AnnData(X=np.zeros((len(obs), 2)), obs=obs)
    unique_regions = sorted(set(regions))
    region: str | list[str] = unique_regions[0] if len(unique_regions) == 1 else unique_regions
    return TableModel.parse(adata, region=region, region_key="region", instance_key="instance_id")


class TestConvertV02ToV03:
    def test_single_region(self) -> None:
        table = _legacy_table(["cells"] * 4, [1, 2, 3, 4])
        converted, edges = convert_table_v02_to_v03(table, "table")

        assert list(converted.obs.columns) == ["kind", "cells"]
        assert converted.obs["cells"].dtype == "Int64"
        assert converted.obs["cells"].tolist() == [1, 2, 3, 4]
        assert converted.uns["spatialdata_attrs"] == {"version": "0.3"}
        assert edges == [
            Edge(from_="table", to="cells", method="join", on={"left": "cells", "right": "index"}, cardinality="1:1")
        ]
        # the input is left untouched
        assert get_table_keys(table) == ("cells", "region", "instance_id")

    def test_multiple_regions_and_labels(self) -> None:
        table = _legacy_table(["cells", "cells", "nuclei", "nuclei"], [1, 2, 1, 5])
        converted, edges = convert_table_v02_to_v03(table, "table", labels=["nuclei"])

        assert converted.obs["cells"].tolist()[:2] == [1, 2]
        assert converted.obs["cells"].isna().tolist() == [False, False, True, True]
        assert converted.obs["nuclei"].isna().tolist() == [True, True, False, False]
        assert converted.obs["nuclei"].tolist()[2:] == [1, 5]
        assert [e.to for e in edges] == ["cells", "nuclei"]
        assert edges[0].on == {"left": "cells", "right": "index"}
        assert edges[1].on == {"left": "nuclei", "right": "value"}

    def test_string_ids(self) -> None:
        table = _legacy_table(["cells", "cells", "nuclei", "nuclei"], ["a", "b", "c", "d"])
        converted, _ = convert_table_v02_to_v03(table, "table")
        assert converted.obs["cells"].dtype == "string"
        assert converted.obs["cells"].tolist()[:2] == ["a", "b"]
        assert converted.obs["cells"].isna().tolist()[2:] == [True, True]

    def test_copy_false_converts_in_place(self) -> None:
        table = _legacy_table(["cells"] * 4, [1, 2, 3, 4])
        converted, _ = convert_table_v02_to_v03(table, "table", copy=False)
        assert converted is table
        assert "cells" in table.obs

    def test_column_clash(self) -> None:
        table = _legacy_table(["cells"] * 4, [1, 2, 3, 4])
        table.obs["cells"] = 0
        with pytest.raises(ValueError, match=r"link column\(s\) \['cells'\]"):
            convert_table_v02_to_v03(table, "table")

    def test_requires_region_annotation(self) -> None:
        table = TableModel.parse(AnnData(X=np.zeros((2, 1))))
        with pytest.raises(ValueError, match="no `region`/`region_key`/`instance_key` metadata"):
            convert_table_v02_to_v03(table, "table")


class TestConvertV03ToV02:
    def test_roundtrip(self) -> None:
        table = _legacy_table(["cells", "cells", "nuclei", "nuclei"], [1, 2, 1, 5])
        converted, edges = convert_table_v02_to_v03(table, "table")
        back = convert_table_v03_to_v02(converted, edges, "table")

        pd.testing.assert_frame_equal(back.obs, table.obs, check_like=True, check_categorical=False)
        assert get_table_keys(back) == (["cells", "nuclei"], "region", "instance_id")
        assert TableModel.validate(back) is back

    def test_accepts_edge_dicts_and_single_region(self) -> None:
        table = _legacy_table(["cells"] * 4, [1, 2, 3, 4])
        converted, edges = convert_table_v02_to_v03(table, "table")
        back = convert_table_v03_to_v02(converted, [e.to_dict() for e in edges], "table")
        assert get_table_keys(back) == ("cells", "region", "instance_id")
        assert back.obs["instance_id"].tolist() == [1, 2, 3, 4]

    def test_row_annotating_two_elements_raises(self) -> None:
        table = _legacy_table(["cells", "cells", "nuclei", "nuclei"], [1, 2, 1, 5])
        converted, edges = convert_table_v02_to_v03(table, "table")
        converted.obs.loc["row0", "nuclei"] = 7
        with pytest.raises(ValueError, match="several elements; offending rows: \\['row0'\\]"):
            convert_table_v03_to_v02(converted, edges, "table")

    def test_unlinked_row_raises(self) -> None:
        table = _legacy_table(["cells", "cells", "nuclei", "nuclei"], [1, 2, 1, 5])
        converted, edges = convert_table_v02_to_v03(table, "table")
        converted.obs.loc["row0", "cells"] = pd.NA
        with pytest.raises(ValueError, match="annotate no element; offending rows: \\['row0'\\]"):
            convert_table_v03_to_v02(converted, edges, "table")

    def test_errors(self) -> None:
        table = _legacy_table(["cells"] * 4, [1, 2, 3, 4])
        converted, edges = convert_table_v02_to_v03(table, "table")
        with pytest.raises(ValueError, match="already carries"):
            convert_table_v03_to_v02(table, edges, "table")
        with pytest.raises(ValueError, match="No edge links the table 'other'"):
            convert_table_v03_to_v02(converted, edges, "other")
        with pytest.raises(ValueError, match=r"link column\(s\) \['missing'\]"):
            convert_table_v03_to_v02(
                converted, [Edge(from_="table", to="cells", on={"left": "missing", "right": "index"})], "table"
            )
        with pytest.raises(ValueError, match="already has a column named 'kind'"):
            convert_table_v03_to_v02(converted, edges, "table", region_key="kind")


class TestTableModelV03:
    def test_validate_and_parse_accept_version_only_metadata(self) -> None:
        adata = AnnData(X=np.zeros((2, 1)), obs=pd.DataFrame({"cells": pd.array([1, 2], dtype="Int64")}))
        adata.uns["spatialdata_attrs"] = {"version": "0.3"}
        assert TableModel.validate(adata) is adata
        assert TableModel.parse(adata) is adata
        assert not TableModel.has_region_annotation(adata)

    def test_validate_rejects_unknown_keys(self) -> None:
        adata = AnnData(X=np.zeros((2, 1)))
        adata.uns["spatialdata_attrs"] = {"version": "0.3", "instance_keys": {}}
        with pytest.raises(ValueError, match=r"unexpected keys \['instance_keys'\]"):
            TableModel.validate(adata)
        adata.uns["spatialdata_attrs"] = {"version": 3}
        with pytest.raises(ValueError, match="must be a string"):
            TableModel.validate(adata)

    def test_get_table_keys_points_to_links(self) -> None:
        adata = AnnData(X=np.zeros((2, 1)))
        adata.uns["spatialdata_attrs"] = {"version": "0.3"}
        with pytest.raises(ValueError, match="table format 0.3.*`sdata.links`"):
            get_table_keys(adata)
