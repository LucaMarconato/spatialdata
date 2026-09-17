"""The compatibility layer that lets the join code read both table formats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from shapely.geometry import Point

from spatialdata import SpatialData, join_spatialelement_table
from spatialdata._core.links.tables import (
    annotated_elements,
    convert_legacy_table,
    get_table_links,
    resolve_instance_ids,
    right_key_for_element,
    table_instance_keys,
    table_instance_keys_any_format,
)
from spatialdata.models import Labels2DModel, ShapesModel, TableModel


def _shapes(index: list[int]) -> GeoDataFrame:
    return ShapesModel.parse(GeoDataFrame(geometry=[Point(i, i).buffer(1) for i in index], index=index))


def _legacy_sdata() -> SpatialData:
    obs = pd.DataFrame(
        {"region": ["cells", "cells", "nuclei"], "instance_id": [1, 2, 1], "v": [10, 20, 30]},
        index=["a", "b", "c"],
    )
    table = TableModel.parse(
        AnnData(X=np.zeros((3, 1)), obs=obs),
        region=["cells", "nuclei"],
        region_key="region",
        instance_key="instance_id",
    )
    return SpatialData(
        shapes={"cells": _shapes([1, 2]), "nuclei": _shapes([1])},
        labels={"cell_labels": Labels2DModel.parse(np.array([[0, 1], [2, 2]], dtype=np.int32), dims=("y", "x"))},
        tables={"table": table},
    )


class TestLegacyTables:
    def test_resolve_instance_ids(self) -> None:
        sdata = _legacy_sdata()
        ids = resolve_instance_ids(sdata["table"], sdata=sdata, table_name="table")
        assert set(ids) == {"cells", "nuclei"}
        # the series are indexed by row position and hold the instance ids of the rows annotating each element
        assert ids["cells"].index.tolist() == [0, 1]
        assert ids["cells"].tolist() == [1, 2]
        assert ids["nuclei"].index.tolist() == [2]
        assert ids["nuclei"].tolist() == [1]

    def test_resolve_instance_ids_without_sdata(self) -> None:
        # the legacy metadata is self-contained: the links of the object are not needed
        sdata = _legacy_sdata()
        assert set(resolve_instance_ids(sdata["table"])) == {"cells", "nuclei"}

    def test_annotated_elements(self) -> None:
        sdata = _legacy_sdata()
        assert annotated_elements(sdata["table"], sdata=sdata, table_name="table") == ["cells", "nuclei"]

    def test_annotated_elements_non_empty(self) -> None:
        sdata = _legacy_sdata()
        subset = sdata["table"][sdata["table"].obs["region"] == "cells"].copy()
        # the declared regions still list both, but only one of them has rows
        assert annotated_elements(subset) == ["cells", "nuclei"]
        assert annotated_elements(subset, non_empty=True) == ["cells"]

    def test_region_key_column_wins_over_stale_metadata(self) -> None:
        sdata = _legacy_sdata()
        table = sdata["table"]
        # the column can be edited without updating the metadata; the elements found there are not lost
        table.obs["region"] = pd.Categorical(["cells", "cells", "other"])
        assert set(resolve_instance_ids(table)) == {"cells", "nuclei", "other"}

    def test_instance_keys(self) -> None:
        sdata = _legacy_sdata()
        assert table_instance_keys_any_format(sdata["table"], sdata=sdata, table_name="table") == {
            "cells": "instance_id",
            "nuclei": "instance_id",
        }
        # the legacy metadata does not produce links
        assert table_instance_keys(sdata, "table") == {}


class TestFormatV03Tables:
    @pytest.fixture
    def sdata(self) -> SpatialData:
        sdata = _legacy_sdata()
        sdata["table"] = convert_legacy_table(sdata, "table", sdata["table"])
        return sdata

    def test_convert_legacy_table(self, sdata: SpatialData) -> None:
        table = sdata["table"]
        assert table.uns["spatialdata_attrs"] == {"version": "0.3"}
        assert list(table.obs.columns) == ["v", "cells", "nuclei"]
        assert table.obs["cells"].tolist()[:2] == [1, 2]
        assert table.obs["cells"].isna().tolist() == [False, False, True]
        assert [(e.from_, e.to, e.on) for e in sdata.links] == [
            ("table", "cells", {"left": "cells", "right": "index"}),
            ("table", "nuclei", {"left": "nuclei", "right": "index"}),
        ]

    def test_links_and_instance_keys(self, sdata: SpatialData) -> None:
        assert set(get_table_links(sdata, "table")) == {"cells", "nuclei"}
        assert table_instance_keys(sdata, "table") == {"cells": "cells", "nuclei": "nuclei"}
        assert table_instance_keys_any_format(sdata["table"], sdata=sdata, table_name="table") == {
            "cells": "cells",
            "nuclei": "nuclei",
        }

    def test_resolve_instance_ids(self, sdata: SpatialData) -> None:
        ids = resolve_instance_ids(sdata["table"], sdata=sdata, table_name="table")
        assert ids["cells"].index.tolist() == [0, 1]
        assert ids["cells"].tolist() == [1, 2]
        assert ids["nuclei"].index.tolist() == [2]
        assert ids["nuclei"].tolist() == [1]

    def test_resolve_instance_ids_needs_the_links(self, sdata: SpatialData) -> None:
        with pytest.raises(ValueError, match="table format 0.3"):
            resolve_instance_ids(sdata["table"])
        # ...unless the columns are passed explicitly
        ids = resolve_instance_ids(sdata["table"], instance_keys={"cells": "cells"})
        assert ids["cells"].tolist() == [1, 2]

    def test_resolve_instance_ids_missing_column(self, sdata: SpatialData) -> None:
        with pytest.raises(ValueError, match="link column 'missing'"):
            resolve_instance_ids(sdata["table"], instance_keys={"cells": "missing"})

    def test_annotated_elements(self, sdata: SpatialData) -> None:
        assert annotated_elements(sdata["table"], sdata=sdata, table_name="table") == ["cells", "nuclei"]

    def test_right_key_for_element(self, sdata: SpatialData) -> None:
        # labels join on their pixel values, everything else on the index
        assert right_key_for_element(sdata, "cell_labels") == "value"
        assert right_key_for_element(sdata, "cells") == "index"
        assert right_key_for_element(sdata, "does_not_exist") == "index"


class TestJoinsAreFormatAgnostic:
    @pytest.mark.parametrize("how", ["left", "inner", "right", "left_exclusive", "right_exclusive"])
    def test_join_gives_the_same_result_in_both_formats(self, how: str) -> None:
        legacy = _legacy_sdata()
        # drop one instance so that the exclusive joins have something to return
        legacy["cells"] = legacy["cells"].iloc[:1]
        converted = _legacy_sdata()
        converted["cells"] = converted["cells"].iloc[:1]
        converted["table"] = convert_legacy_table(converted, "table", converted["table"])

        legacy_elements, legacy_table = join_spatialelement_table(
            sdata=legacy, spatial_element_names="cells", table_name="table", how=how
        )
        new_elements, new_table = join_spatialelement_table(
            sdata=converted, spatial_element_names="cells", table_name="table", how=how
        )

        assert (legacy_table is None) == (new_table is None)
        if legacy_table is not None:
            assert legacy_table.obs_names.tolist() == new_table.obs_names.tolist()
        assert (legacy_elements["cells"] is None) == (new_elements["cells"] is None)
        if legacy_elements["cells"] is not None:
            assert legacy_elements["cells"].index.tolist() == new_elements["cells"].index.tolist()

    def test_join_without_sdata_needs_instance_keys(self) -> None:
        sdata = _legacy_sdata()
        sdata["table"] = convert_legacy_table(sdata, "table", sdata["table"])
        with pytest.raises(ValueError, match="table format 0.3"):
            join_spatialelement_table(
                spatial_element_names=["cells"], spatial_elements=[sdata["cells"]], table=sdata["table"], how="left"
            )
        _, table = join_spatialelement_table(
            spatial_element_names=["cells"],
            spatial_elements=[sdata["cells"]],
            table=sdata["table"],
            how="left",
            instance_keys={"cells": "cells"},
        )
        assert table.obs_names.tolist() == ["a", "b"]
