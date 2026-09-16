from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from shapely.geometry import Point

from spatialdata import SpatialData
from spatialdata._core.links import LINK_ATTRS_KEY, Edge, Links
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, TableModel


@pytest.fixture
def sdata() -> SpatialData:
    labels = Labels2DModel.parse(np.array([[0, 1], [2, 2]], dtype=np.int32), dims=("y", "x"))
    cells = ShapesModel.parse(
        GeoDataFrame({"roi": [0, 0]}, geometry=[Point(0, 0).buffer(1), Point(3, 3).buffer(1)], index=[1, 2])
    )
    transcripts = PointsModel.parse(pd.DataFrame({"x": [0.0, 3.0], "y": [0.0, 3.0], "cell_id": [1, 2]}))
    table = TableModel.parse(AnnData(X=np.zeros((2, 1)), obs=pd.DataFrame({"cell": [1, 2]}, index=["a", "b"])))
    return SpatialData(
        labels={"cell_labels": labels},
        shapes={"cells": cells},
        points={"transcripts": transcripts},
        tables={"table": table},
    )


class TestLinksView:
    def test_empty(self, sdata: SpatialData) -> None:
        assert isinstance(sdata.links, Links)
        assert len(sdata.links) == 0
        assert list(sdata.links) == []
        assert repr(sdata.links) == "Links: 0 edges"
        assert LINK_ATTRS_KEY not in sdata.attrs

    def test_add_writes_to_attrs(self, sdata: SpatialData) -> None:
        edge = sdata.links.add("transcripts", "cells", on=("cell_id", "index"), cardinality="n:1")
        assert sdata.attrs[LINK_ATTRS_KEY] == [edge.to_dict()]
        assert len(sdata.links) == 1
        assert sdata.links[0] == edge == sdata.links[-1]
        assert list(sdata.links) == [edge]

    def test_view_reflects_attrs_mutations(self, sdata: SpatialData) -> None:
        sdata.attrs[LINK_ATTRS_KEY] = [Edge(from_="table", to="cells", on={"left": "cell", "right": "index"}).to_dict()]
        assert len(sdata.links) == 1
        assert sdata.links[0].from_ == "table"

    def test_to_dicts_is_a_copy(self, sdata: SpatialData) -> None:
        sdata.links.add("transcripts", "cells", on=("cell_id", "index"))
        dicts = sdata.links.to_dicts()
        dicts[0]["from"] = "changed"
        assert sdata.links[0].from_ == "transcripts"

    def test_add_sjoin_defaults_to_suggested(self, sdata: SpatialData) -> None:
        edge = sdata.links.add("cells", "cell_labels", method="sjoin", predicate="within")
        assert edge.status == "suggested"
        assert edge.on == {"predicate": "within"}

    def test_predicate_requires_sjoin(self, sdata: SpatialData) -> None:
        with pytest.raises(ValueError, match="predicate"):
            sdata.links.add("cells", "cell_labels", on=("index", "value"), predicate="within")

    def test_add_on_mapping_and_invalid_on(self, sdata: SpatialData) -> None:
        edge = sdata.links.add("table", "cells", on={"left": "cell", "right": "index"})
        assert edge.on == {"left": "cell", "right": "index"}
        with pytest.raises(ValueError, match="`on` must be"):
            sdata.links.add("table", "cells", on=("a", "b", "c"))  # type: ignore[arg-type]

    def test_add_invalid_edge_message(self, sdata: SpatialData) -> None:
        with pytest.raises(ValueError, match="invalid edge: 'on' must contain"):
            sdata.links.add("table", "cells")

    def test_add_checks_elements(self, sdata: SpatialData) -> None:
        with pytest.raises(ValueError, match=r"\['nope'\] not found.*validate=False"):
            sdata.links.add("table", "nope", on=("cell", "index"))
        assert LINK_ATTRS_KEY not in sdata.attrs
        sdata.links.add("table", "nope", on=("cell", "index"), validate=False)
        assert len(sdata.links) == 1

    def test_element_targets(self, sdata: SpatialData) -> None:
        edge = sdata.links.add("table", ["cells", "cell_labels"], on=("cell", "element"))
        assert edge.targets == ("cells", "cell_labels")
        assert sdata.links.element_names == {"table", "cells", "cell_labels"}

    def test_remove(self, sdata: SpatialData) -> None:
        e0 = sdata.links.add("transcripts", "cells", on=("cell_id", "index"))
        e1 = sdata.links.add("table", "cells", on=("cell", "index"))
        assert sdata.links.remove(e0) == e0
        assert list(sdata.links) == [e1]
        with pytest.raises(ValueError, match="edge not found"):
            sdata.links.remove(e0)
        assert sdata.links.remove(0) == e1
        assert LINK_ATTRS_KEY not in sdata.attrs

    def test_for_element(self, sdata: SpatialData) -> None:
        e0 = sdata.links.add("transcripts", "cells", on=("cell_id", "index"))
        e1 = sdata.links.add("table", "cells", on=("cell", "index"))
        assert sdata.links.for_element("cells") == [e0, e1]
        assert sdata.links.for_element("cells", direction="from") == []
        assert sdata.links.for_element("table", direction="from") == [e1]
        assert sdata.links.for_element("transcripts", direction="to") == []
        with pytest.raises(ValueError, match="direction"):
            sdata.links.for_element("cells", direction="up")  # type: ignore[arg-type]

    def test_repr(self, sdata: SpatialData) -> None:
        sdata.links.add("transcripts", "cells", on=("cell_id", "index"), cardinality="n:1")
        sdata.links.add("cells", "cell_labels", method="sjoin", predicate="within")
        text = repr(sdata.links)
        assert text.splitlines()[0] == "Links: 2 edges over 3 elements"
        assert "[0] transcripts ──(cell_id ↔ index)──▶ cells" in text
        assert "join  computed  n:1" in text
        assert "[1] cells ──(within)──▶ cell_labels" in text
        assert "sjoin suggested" in text
        sdata.attrs[LINK_ATTRS_KEY].append({"from": "x"})
        assert "<invalid edge: missing keys" in repr(sdata.links)

    def test_non_list_attrs(self, sdata: SpatialData) -> None:
        sdata.attrs[LINK_ATTRS_KEY] = {"edges": []}
        assert "invalid specification" in repr(sdata.links)
        with pytest.raises(ValueError, match="must be a list of edges"):
            len(sdata.links)


class TestValidate:
    def test_links_validate(self, sdata: SpatialData) -> None:
        sdata.links.add("transcripts", "cells", on=("cell_id", "index"))
        assert sdata.links.validate() == list(sdata.links)
        sdata.attrs[LINK_ATTRS_KEY].append({"from": "x"})
        with pytest.raises(ValueError, match="edge 1: missing keys"):
            sdata.links.validate()

    def test_links_validate_missing_element(self, sdata: SpatialData) -> None:
        sdata.links.add("table", "nope", on=("cell", "index"), validate=False)
        with pytest.raises(ValueError, match="element 'nope' not found"):
            sdata.links.validate()

    def test_sdata_validate(self, sdata: SpatialData) -> None:
        sdata.validate()
        sdata.links.add("transcripts", "cells", on=("cell_id", "index"))
        sdata.validate()
        sdata.attrs[LINK_ATTRS_KEY] = "edges"
        with pytest.raises(ValueError, match="not a valid link specification"):
            sdata.validate()
