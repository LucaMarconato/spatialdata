import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from shapely.geometry import Point

from spatialdata import SpatialData
from spatialdata._core.relationships import Edge, Relationships
from spatialdata.models import Labels2DModel, ShapesModel, TableModel


def _make_anndata(*args, **kwargs) -> AnnData:
    # Work around an anndata>=0.13 quirk (unrelated to this PR) where a freshly constructed
    # AnnData reports a phantom `None`-keyed layer, which trips
    # `spatialdata._core.validation.check_all_keys_case_insensitively_unique` inside
    # `TableModel.parse`. This affects table construction throughout the existing test suite on
    # this branch already (e.g. `tests/models/test_models.py::test_table_model_invalid_names`),
    # not something introduced here.
    adata = AnnData(*args, **kwargs)
    adata.layers.pop(None, None)
    return adata


def _cells_and_features_sdata() -> SpatialData:
    # mirrors the sp-ops spec's `pheno/merged/{cells, cells_features}` example
    # (docs/joinable-components.md): a labels element and a table joined by pixel value / label.
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 0, 0],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    cells = Labels2DModel.parse(labels, dims=("y", "x"))

    obs = pd.DataFrame({"label": [1, 2, 3], "area": [4, 4, 2]})
    cells_features = TableModel.parse(_make_anndata(X=np.zeros((3, 1)), obs=obs))

    return SpatialData(labels={"cells": cells}, tables={"cells_features": cells_features})


def _key_join_edge() -> Edge:
    return Edge(
        from_="cells",
        to="cells_features",
        method="join",
        on={"left": "value", "right": "label"},
        status="computed",
        cardinality="1:1",
    )


class TestEdge:
    def test_to_dict_matches_spec_shape(self) -> None:
        edge = _key_join_edge()
        assert edge.to_dict() == {
            "from": "cells",
            "to": "cells_features",
            "method": "join",
            "on": {"left": "value", "right": "label"},
            "status": "computed",
            "cardinality": "1:1",
        }

    def test_from_dict_roundtrip(self) -> None:
        d = _key_join_edge().to_dict()
        edge = Edge.from_dict(d)
        assert edge == _key_join_edge()

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            Edge(from_="a", to="b", method="merge", on={})  # type: ignore[arg-type]

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Edge(from_="a", to="b", method="join", on={}, status="maybe")  # type: ignore[arg-type]

    def test_invalid_cardinality_raises(self) -> None:
        with pytest.raises(ValueError, match="cardinality"):
            Edge(from_="a", to="b", method="join", on={}, cardinality="one-to-many")

    @pytest.mark.parametrize("cardinality", ["1:1", "n:1", "1:n", "n:n", "3:1"])
    def test_valid_cardinality_accepted(self, cardinality: str) -> None:
        Edge(from_="a", to="b", method="join", on={}, cardinality=cardinality)


class TestRelationships:
    def test_to_dicts_from_dicts_roundtrip(self) -> None:
        edges = [
            _key_join_edge(),
            Edge(from_="reads", to="cells", method="sjoin", on={"predicate": "within"}, status="suggested"),
        ]
        rel = Relationships(edges)
        dicts = rel.to_dicts()
        assert dicts == [e.to_dict() for e in edges]
        assert Relationships.from_dicts(dicts) == rel

    def test_for_element_direction(self) -> None:
        e1 = _key_join_edge()
        e2 = Edge(from_="cells_features", to="library", method="join", on={"left": "barcode", "right": "barcode"})
        rel = Relationships([e1, e2])
        assert rel.for_element("cells_features", direction="from") == Relationships([e2])
        assert rel.for_element("cells_features", direction="to") == Relationships([e1])
        assert rel.for_element("cells_features", direction="either") == Relationships([e1, e2])
        assert rel.for_element("nonexistent") == Relationships([])

    def test_for_element_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            Relationships([_key_join_edge()]).for_element("cells", direction="up")  # type: ignore[arg-type]

    def test_add_and_len_and_iter(self) -> None:
        rel = Relationships()
        assert len(rel) == 0
        rel.add(_key_join_edge())
        assert len(rel) == 1
        assert list(rel) == [_key_join_edge()]

    def test_check_flags_missing_endpoints(self) -> None:
        sdata = _cells_and_features_sdata()
        edge = Edge(from_="cells", to="does_not_exist", method="join", on={"left": "value", "right": "label"})
        problems = Relationships([edge]).check(sdata)
        assert len(problems) == 1
        assert "does_not_exist" in problems[0]

    def test_check_flags_malformed_cardinality(self) -> None:
        sdata = _cells_and_features_sdata()
        edge = _key_join_edge()
        # bypass __post_init__ validation to simulate a malformed edge loaded from an external source
        bad_edge = object.__new__(Edge)
        object.__setattr__(bad_edge, "from_", edge.from_)
        object.__setattr__(bad_edge, "to", edge.to)
        object.__setattr__(bad_edge, "method", edge.method)
        object.__setattr__(bad_edge, "on", edge.on)
        object.__setattr__(bad_edge, "status", edge.status)
        object.__setattr__(bad_edge, "cardinality", "not-a-cardinality")
        problems = Relationships([bad_edge]).check(sdata)
        assert len(problems) == 1
        assert "cardinality" in problems[0]

    def test_check_suggested_status_is_not_a_problem(self) -> None:
        sdata = _cells_and_features_sdata()
        edge = Edge(from_="cells", to="cells_features", method="sjoin", on={"predicate": "within"}, status="suggested")
        assert Relationships([edge]).check(sdata) == []

    def test_resolve_labels_value_join(self) -> None:
        sdata = _cells_and_features_sdata()
        edge = _key_join_edge()
        merged = Relationships([edge]).resolve(sdata, edge)
        assert sorted(merged["value"].tolist()) == [1, 2, 3]
        assert sorted(merged["label"].tolist()) == [1, 2, 3]
        assert set(merged["area"]) == {4, 4, 2}

    def test_resolve_plain_column_join(self) -> None:
        left = TableModel.parse(_make_anndata(X=np.zeros((2, 1)), obs=pd.DataFrame({"barcode": ["AAA", "CCC"]})))
        right = TableModel.parse(
            _make_anndata(X=np.zeros((2, 1)), obs=pd.DataFrame({"barcode": ["AAA", "CCC"], "perturbation": ["g1", "g2"]}))
        )
        sdata = SpatialData(tables={"reads": left, "library": right})
        edge = Edge(from_="reads", to="library", method="join", on={"left": "barcode", "right": "barcode"})
        merged = Relationships([edge]).resolve(sdata, edge)
        assert sorted(merged["perturbation"].tolist()) == ["g1", "g2"]

    def test_resolve_sjoin(self) -> None:
        points = GeoDataFrame({"id": [0, 1], "radius": [0.1, 0.1]}, geometry=[Point(0, 0), Point(10, 10)])
        polygons = GeoDataFrame(
            {"id": [0]}, geometry=[Point(0, 0).buffer(1)]
        )
        sdata = SpatialData(shapes={"reads": ShapesModel.parse(points), "cells": ShapesModel.parse(polygons)})
        edge = Edge(from_="reads", to="cells", method="sjoin", on={"predicate": "within"}, status="suggested")
        result = Relationships([edge]).resolve(sdata, edge)
        assert len(result) == 1


class TestSpatialDataRelationshipsProperty:
    def test_default_is_empty(self) -> None:
        sdata = SpatialData()
        assert isinstance(sdata.relationships, Relationships)
        assert len(sdata.relationships) == 0

    def test_constructor_argument(self) -> None:
        rel = Relationships([_key_join_edge()])
        sdata = SpatialData(relationships=rel)
        assert sdata.relationships == rel

    def test_setter(self) -> None:
        sdata = SpatialData()
        rel = Relationships([_key_join_edge()])
        sdata.relationships = rel
        assert sdata.relationships == rel

    def test_setter_rejects_non_relationships(self) -> None:
        sdata = SpatialData()
        with pytest.raises(TypeError, match="Relationships"):
            sdata.relationships = [_key_join_edge()]  # type: ignore[assignment]
