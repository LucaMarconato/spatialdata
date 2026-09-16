from __future__ import annotations

from typing import Any

import pytest

from spatialdata._core.links import ELEMENT, INDEX, LINK_ATTRS_KEY, VALUE, Edge, validate_link_spec


def _join_edge(**kwargs: Any) -> Edge:
    base: dict[str, Any] = {
        "from_": "transcripts",
        "to": "cells",
        "method": "join",
        "on": {"left": "cell_id", "right": INDEX},
        "cardinality": "n:1",
    }
    base.update(kwargs)
    return Edge(**base)


class TestEdge:
    def test_join_edge_to_dict_and_back(self) -> None:
        edge = _join_edge()
        d = edge.to_dict()
        assert d == {
            "from": "transcripts",
            "to": "cells",
            "method": "join",
            "on": {"left": "cell_id", "right": "index"},
            "status": "computed",
            "cardinality": "n:1",
        }
        assert Edge.from_dict(d) == edge

    def test_cardinality_is_optional(self) -> None:
        assert "cardinality" not in _join_edge(cardinality=None).to_dict()

    def test_sjoin_edge(self) -> None:
        edge = Edge(from_="cells", to="rois", method="sjoin", on={"predicate": "within"}, status="suggested")
        assert edge.to_dict()["on"] == {"predicate": "within"}
        assert Edge.from_dict(edge.to_dict()) == edge

    def test_targets_and_elements(self) -> None:
        edge = Edge(from_="meta", to=["im1", "im2"], on={"left": "image", "right": ELEMENT})
        assert edge.to == ("im1", "im2")
        assert edge.targets == ("im1", "im2")
        assert edge.elements == ("meta", "im1", "im2")
        assert edge.to_dict()["to"] == ["im1", "im2"]
        assert _join_edge().targets == ("cells",)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "merge"}, "'method'"),
            ({"status": "maybe"}, "'status'"),
            ({"cardinality": "one-to-many"}, "'cardinality'"),
            ({"on": {"left": "cell_id"}}, r"missing \['right'\]"),
            ({"on": {"left": "", "right": "index"}}, r"on\['left'\]"),
            ({"on": "cell_id"}, "'on' must be a mapping"),
            ({"from_": ""}, "'from'"),
            ({"to": ""}, "'to'"),
            ({"to": ["a", "b"]}, r"on\['right'\] == 'element'"),
            ({"method": "sjoin", "on": {"left": "a", "right": "b"}}, "'predicate'"),
            ({"method": "sjoin", "on": {"predicate": "within"}, "to": ["a", "b"]}, "single element"),
        ],
    )
    def test_invalid_edges(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _join_edge(**kwargs)

    def test_from_dict_errors(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            Edge.from_dict(["from"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="missing keys 'to', 'method', 'on'"):
            Edge.from_dict({"from": "x"})
        with pytest.raises(ValueError, match="unknown keys 'weight'"):
            Edge.from_dict({**_join_edge().to_dict(), "weight": 1})

    def test_from_dict_defaults(self) -> None:
        d = {"from": "a", "to": "b", "method": "join", "on": {"left": INDEX, "right": VALUE}}
        edge = Edge.from_dict(d)
        assert edge.status == "computed"
        assert edge.cardinality is None


class TestValidateLinkSpec:
    def test_returns_edges(self) -> None:
        assert validate_link_spec([_join_edge().to_dict()]) == [_join_edge()]

    def test_not_a_list(self) -> None:
        match = rf"attrs\['{LINK_ATTRS_KEY}'\] is not a valid link specification: expected a list"
        with pytest.raises(ValueError, match=match):
            validate_link_spec({"edges": []})

    def test_reports_all_malformed_edges(self) -> None:
        spec = [{"from": "x"}, _join_edge().to_dict(), {"from": "a", "to": "b", "method": "nope", "on": {}}]
        with pytest.raises(ValueError) as excinfo:
            validate_link_spec(spec)
        message = str(excinfo.value)
        assert "edge 0: missing keys" in message
        assert "edge 2:" in message
        assert "edge 1" not in message

    def test_element_names(self) -> None:
        spec = [_join_edge().to_dict()]
        assert validate_link_spec(spec, element_names=["transcripts", "cells"])
        with pytest.raises(ValueError, match="element 'cells' not found"):
            validate_link_spec(spec, element_names=["transcripts"])
