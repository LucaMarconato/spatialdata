import numpy as np

from spatialdata import SpatialData
from spatialdata._core.spatialdata import (
    _REPR_TREE_MAX_CHILDREN_SHOWN,
    _insert_into_tree,
    _render_tree,
)
from spatialdata.models import Image2DModel


class TestInsertIntoTree:
    def test_nests_by_slash(self) -> None:
        tree: dict = {}
        _insert_into_tree(tree, ["a", "b", "c"], "leaf")
        assert tree == {"a": {"b": {"c": "leaf"}}}

    def test_siblings_share_a_folder(self) -> None:
        tree: dict = {}
        _insert_into_tree(tree, ["a", "b"], "leaf-b")
        _insert_into_tree(tree, ["a", "c"], "leaf-c")
        assert tree == {"a": {"b": "leaf-b", "c": "leaf-c"}}

    def test_flat_leaf_at_root(self) -> None:
        tree: dict = {}
        _insert_into_tree(tree, ["library"], "leaf")
        assert tree == {"library": "leaf"}


class TestRenderTree:
    def test_no_truncation_below_threshold(self) -> None:
        tree = {f"channel{i}": "" for i in range(_REPR_TREE_MAX_CHILDREN_SHOWN + 1)}
        lines = _render_tree(tree)
        assert len(lines) == _REPR_TREE_MAX_CHILDREN_SHOWN + 1
        assert not any("more" in line for line in lines)

    def test_truncates_long_sibling_lists(self) -> None:
        n = 12
        tree = {f"channel{i}": "" for i in range(n)}
        lines = _render_tree(tree)
        # shown children + one "... N more" line
        assert len(lines) == _REPR_TREE_MAX_CHILDREN_SHOWN + 1
        assert lines[-1].endswith(f"... {n - _REPR_TREE_MAX_CHILDREN_SHOWN} more")
        # the last connector is "└──", including on the truncation line, and it's the only one
        assert sum(1 for line in lines if "└──" in line) == 1
        assert lines[-1].strip().startswith("└──")

    def test_last_shown_child_is_not_marked_last_when_truncated(self) -> None:
        tree = {f"channel{i}": "" for i in range(12)}
        lines = _render_tree(tree)
        for line in lines[:-1]:
            assert "├──" in line

    def test_recurses_into_subfolders(self) -> None:
        tree = {"a": {"b": "leaf"}}
        lines = _render_tree(tree)
        assert lines == ["└── a/ (1 element)", "    └── b: leaf"]


class TestSpatialDataReprTree:
    @staticmethod
    def _images_sdata(n_rounds: int, n_channels: int) -> SpatialData:
        images = {}
        for r in range(n_rounds):
            for c in range(n_channels):
                images[f"sample/tile0/round{r}/channel{c}"] = Image2DModel.parse(
                    np.zeros((1, 4, 4), dtype="uint8"), dims=("c", "y", "x")
                )
        return SpatialData(images=images)

    def test_repr_respects_hierarchy(self) -> None:
        sdata = self._images_sdata(n_rounds=1, n_channels=1)
        text = repr(sdata)
        assert "└── sample/ (1 element)" in text
        assert "└── tile0/ (1 element)" in text
        assert "└── round0/ (1 element)" in text
        assert "channel0: [Image2D]" in text
        # the flat "sample/tile0/round0/channel0: ..." line from the old repr must be gone
        assert "sample/tile0/round0/channel0:" not in text

    def test_repr_truncates_many_channels(self) -> None:
        n_channels = 12
        sdata = self._images_sdata(n_rounds=1, n_channels=n_channels)
        text = repr(sdata)
        assert f"... {n_channels - _REPR_TREE_MAX_CHILDREN_SHOWN} more" in text
        # an elided channel must not appear anywhere in the output
        assert f"channel{n_channels - 1}:" not in text

    def test_coordinate_systems_section_is_also_a_truncated_tree(self) -> None:
        n_channels = 12
        sdata = self._images_sdata(n_rounds=1, n_channels=n_channels)
        text = repr(sdata)
        cs_section = text.split("with coordinate systems:")[1]
        assert "└── sample/ (1 element)" in cs_section
        assert f"... {n_channels - _REPR_TREE_MAX_CHILDREN_SHOWN} more" in cs_section
        assert f"channel{n_channels - 1}" not in cs_section
