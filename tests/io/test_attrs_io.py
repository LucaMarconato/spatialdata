"""Test attrs read/write for all SpatialData container formats."""

from __future__ import annotations

import copy
import tempfile
import warnings
from pathlib import Path

import pytest
import zarr

from spatialdata import SpatialData, read_zarr
from spatialdata._core.links import LINK_ATTRS_KEY
from spatialdata._io.format import (
    SpatialDataContainerFormats,
    SpatialDataContainerFormatType,
)

FORMAT_V01 = SpatialDataContainerFormats["0.1"]
FORMAT_V02 = SpatialDataContainerFormats["0.2"]
FORMAT_V03 = SpatialDataContainerFormats["0.3"]


@pytest.mark.filterwarnings("ignore:SpatialData is not stored in the most current format:UserWarning")
@pytest.mark.parametrize("sdata_container_format", [FORMAT_V01, FORMAT_V02, FORMAT_V03])
class TestAttrsIO:
    """Test SpatialData.attrs read/write for all container formats."""

    def test_attrs_write_and_read(
        self,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        """Test attrs with complex nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"

            # Set complex attrs
            my_attrs = {
                "string_key": "test_value",
                "int_key": 123,
                "float_key": 3.14,
                "bool_key": True,
                "list_key": [1, 2, 3],
                "nested_dict": {
                    "inner_key1": None,
                    "inner_key2": 456,
                    "nested_list": ["a", "b", "c"],
                },
            }
            sdata = SpatialData()
            sdata.attrs = my_attrs

            # Write to disk with sdata_container_format
            sdata.write(f, sdata_formats=sdata_container_format)

            # Read back and verify attrs
            sdata_read = read_zarr(f)
            assert sdata_read.attrs == my_attrs

    def test_attrs_incremental_write(
        self,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        """Test incremental write of attrs using write_attrs() and write_metadata()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"

            # Initial write with attrs using sdata_container_format
            initial_attrs = {"initial_key": "initial_value", "counter": 0}
            sdata = SpatialData()
            sdata.attrs = initial_attrs
            sdata.write(f, sdata_formats=sdata_container_format)

            # Verify initial attrs
            sdata_read1 = read_zarr(f)
            assert sdata_read1.attrs == initial_attrs

            # Update attrs using write_attrs()
            sdata_read1.attrs["counter"] = 1
            sdata_read1.attrs["new_key"] = "new_value"
            sdata_read1.write_attrs(sdata_format=sdata_container_format)

            # Read back and verify updated attrs
            sdata_read2 = read_zarr(f)
            assert sdata_read2.attrs["counter"] == 1
            assert sdata_read2.attrs["new_key"] == "new_value"
            assert sdata_read2.attrs["initial_key"] == "initial_value"

            # Update attrs using write_metadata()
            sdata_read2.attrs["counter"] = 2
            sdata_read2.attrs["another_key"] = "another_value"
            sdata_read2.write_metadata(sdata_format=sdata_container_format)

            # Read back and verify all attrs
            sdata_read3 = read_zarr(f)
            assert sdata_read3.attrs["counter"] == 2
            assert sdata_read3.attrs["new_key"] == "new_value"
            assert sdata_read3.attrs["another_key"] == "another_value"
            assert sdata_read3.attrs["initial_key"] == "initial_value"


def _assert_attrs_preserved_across_formats(
    format_from: SpatialDataContainerFormatType, format_to: SpatialDataContainerFormatType
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f_from = Path(tmpdir) / "data_from.zarr"
        f_to = Path(tmpdir) / "data_to.zarr"

        my_attrs = {
            "test_key": "test_value",
            "counter": 1,
            "nested": {"inner": "value"},
        }
        sdata = SpatialData()
        sdata.attrs = my_attrs
        sdata.write(f_from, sdata_formats=format_from)

        sdata_from = read_zarr(f_from)
        assert sdata_from.attrs == my_attrs

        sdata_from.write(f_to, sdata_formats=format_to)

        sdata_to = read_zarr(f_to)
        assert sdata_to.attrs == my_attrs


@pytest.mark.filterwarnings("ignore:SpatialData is not stored in the most current format:UserWarning")
def test_attrs_v1_to_v2() -> None:
    """Test that attrs are preserved when converting from V01 to V02."""
    _assert_attrs_preserved_across_formats(FORMAT_V01, FORMAT_V02)


def test_attrs_v2_to_v3() -> None:
    """Test that attrs are preserved when converting from V02 to V03."""
    _assert_attrs_preserved_across_formats(FORMAT_V02, FORMAT_V03)


_EDGES = [
    {
        "from": "transcripts",
        "to": "cells",
        "method": "join",
        "on": {"left": "cell_id", "right": "index"},
        "status": "computed",
        "cardinality": "n:1",
    },
    {"from": "cells", "to": "rois", "method": "sjoin", "on": {"predicate": "within"}, "status": "suggested"},
]


class TestLinkAttrsIO:
    """From container format 0.3 on, ``attrs["spatialdata_link"]`` is reserved for the links between elements."""

    def test_links_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData()
            sdata.attrs[LINK_ATTRS_KEY] = copy.deepcopy(_EDGES)
            sdata.write(f)

            assert zarr.open_group(f, mode="r").attrs["spatialdata_attrs"]["version"] == "0.3"
            sdata_read = read_zarr(f)
            assert sdata_read.attrs[LINK_ATTRS_KEY] == _EDGES
            assert len(sdata_read.links) == 2

    def test_malformed_links_fail_on_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData()
            sdata.attrs[LINK_ATTRS_KEY] = [{"from": "x"}]
            with pytest.raises(ValueError, match="not a valid link specification"):
                sdata.write(f)

    def test_v02_treats_link_key_as_plain_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData()
            sdata.attrs[LINK_ATTRS_KEY] = {"not": "a list"}
            sdata.write(f, sdata_formats=FORMAT_V02)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sdata_read = read_zarr(f)
            assert not [w for w in caught if "link specification" in str(w.message)]
            assert sdata_read.attrs[LINK_ATTRS_KEY] == {"not": "a list"}

    def test_v03_tolerates_malformed_links_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            sdata = SpatialData()
            sdata.attrs[LINK_ATTRS_KEY] = copy.deepcopy(_EDGES)
            sdata.write(f)

            # corrupt the stored specification, bypassing the validation performed on write
            group = zarr.open_group(f, mode="r+")
            attrs = group.attrs.asdict()
            attrs[LINK_ATTRS_KEY] = [{"from": "x"}]
            group.attrs.put(attrs)

            with pytest.warns(UserWarning, match="link specification .* is invalid and was left untouched"):
                sdata_read = read_zarr(f)
            assert sdata_read.attrs[LINK_ATTRS_KEY] == [{"from": "x"}]
