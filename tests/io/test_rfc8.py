from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData

import spatialdata as sd

# The two example stores from the sp-ops compliance exercise (see the project's own
# `spops_conformant/` build scripts). Skipped if the data isn't present on this machine.
_DATA_DIR = Path("/Users/macbook/ssd/biodata/ops/spops_conformant")
P001_STORE = _DATA_DIR / "P001_spops.zarr"
BIOHUB_STORE = _DATA_DIR / "Biohub_OPS0001_spops.zarr"

pytestmark = pytest.mark.filterwarnings("ignore")


def _make_anndata(*args, **kwargs) -> AnnData:
    # see tests/core/test_relationships.py for why this is needed on anndata>=0.13
    adata = AnnData(*args, **kwargs)
    adata.layers.pop(None, None)
    return adata


def _write_collection(group: zarr.Group, name: str, nodes: list[dict], extra_attributes: dict | None = None) -> None:
    group.attrs["ome"] = {
        "version": "0.6-rfc8-draft",
        "type": "collection",
        "name": name,
        "nodes": nodes,
        "attributes": extra_attributes or {},
    }


def _write_synthetic_rfc8_store(tmp_path: Path) -> Path:
    """Build a minimal RFC-8 store mirroring layout.md's example, with a `round`-axis image."""
    root_path = tmp_path / "synthetic.zarr"
    root = zarr.open_group(str(root_path), mode="w")

    library = _make_anndata(X=np.zeros((2, 1)), obs=pd.DataFrame({"barcode": ["AAA", "CCC"]}))
    library.write_zarr(str(root_path / "library"))

    plate = root.create_group("plate1_processed")
    well = plate.create_group("A")
    well = well.create_group("1")

    image_group = well.create_group("image")
    # (round, c, y, x): two rounds, two channels, 4x4 pixels, matching D6's axis order.
    data = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4)
    image_group.create_array(name="0", data=data)
    image_group.attrs["ome"] = {
        "version": "0.6-rfc8-draft",
        "type": "multiscale",
        "multiscales": [
            {
                "version": "0.6-rfc8-draft",
                "axes": [{"name": n} for n in ("round", "c", "y", "x")],
                "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1]}]}],
            }
        ],
        "attributes": {
            "sp-ops:rounds": [{"index": 0, "acquisition": {"id": "iss-c1"}}, {"index": 1, "acquisition": {"id": "iss-c2"}}],
            "sp-ops:channels": [{"name": "DAPI", "role": "nuclear"}, {"name": "A", "role": "base"}],
        },
    }

    features = _make_anndata(X=np.zeros((2, 1)), obs=pd.DataFrame({"barcode": ["AAA", "CCC"]}))
    features.write_zarr(str(root_path / "plate1_processed/A/1/features"))

    _write_collection(
        well,
        "A/1",
        nodes=[
            {"type": "multiscale", "name": "image", "path": {"type": "zarr", "path": "./image"}},
            {"type": "sp-ops:table", "name": "features", "path": {"type": "zarr", "path": "./features"}},
        ],
        extra_attributes={
            "sp-ops:relationships": {
                "version": "0.2.0-draft",
                "edges": [
                    {
                        "from": "features",
                        "to": "../../../library",
                        "method": "join",
                        "on": {"left": "barcode", "right": "barcode"},
                        "status": "computed",
                        "cardinality": "n:1",
                    }
                ],
            }
        },
    )
    _write_collection(plate, "plate1_processed", nodes=[{"type": "collection", "name": "A/1", "path": {"type": "zarr", "path": "./A/1"}}])
    _write_collection(
        root,
        "screen",
        nodes=[
            {"type": "sp-ops:table", "name": "library", "path": {"type": "zarr", "path": "./library"}},
            {"type": "collection", "name": "plate1_processed", "path": {"type": "zarr", "path": "./plate1_processed"}},
        ],
        extra_attributes={"sp-ops:spec": {"version": "0.2.0-draft"}},
    )
    return root_path


class TestSyntheticRfc8Store:
    def test_read_zarr_succeeds_and_builds_hierarchy(self, tmp_path: Path) -> None:
        store_path = _write_synthetic_rfc8_store(tmp_path)
        sdata = sd.read_zarr(store_path)
        assert set(sdata._elements.keys()) == {
            "library",
            "plate1_processed/A/1/image",
            "plate1_processed/A/1/features",
        }

    def test_round_axis_image_reads_with_round_dim(self, tmp_path: Path) -> None:
        store_path = _write_synthetic_rfc8_store(tmp_path)
        sdata = sd.read_zarr(store_path)
        image = sdata["plate1_processed/A/1/image"]
        assert image.dims == ("round", "c", "y", "x")
        assert image.sel(round=1, c="A").shape == (4, 4)

    def test_prefix_subview(self, tmp_path: Path) -> None:
        store_path = _write_synthetic_rfc8_store(tmp_path)
        sdata = sd.read_zarr(store_path)
        well = sdata["plate1_processed/A/1"]
        assert set(well._elements.keys()) == {"image", "features"}

    def test_relationships_resolved_and_queryable(self, tmp_path: Path) -> None:
        store_path = _write_synthetic_rfc8_store(tmp_path)
        sdata = sd.read_zarr(store_path)
        edges = sdata.relationships.for_element("plate1_processed/A/1/features", direction="from")
        assert len(edges) == 1
        edge = edges[0]
        assert edge.to == "library"
        merged = sdata.relationships.resolve(sdata, edge)
        assert sorted(merged["barcode"].tolist()) == ["AAA", "CCC"]


@pytest.mark.skipif(not P001_STORE.exists(), reason="P001_spops.zarr example store not present on this machine")
class TestP001ExampleStore:
    def test_read_zarr_succeeds(self) -> None:
        sdata = sd.read_zarr(P001_STORE)
        assert len(sdata._elements) == 240

    def test_prefix_subview_on_a_tile(self) -> None:
        sdata = sd.read_zarr(P001_STORE)
        tile = sdata["plate1_raw/A/1/iss/tiles/tile0"]
        assert "round0/channel0" in tile._elements
        channel = tile["round0/channel0"]
        assert channel.dims == ("c", "y", "x")

    def test_layout_shapes_read(self) -> None:
        sdata = sd.read_zarr(P001_STORE)
        layout = sdata["plate1_raw/A/1/iss/tiles/layout"]
        assert len(layout) == 2

    def test_no_relationships_in_raw_only_store(self) -> None:
        sdata = sd.read_zarr(P001_STORE)
        assert len(sdata.relationships) == 0


@pytest.mark.skipif(not BIOHUB_STORE.exists(), reason="Biohub_OPS0001_spops.zarr example store not present on this machine")
class TestBiohubExampleStore:
    def test_read_zarr_succeeds(self) -> None:
        sdata = sd.read_zarr(BIOHUB_STORE)
        assert len(sdata._elements) == 19

    def test_merged_image_is_a_6_channel_image(self) -> None:
        sdata = sd.read_zarr(BIOHUB_STORE)
        image = sdata["plate1_processed/A/1/pheno/merged/image"]
        assert image["scale0"].ds["image"].sizes["c"] == 6

    def test_qc_overlay_reclassified_as_image_not_labels(self) -> None:
        # grid_overlay/iss_gene_image/iss_guide_image are RGBA renderings marked as `labels`
        # nodes in the source data, but don't fit spatialdata's Labels model (no `c` axis
        # allowed there) -- the reader must treat them as Images instead. See
        # `_read_rfc8_multiscale`'s docstring.
        sdata = sd.read_zarr(BIOHUB_STORE)
        from spatialdata.models import Image2DModel, get_model

        overlay = sdata["plate1_processed/A/1/pheno/merged/grid_overlay"]
        assert get_model(overlay) is Image2DModel

    def test_segmentation_reads_as_labels(self) -> None:
        from spatialdata.models import Labels2DModel, get_model

        sdata = sd.read_zarr(BIOHUB_STORE)
        cell_seg = sdata["plate1_processed/A/1/pheno/merged/cell_seg"]
        assert get_model(cell_seg) is Labels2DModel

    def test_relationships_resolved(self) -> None:
        sdata = sd.read_zarr(BIOHUB_STORE)
        edges = sdata.relationships.for_element("library")
        assert len(edges) >= 1
        assert all(e.to == "library" for e in edges)
