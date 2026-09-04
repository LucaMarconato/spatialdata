"""
Manual smoke test for the sp-ops stack: round/t axes, relationships, RFC-8 reader.

Run from the repo root with its dev venv, on branch vibecoded-experiment/sp-ops-example
(or any branch above it in the stack):

    cd /Users/macbook/embl/projects/basel/spatialdata_luca
    .venv/bin/python examples/sp_ops/try_sp_ops_rfc8.py

Requires the two example sp-ops-conformant stores built for the compliance review that
motivated this stack, at OPS_DIR below -- not shipped with this repo.

Note: neither real store actually carries `round` as an *array axis* -- P001 is `raw`
(round is a folder level, not a dim) and Biohub is single-round `processed` data. So this
script also builds a tiny synthetic (round, c, y, x) image directly, to exercise that part
of the axis support for real.
"""

from pathlib import Path

import numpy as np

import spatialdata as sd
from spatialdata.models import Image2DModel

OPS_DIR = Path("/Users/macbook/ssd/biodata/ops/spops_conformant")


def main() -> None:
    # --- RFC-8 reader: two real conformant stores ---
    p001 = sd.read_zarr(OPS_DIR / "P001_spops.zarr")
    print(f"P001_spops.zarr (raw):        {len(p001._elements)} elements")

    biohub = sd.read_zarr(OPS_DIR / "Biohub_OPS0001_spops.zarr")
    print(f"Biohub_OPS0001_spops.zarr:    {len(biohub._elements)} elements")

    # --- repr: nested, truncated tree instead of a flat dump of all elements ---
    print("\nBiohub_OPS0001_spops.zarr repr:")
    print(biohub)

    # --- relationships: sp-ops:relationships resolved from the Biohub store ---
    print("\nsp-ops:relationships edges (Biohub):")
    for edge in biohub.relationships:
        print(f"  {edge.from_}  --{edge.method}({edge.status})-->  {edge.to}")
    print("check():", biohub.relationships.check(biohub) or "(no problems)")

    # --- round/t axes: a synthetic (round, c, y, x) image, since neither ---
    # --- real store above has `round` as an array axis (see module docstring) ---
    data = np.zeros((3, 2, 8, 8), dtype="uint16")
    img = Image2DModel.parse(data, dims=("round", "c", "y", "x"))
    print(f"\nsynthetic round-axis image dims: {img.dims}, shape: {img.shape}")
    print("img.sel(round=1, c=0) shape:", img.sel(round=1, c=0).shape)


if __name__ == "__main__":
    main()
