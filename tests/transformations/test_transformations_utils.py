from __future__ import annotations

import numpy as np
from dask.array.core import from_array
from xarray import DataArray

from spatialdata.transformations._utils import compute_coordinates, convert_transformations_to_affine
from spatialdata.transformations.operations import get_transformation, set_transformation
from spatialdata.transformations.transformations import Affine, Scale, Sequence, Translation


def test_compute_coordinates_round_and_t_axes():
    """`round`/`t` get plain integer coordinates (unlike x/y/z's pixel-center offset), so `.sel(round=0)` works."""
    data = DataArray(from_array(np.zeros((2, 3, 4, 5))), dims=("round", "c", "y", "x"))
    out = compute_coordinates(data)
    np.testing.assert_array_equal(out.coords["round"].values, np.arange(2))
    assert out.coords["y"].values[0] == 0.5
    assert out.coords["x"].values[0] == 0.5


def test_convert_transformations_to_affine(full_sdata):
    translation = Translation([1, 2, 3], axes=("x", "y", "z"))
    scale = Scale([1, 2, 3], axes=("x", "y", "z"))
    sequence = Sequence([translation, scale])
    for _, _, element in full_sdata.gen_spatial_elements():
        set_transformation(element, transformation=sequence, to_coordinate_system="test")
    convert_transformations_to_affine(full_sdata, "test")
    for _, _, element in full_sdata.gen_spatial_elements():
        t = get_transformation(element, "test")
        assert isinstance(t, Affine)
