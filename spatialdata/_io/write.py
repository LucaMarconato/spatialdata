import os
from collections.abc import Mapping
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from shapely.io import to_ragged_array
from spatial_image import SpatialImage

from spatialdata._core.core_utils import (
    ValidAxis_t,
    _get_transform,
    _get_transform_xarray,
    get_dims,
)
from spatialdata._core.transformations import (
    BaseTransformation,
    _get_current_output_axes,
)
from spatialdata._io.format import (
    PointsFormat,
    PolygonsFormat,
    ShapesFormat,
    SpatialDataFormatV01,
)

__all__ = ["write_image", "write_labels", "write_points", "write_polygons", "write_table"]


def _overwrite_coordinate_transformations_non_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformation: BaseTransformation
) -> None:
    output_axes = _get_current_output_axes(transformation=transformation, input_axes=axes)
    ngff_transformation = transformation.to_ngff(input_axes=axes, output_axes=output_axes)
    coordinate_transformations = [ngff_transformation.to_dict()]
    group.attrs["coordinateTransformations"] = coordinate_transformations


def _overwrite_coordinate_transformations_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformations: list[BaseTransformation]
) -> None:
    # prepare the transformations in the dict representation
    ngff_transformations = []
    for t in transformations:
        output_axes = _get_current_output_axes(transformation=t, input_axes=tuple(axes))
        ngff_transformations.append(t.to_ngff(input_axes=tuple(axes), output_axes=tuple(output_axes)))
    coordinate_transformations = [[t.to_dict()] for t in ngff_transformations]
    # replace the metadata storage
    multiscales = group.attrs["multiscales"]
    assert len(multiscales) == 1
    multiscale = multiscales[0]
    datasets = multiscale["datasets"]
    assert len(datasets) == len(transformations)
    for i in range(len(transformations)):
        dataset = datasets[i]
        assert dataset["path"] == str(i)
        ct = coordinate_transformations[i]
        dataset["coordinateTransformations"] = ct
    group.attrs["multiscales"] = multiscales


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    # coordinate_transformations: list[dict[str, Any]],
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
    fmt: Format = SpatialDataFormatV01(),
) -> None:
    """Write metdata to a group."""
    axes = _get_valid_axes(axes=axes, fmt=fmt)

    group.attrs["@type"] = group_type
    group.attrs["axes"] = axes
    # we write empty coordinateTransformations and then overwrite them with _overwrite_coordinate_transformations_non_raster()
    group.attrs["coordinateTransformations"] = []
    # group.attrs["coordinateTransformations"] = coordinate_transformations
    group.attrs["spatialdata_attrs"] = attrs


# TODO: write_image and write_labels are very similar, refactor and simplify
def write_image(
    image: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = SpatialDataFormatV01(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:
    subgroup = group.require_group(name)
    if isinstance(image, SpatialImage):
        data = image.data
        t = _get_transform(image)
        axes: tuple[str, ...] = tuple(image.dims)
        chunks = image.chunks
        parsed_axes = _get_valid_axes(axes=list(axes), fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        # scaler needs to be None since we are passing the data already downscaled for the multiscale case
        write_image_ngff(
            image=data,
            group=subgroup,
            scaler=None,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            **metadata,
        )
        assert t is not None
        _overwrite_coordinate_transformations_raster(group=subgroup, transformations=[t], axes=axes)
    elif isinstance(image, MultiscaleSpatialImage):
        data = _iter_multiscale(image, "data")
        list_of_input_axes: list[Any] = _iter_multiscale(image, "dims")
        assert len(set(list_of_input_axes)) == 1
        input_axes = list_of_input_axes[0]
        transformations = []
        for k in image.keys():
            d = dict(image[k])
            assert len(d) == 1
            xdata = d.values().__iter__().__next__()
            transformation = _get_transform_xarray(xdata)
            assert transformation is not None
            transformations.append(transformation)
        chunks = _iter_multiscale(image, "chunks")
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multiscale_ngff(
            pyramid=data,
            group=subgroup,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            name=name,
            **metadata,
        )
        _overwrite_coordinate_transformations_raster(
            group=subgroup, transformations=transformations, axes=tuple(input_axes)
        )
    else:
        raise ValueError("Not a valid image object")


def write_labels(
    labels: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = SpatialDataFormatV01(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if isinstance(labels, SpatialImage):
        data = labels.data
        t = _get_transform(labels)
        input_axes: tuple[str, ...] = tuple(labels.dims)
        chunks = labels.chunks
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        # scaler needs to be None since we are passing the data already downscaled for the multiscale case
        write_labels_ngff(
            labels=data,
            name=name,
            group=group,
            scaler=None,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            label_metadata=label_metadata,
            **metadata,
        )
        assert t is not None
        _overwrite_coordinate_transformations_raster(group=group["labels"][name], transformations=[t], axes=input_axes)
    elif isinstance(labels, MultiscaleSpatialImage):
        data = _iter_multiscale(labels, "data")
        list_of_input_axes: list[Any] = _iter_multiscale(labels, "dims")
        assert len(set(list_of_input_axes)) == 1
        input_axes = list_of_input_axes[0]
        transformations = []
        for k in labels.keys():
            d = dict(labels[k])
            assert len(d) == 1
            xdata = d.values().__iter__().__next__()
            transformation = _get_transform_xarray(xdata)
            assert transformation is not None
            transformations.append(transformation)
        chunks = _iter_multiscale(labels, "chunks")
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multiscale_labels_ngff(
            pyramid=data,
            group=group,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            name=name,
            label_metadata=label_metadata,
            **metadata,
        )
        _overwrite_coordinate_transformations_raster(
            group=group["labels"][name], transformations=transformations, axes=tuple(input_axes)
        )
    else:
        raise ValueError("Not a valid labels object")


def write_polygons(
    polygons: GeoDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:polygons",
    fmt: Format = PolygonsFormat(),
) -> None:
    axes = get_dims(polygons)
    t = _get_transform(polygons)
    polygons_groups = group.require_group(name)
    geometry, coords, offsets = to_ragged_array(polygons.geometry)
    polygons_groups.create_dataset(name="coords", data=coords)
    for i, o in enumerate(offsets):
        polygons_groups.create_dataset(name=f"offset{i}", data=o)
    polygons_groups.create_dataset(name="Index", data=polygons.index.values)

    attrs = fmt.attrs_to_dict(geometry)
    attrs["version"] = fmt.spatialdata_version

    _write_metadata(
        polygons_groups,
        group_type=group_type,
        # coordinate_transformations=coordinate_transformations,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    _overwrite_coordinate_transformations_non_raster(group=polygons_groups, axes=axes, transformation=t)


def write_shapes(
    shapes: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = ShapesFormat(),
) -> None:
    axes = get_dims(shapes)
    transform = shapes.uns.pop("transform")
    assert transform is not None
    write_adata(group, name, shapes)  # creates group[name]
    shapes.uns["transform"] = transform

    attrs = fmt.attrs_to_dict(shapes.uns)
    attrs["version"] = fmt.spatialdata_version

    shapes_group = group[name]
    _write_metadata(
        shapes_group,
        group_type=group_type,
        # coordinate_transformations=coordinate_transformations,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert transform is not None
    _overwrite_coordinate_transformations_non_raster(group=shapes_group, axes=axes, transformation=transform)


def write_points(
    points: pa.Table,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = PointsFormat(),
) -> None:
    axes = get_dims(points)
    t = _get_transform(points)

    points_groups = group.require_group(name)
    path = os.path.join(points_groups._store.path, points_groups.path, "points.parquet")
    pq.write_table(points, path)

    attrs = {}
    attrs["version"] = fmt.spatialdata_version
    (0, get_dims(points))

    _write_metadata(
        points_groups,
        group_type=group_type,
        # coordinate_transformations=coordinate_transformations,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    _overwrite_coordinate_transformations_non_raster(group=points_groups, axes=axes, transformation=t)


def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    fmt: Format = SpatialDataFormatV01(),
) -> None:
    region = table.uns["spatialdata_attrs"]["region"]
    region_key = table.uns["spatialdata_attrs"].get("region_key", None)
    instance_key = table.uns["spatialdata_attrs"].get("instance_key", None)
    fmt.validate_table(table, region_key, instance_key)
    write_adata(group, name, table)  # creates group[name]
    tables_group = group[name]
    tables_group.attrs["@type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key
    tables_group.attrs["version"] = fmt.spatialdata_version


def _iter_multiscale(
    data: MultiscaleSpatialImage,
    attr: str,
    key: Optional[str] = None,
) -> list[Any]:
    # TODO: put this check also in the validator for raster multiscales
    name = None
    for i in data.keys():
        variables = list(data[i].variables)
        if len(variables) != 1:
            raise ValueError("MultiscaleSpatialImage must have exactly one variable (the variable name is arbitrary)")
        if name is not None:
            if name != variables[0]:
                raise ValueError("MultiscaleSpatialImage must have the same variable name across all levels")
        name = variables[0]
    if key is None:
        return [getattr(data[i][name], attr) for i in data.keys()]
    else:
        return [getattr(data[i][name], attr).get(key) for i in data.keys()]
