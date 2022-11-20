from __future__ import annotations

from functools import singledispatch
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from dask.array.core import Array as DaskArray
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import parse_url
from spatial_image import SpatialImage

from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Label2DModel,
    Label3DModel,
    PointModel,
    PolygonModel,
    ShapeModel,
    TableModel,
)
from spatialdata._io.write import (
    write_image,
    write_labels,
    write_points,
    write_polygons,
    write_shapes,
    write_table,
)

# schema for elements
Label2d_s = Label2DModel()
Label3D_s = Label3DModel()
Image2D_s = Image2DModel()
Image3D_s = Image3DModel()
Polygon_s = PolygonModel
Point_s = PointModel()
Shape_s = ShapeModel()
Table_s = TableModel()


class SpatialData:
    """Spatial data structure."""

    images: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})
    labels: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})
    points: Mapping[str, AnnData] = MappingProxyType({})
    polygons: Mapping[str, GeoDataFrame] = MappingProxyType({})
    shapes: Mapping[str, AnnData] = MappingProxyType({})
    _table: Optional[AnnData] = None

    def __init__(
        self,
        images: Mapping[str, Any] = MappingProxyType({}),
        labels: Mapping[str, Any] = MappingProxyType({}),
        points: Mapping[str, Any] = MappingProxyType({}),
        polygons: Mapping[str, Any] = MappingProxyType({}),
        shapes: Mapping[str, Any] = MappingProxyType({}),
        table: Optional[AnnData] = None,
        **kwargs: Any,
    ) -> None:

        if images is not None:
            self.images = {}
            for k, v in images.items():
                if ndim(v) == 3:
                    Image2D_s.validate(v)
                    self.images[k] = v
                elif ndim(v) == 4:
                    Image3D_s.validate(v)
                    self.images[k] = v

        if labels is not None:
            self.labels = {}
            for k, v in labels.items():
                if ndim(v) == 2:
                    Label2d_s.validate(v)
                    self.labels[k] = v
                elif ndim(v) == 3:
                    Label3D_s.validate(v)
                    self.labels[k] = v

        if polygons is not None:
            self.polygons = {}
            for k, v in polygons.items():
                Polygon_s.validate(v)
                self.polygons[k] = v

        if shapes is not None:
            self.shapes = {}
            for k, v in shapes.items():
                Shape_s.validate(v)
                self.shapes[k] = v

        if points is not None:
            self.points = {}
            for k, v in points.items():
                Point_s.validate(v)
                self.points[k] = v

        if table is not None:
            Table_s.validate(table)
            self._table = table

    def write(self, file_path: str) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        # get union of unique ids of all elements
        elems = set().union(
            *[
                set(i)
                for i in [
                    self.images,
                    self.labels,
                    self.points,
                    self.polygons,
                    self.shapes,
                ]
            ]
        )

        for el in elems:
            elem_group = root.create_group(name=el)
            if self.images is not None and el in self.images.keys():
                write_image(
                    image=self.images[el],
                    group=elem_group,
                    name=el,
                    storage_options={"compressor": None},
                )
            if self.labels is not None and el in self.labels.keys():
                write_labels(
                    labels=self.labels[el],
                    group=elem_group,
                    name=el,
                    storage_options={"compressor": None},
                )
            if self.polygons is not None and el in self.polygons.keys():
                write_polygons(
                    polygons=self.polygons[el],
                    group=elem_group,
                    name=el,
                    storage_options={"compressor": None},
                )
            if self.shapes is not None and el in self.shapes.keys():
                write_shapes(
                    shapes=self.shapes[el],
                    group=elem_group,
                    name=el,
                    storage_options={"compressor": None},
                )
            if self.points is not None and el in self.points.keys():
                write_points(
                    points=self.points[el],
                    group=elem_group,
                    name=el,
                    storage_options={"compressor": None},
                )

        if self.table is not None:
            write_table(table=self.table, group=root, name="table")

    @property
    def table(self) -> AnnData:
        return self._table

    @classmethod
    def read(cls, file_path: str) -> SpatialData:
        from spatialdata._io.read import read_zarr

        sdata = read_zarr(file_path)
        return sdata

    def __repr__(self) -> str:
        return self._gen_repr()

    def _gen_repr(
        self,
    ) -> str:
        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            li = s.rsplit(old, occurrence)
            return new.join(li)

        def h(s: str) -> str:
            return s
            # return hashlib.md5(repr(s).encode()).hexdigest()

        ##
        descr = "SpatialData object with:"
        for attr in ["images", "labels", "points", "polygons", "shapes", "table"]:
            attribute = getattr(self, attr)
            if attribute is not None and len(attribute) > 0:
                descr += f"\n{h('level0')}{attr.capitalize()}"
                if isinstance(attribute, AnnData):
                    descr += f"{h('empty_line')}"
                    descr_class = attribute.__class__.__name__
                    descr += f"{h('level1.0')}'{attribute}': {descr_class} {attribute.shape}"
                    descr = rreplace(descr, h("level1.0"), "    └── ", 1)
                else:
                    # descr = rreplace(descr, h("level0"), "└── ", 1)
                    for k, v in attribute.items():
                        descr += f"{h('empty_line')}"
                        descr_class = v.__class__.__name__
                        if attr == "points":
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} with osbm.spatial {v.shape}"
                        elif attr == "polygons":
                            # assuming 2d
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} " f"shape: {v.shape}"
                        elif attr == "shapes":
                            # assuming 2d
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} " f"shape: {v.shape}"
                        else:
                            if isinstance(v, SpatialImage) or isinstance(v, MultiscaleSpatialImage):
                                descr += f"{h(attr + 'level1.1')}'{k}': {descr_class}"
                            else:
                                descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} {v.shape}"
                        # descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            if attr == "table":
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "polygons", "table", "shapes"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")
        ##
        return descr


@singledispatch
def ndim(arr: Any) -> int:
    raise TypeError(f"Unsupported type: {type(arr)}")


@ndim.register(np.ndarray)
def _(arr: DaskArray) -> int:
    return arr.ndim  # type: ignore[no-any-return]


@ndim.register(DaskArray)
def _(arr: DaskArray) -> int:
    return arr.ndim  # type: ignore[no-any-return]


@ndim.register(SpatialImage)
def _(arr: SpatialImage) -> int:
    return len(arr.dims)


@ndim.register(MultiscaleSpatialImage)
def _(arr: MultiscaleSpatialImage) -> int:
    return arr[list(arr.keys())[0]].dims  # type: ignore[no-any-return]
