from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal, cast

import zarr
import zarr.storage
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from ome_zarr.format import Format
from pyarrow import ArrowInvalid
from upath import UPath
from zarr.errors import ArrayNotFoundError

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import (
    BadFileHandleMethod,
    _resolve_zarr_store,
    handle_read_errors,
)
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger
from spatialdata._types import Raster_T


def _read_zarr_group_spatialdata_element(
    root_group: zarr.Group,
    root_store_path: str,
    sdata_version: Literal["0.1", "0.2"],
    selector: set[str],
    read_func: Callable[..., Any],
    group_name: Literal["images", "labels", "shapes", "points", "tables"],
    element_type: Literal["image", "labels", "shapes", "points", "tables"],
    element_container: (dict[str, Raster_T] | dict[str, DaskDataFrame] | dict[str, GeoDataFrame] | dict[str, AnnData]),
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN],
) -> None:
    with handle_read_errors(
        on_bad_files,
        location=group_name,
        exc_types=JSONDecodeError,
    ):
        if group_name in selector and group_name in root_group:
            group = root_group[group_name]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group = group[subgroup_name]
                elem_group_path = os.path.join(root_store_path, elem_group.path)
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        KeyError,
                        ArrayNotFoundError,
                        OSError,
                        ArrowInvalid,
                        JSONDecodeError,
                        ValueError,
                    ),
                ):
                    if element_type in ["image", "labels"]:
                        reader_format = get_raster_format_for_read(elem_group, sdata_version)
                        element = read_func(
                            elem_group_path,
                            cast(Literal["image", "labels"], element_type),
                            reader_format,
                        )
                    elif element_type in ["shapes", "points", "tables"]:
                        element = read_func(elem_group_path)
                    else:
                        raise ValueError(f"Unknown element type {element_type}")
                    element_container[subgroup_name] = element
                    count += 1
            logger.debug(f"Found {count} elements in {group}")


def get_raster_format_for_read(
    group: zarr.Group,
    sdata_version: Literal["0.1", "0.2"],
    return_ome_zarr_format: bool = False,
) -> Format:
    """Get raster format of stored raster data.

    This checks the image or label element zarr group metadata to retrieve the format that is used by
    ome-zarr's ZarrLocation for reading the data.

    Parameters
    ----------
    group
        The zarr group of the raster element to be read.
    sdata_version
        The version of the SpatialData zarr store retrieved from the spatialdata attributes.
    return_ome_zarr_format
        Whether to return the ome-zarr Format or a SpatialData raster format class (which is a subclass of Format)

    Returns
    -------
    The ome-zarr format to use for reading the raster element.
    """
    from spatialdata._io.format import (
        sdata_zarr_version_to_ome_zarr_format,
        sdata_zarr_version_to_raster_format,
    )

    if sdata_version == "0.1":
        group_version = group.metadata.attributes["multiscales"][0]["version"]
    elif sdata_version == "0.2":
        group_version = group.metadata.attributes["ome"]["version"]
    else:
        raise ValueError(f"Unknown SpatialData zarr version {sdata_version}")
    if return_ome_zarr_format:
        return sdata_zarr_version_to_ome_zarr_format[group_version]
    return sdata_zarr_version_to_raster_format[group_version]


def read_zarr(
    store: str | Path | UPath | zarr.Group,
    selection: None | tuple[str] = None,
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN] = BadFileHandleMethod.ERROR,
) -> SpatialData:
    """
    Read a SpatialData dataset from a zarr store (on-disk or remote).

    Parameters
    ----------
    store
        Path, URL, or zarr.Group to the zarr store (on-disk or remote).

    selection
        List of elements to read from the zarr store (images, labels, points, shapes, table). If None, all elements are
        read.

    on_bad_files
        Specifies what to do upon encountering a bad file, e.g. corrupted, invalid or missing files.
        Allowed values are :

        - 'error', raise an exception when a bad file is encountered. Reading aborts immediately
          with an error.
        - 'warn', raise a warning when a bad file is encountered and skip that file. A SpatialData
          object is returned containing only elements that could be read. Failures can only be
          determined from the warnings.

    Returns
    -------
    A SpatialData object.
    """
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store)
    root_group = zarr.open_group(resolved_store, mode="r")

    from spatialdata._io.io_collection import is_rfc8_collection_store, read_rfc8_zarr

    if is_rfc8_collection_store(root_group):
        return read_rfc8_zarr(root_group, root_group.store.root)

    # the following is the SpatialDataContainerFormat version
    if "spatialdata_attrs" not in root_group.metadata.attributes:
        # backward compatibility for pre-versioned SpatialData zarr stores
        sdata_version: Literal["0.1", "0.2", "0.3"] = "0.1"
    else:
        sdata_version = root_group.metadata.attributes["spatialdata_attrs"]["version"]
    if sdata_version == "0.1":
        warnings.warn(
            "SpatialData is not stored in the most current format. If you want to use Zarr v3"
            ", please write the store to a new location using `sdata.write()`.",
            UserWarning,
            stacklevel=2,
        )
    root_store_path = root_group.store.root

    images: dict[str, Raster_T] = {}
    labels: dict[str, Raster_T] = {}
    points: dict[str, DaskDataFrame] = {}
    shapes: dict[str, GeoDataFrame] = {}
    tables: dict[str, AnnData] = {}

    if sdata_version == "0.3":
        # New flat layout: scan for elements by their spatialdata_attrs.element_type marker
        import os as _os

        scanned = _scan_for_elements(root_group, root_store_path)
        for elem_name, (element_type, abs_path) in scanned.items():
            leaf_name = elem_name.split("/")[-1]
            if element_type == "images":
                reader_format = get_raster_format_for_read(root_group[elem_name], sdata_version="0.2")
                images[elem_name] = _read_multiscale(abs_path, "image", reader_format)
            elif element_type == "labels":
                # In new flat layout labels are stored under element_group/labels/leaf_name
                label_abs_path = _os.path.join(abs_path, "labels", leaf_name)
                reader_format = get_raster_format_for_read(
                    root_group[elem_name]["labels"][leaf_name], sdata_version="0.2"
                )
                labels[elem_name] = _read_multiscale(label_abs_path, "labels", reader_format)
            elif element_type == "points":
                points[elem_name] = _read_points(abs_path)
            elif element_type == "shapes":
                shapes[elem_name] = _read_shapes(abs_path)
            elif element_type == "tables":
                tables[elem_name] = _read_table(abs_path)
            else:
                logger.warning(f"Unknown element_type {element_type!r} for element {elem_name!r}, skipping.")
    else:
        # Old typed layout (v0.1 / v0.2): elements are at root/element_type/element_name
        selector = {"images", "labels", "points", "shapes", "tables"} if not selection else set(selection or [])
        logger.debug(f"Reading selection {selector}")

        # we could make this more readable. One can get lost when looking at this dict and iteration over the items
        group_readers: dict[
            Literal["images", "labels", "shapes", "points", "tables"],
            tuple[
                Callable[..., Any],
                Literal["image", "labels", "shapes", "points", "tables"],
                dict[str, Raster_T] | dict[str, DaskDataFrame] | dict[str, GeoDataFrame] | dict[str, AnnData],
            ],
        ] = {
            # ome-zarr-py needs a kwargs that has "image" has key. So here we have "image" and not "images"
            "images": (_read_multiscale, "image", images),
            "labels": (_read_multiscale, "labels", labels),
            "points": (_read_points, "points", points),
            "shapes": (_read_shapes, "shapes", shapes),
            "tables": (_read_table, "tables", tables),
        }
        for group_name, (
            read_func,
            element_type,
            element_container,
        ) in group_readers.items():
            _read_zarr_group_spatialdata_element(
                root_group=root_group,
                root_store_path=root_store_path,
                sdata_version=sdata_version,
                selector=selector,
                read_func=read_func,
                group_name=group_name,
                element_type=element_type,
                element_container=element_container,
                on_bad_files=on_bad_files,
            )

    # read attrs metadata
    attrs = root_group.attrs.asdict()
    if "spatialdata_attrs" in attrs:
        # when refactoring the read_zarr function into reading componenets separately (and according to the version),
        # we can move the code below (.pop()) into attrs_from_dict()
        attrs.pop("spatialdata_attrs")
    else:
        attrs = None

    all_elements: dict[str, Any] = {}
    all_elements.update(images)
    all_elements.update(labels)
    all_elements.update(points)
    all_elements.update(shapes)
    all_elements.update(tables)

    sdata = SpatialData(elements=all_elements, attrs=attrs)
    sdata.path = resolved_store.root
    return sdata


def _get_groups_for_element(
    zarr_path: Path, element_type: str, element_name: str, use_consolidated: bool = True
) -> tuple[zarr.Group, zarr.Group, zarr.Group]:
    """
    Get the Zarr groups for the root, element_type and element for a specific element.

    The store must exist, but creates the element type group and the element group if they don't exist. In all cases
    the zarr group will also be opened. When writing data to disk this should always be done with 'use_consolidated'
    being 'False'. If a user wrote the data previously with consolidation of the metadata and then they write new data
    in the zarr store, it can give errors otherwise, due to partially written elements not yet being present in the
    consolidated metadata store, e.g. when first writing the element and then opening the zarr group again for writing
    transformations.

    Parameters
    ----------
    zarr_path
        The path to the Zarr storage.
    element_type
        type of the element; must be in ["images", "labels", "points", "polygons", "shapes", "tables"].
    element_name
        name of the element
    use_consolidated
        whether to open zarr groups using consolidated metadata. This should be false when writing as we open
        zarr groups multiple times when writing an element. If the consolidated metadata store is out of sync with
        what is written on disk this leads to errors.

    Returns
    -------
    The Zarr groups for the root, element_type and element for a specific element.
    """
    if not isinstance(zarr_path, Path):
        raise ValueError("zarr_path should be a Path object")

    if element_type not in [
        "images",
        "labels",
        "points",
        "polygons",
        "shapes",
        "tables",
    ]:
        raise ValueError(f"Unknown element type {element_type}")
    resolved_store = _resolve_zarr_store(zarr_path)

    # When writing, use_consolidated must be set to False. Otherwise, the metadata store
    # can get out of sync with newly added elements (e.g., labels), leading to errors.
    root_group = zarr.open_group(store=resolved_store, mode="r+", use_consolidated=use_consolidated)
    element_type_group = root_group.require_group(element_type)
    element_type_group = zarr.open_group(element_type_group.store_path, mode="a", use_consolidated=use_consolidated)

    element_name_group = element_type_group.require_group(element_name)
    return root_group, element_type_group, element_name_group


def _group_for_element_exists(zarr_path: Path, element_type: str, element_name: str) -> bool:
    """
    Check if the group for an element exists (old typed-layout: root/element_type/element_name).

    Parameters
    ----------
    element_type
        type of the element; must be in ["images", "labels", "points", "polygons", "shapes", "tables"].
    element_name
        name of the element

    Returns
    -------
    True if the group exists, False otherwise.
    """
    store = _resolve_zarr_store(zarr_path)
    root = zarr.open_group(store=store, mode="r")
    assert element_type in [
        "images",
        "labels",
        "points",
        "polygons",
        "shapes",
        "tables",
    ]
    exists = element_type in root and element_name in root[element_type]
    store.close()
    return exists


def _group_for_flat_element_exists(zarr_path: Path, element_name: str) -> bool:
    """
    Check if the group for an element exists in the new flat layout (root/element_name).

    Parameters
    ----------
    zarr_path
        The path to the Zarr storage.
    element_name
        name of the element (may contain "/" for hierarchical elements)

    Returns
    -------
    True if the group exists, False otherwise.
    """
    store = _resolve_zarr_store(zarr_path)
    root = zarr.open_group(store=store, mode="r")
    try:
        exists = element_name in root or _path_in_zarr_group(root, element_name)
    except Exception:
        exists = False
    store.close()
    return exists


def _path_in_zarr_group(group: zarr.Group, path: str) -> bool:
    """Check if a path (possibly with '/') exists inside a zarr group."""
    parts = path.split("/", 1)
    if parts[0] not in group:
        return False
    if len(parts) == 1:
        return True
    return _path_in_zarr_group(group[parts[0]], parts[1])


def _get_flat_element_group(
    zarr_path: Path, element_name: str, use_consolidated: bool = True
) -> zarr.Group:
    """
    Get the Zarr group for an element in the new flat layout (root/element_name).

    Parameters
    ----------
    zarr_path
        The path to the Zarr storage.
    element_name
        Name of the element (may contain "/" for hierarchical paths).
    use_consolidated
        Whether to open zarr groups using consolidated metadata.

    Returns
    -------
    The element group.
    """
    if not isinstance(zarr_path, Path):
        raise ValueError("zarr_path should be a Path object")

    resolved_store = _resolve_zarr_store(zarr_path)
    root_group = zarr.open_group(store=resolved_store, mode="r+", use_consolidated=use_consolidated)
    element_group = root_group[element_name]
    return element_group


def _get_or_create_flat_element_group(
    zarr_path: Path, element_type: str, element_name: str
) -> tuple[zarr.Group, zarr.Group]:
    """
    Get or create a Zarr group for an element using the new flat layout (root/element_name, no type subdir).

    element_name may contain "/" for hierarchical elements (e.g. "sample1/image").
    Returns (root_group, element_group). Stamps element_type into element_group attrs.

    Parameters
    ----------
    zarr_path
        The path to the Zarr storage.
    element_type
        The type of the element (e.g. "images", "labels", "points", "shapes", "tables").
    element_name
        The name of the element (may contain "/" for hierarchical paths).

    Returns
    -------
    A tuple of (root_group, element_group).
    """
    if not isinstance(zarr_path, Path):
        raise ValueError("zarr_path should be a Path object")

    resolved_store = _resolve_zarr_store(zarr_path)
    root_group = zarr.open_group(store=resolved_store, mode="r+", use_consolidated=False)
    # zarr.require_group handles "/" in element_name transparently as nested groups
    element_group = root_group.require_group(element_name)
    # Stamp the element type so the reader can identify it
    attrs = dict(element_group.attrs)
    if "spatialdata_attrs" not in attrs:
        attrs["spatialdata_attrs"] = {}
    attrs["spatialdata_attrs"]["element_type"] = element_type
    element_group.attrs.update(attrs)
    return root_group, element_group


def _scan_for_elements(
    group: zarr.Group,
    root_path: str,
    prefix: str = "",
) -> dict[str, tuple[str, str]]:
    """
    Recursively scan a zarr group for SpatialData elements (v0.3 flat layout).

    An element is identified by having 'element_type' inside 'spatialdata_attrs' in its .zattrs.
    Stops recursing into a group once it is identified as an element.

    Parameters
    ----------
    group
        The zarr group to scan.
    root_path
        The absolute path to the root zarr store.
    prefix
        Path prefix accumulated during recursion.

    Returns
    -------
    dict mapping element_name -> (element_type, abs_path)
    """
    import os as _os

    found: dict[str, tuple[str, str]] = {}
    for name in group:
        if name.startswith("."):
            continue
        try:
            sub = group[name]
        except Exception:
            continue
        if not hasattr(sub, "attrs"):
            continue
        elem_name = f"{prefix}{name}" if prefix else name
        try:
            attrs = sub.attrs.asdict()
        except Exception:
            attrs = {}
        sdata_attrs = attrs.get("spatialdata_attrs", {})
        element_type = sdata_attrs.get("element_type")

        if element_type is not None:
            # This is a SpatialData element — don't recurse further
            abs_path = _os.path.join(root_path, sub.path) if root_path else sub.path
            found[elem_name] = (element_type, abs_path)
        elif hasattr(sub, "keys"):
            # Structural group — recurse
            found.update(_scan_for_elements(sub, root_path, prefix=f"{elem_name}/"))
    return found


def _write_consolidated_metadata(path: Path | str | None) -> None:
    if path is not None:
        f = zarr.open_group(path, mode="r+", use_consolidated=False)
        # .parquet files are not recognized as proper zarr and thus throw a warning. This does not affect SpatialData.
        # and therefore we silence it for our users as they can't do anything about this.
        # TODO check with remote PR whether we can prevent this warning at least for points data and whether with zarrv3
        # that pr would still work.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)
            zarr.consolidate_metadata(f.store)
        f.store.close()
