"""Reader for NGFF RFC-8 collection-node stores with ``sp-ops:*`` attributes.

RFC-8 (https://ngff.openmicroscopy.org/rfc/8/) describes a hierarchical Zarr layout built from
``collection`` groups, distinct from the flat/typed layouts the rest of ``spatialdata`` reads
and writes. This module detects that layout at the store root and walks it directly, rather
than going through the ``spatialdata_attrs.version`` dispatch used elsewhere.

Scope (vertical slice): read-only, targeting the two example ``sp-ops`` stores exercised by
the project's own test suite. Not general RFC-8 support -- see the caveats in
``_read_rfc8_multiscale`` and ``_walk_collection``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dask.array as da
import zarr
from anndata import AnnData
from anndata import read_zarr as _read_anndata_zarr
from geopandas import GeoDataFrame
from geopandas import read_parquet as _read_geoparquet
from xarray import DataArray, DataTree

from spatialdata._core.relationships import Edge, Relationships
from spatialdata._core.spatialdata import SpatialData
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._types import Raster_T
from spatialdata.models import ShapesModel, TableModel
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations._utils import _set_transformations, compute_coordinates
from spatialdata.transformations.transformations import Identity

logger = logging.getLogger("spatialdata")

_D6_AXIS_ORDER = ("round", "t", "c", "z", "y", "x")


def is_rfc8_collection_store(root_group: zarr.Group) -> bool:
    """Return True if the root group's attributes declare an RFC-8 ``collection`` node."""
    ome = root_group.metadata.attributes.get("ome")
    return isinstance(ome, dict) and ome.get("type") == "collection"


def read_rfc8_zarr(root_group: zarr.Group, root_store_path: Path) -> SpatialData:
    """Read an RFC-8 collection-node Zarr store into a hierarchical SpatialData object."""
    elements: dict[str, Any] = {}
    relationships = Relationships()
    _walk_collection(root_store_path, prefix_stack=[], elements=elements, relationships=relationships)

    sdata = SpatialData(elements=elements, relationships=relationships)
    sdata.path = root_store_path
    return sdata


def _resolve_relative(prefix_stack: list[str], rel: str) -> str:
    """Resolve a ``sp-ops:relationships`` edge endpoint relative to the collection it is declared on.

    Each ``..`` pops one *hierarchy level* off ``prefix_stack``, not one ``/``-delimited string
    segment -- this matters because a level's own name can itself contain ``/`` (e.g. the well
    name ``A/1``, see D3/RFC-8's own naming tension, noted in the sp-ops spec's example builds).
    """
    stack = list(prefix_stack)
    parts = rel.split("/")
    for i, part in enumerate(parts):
        if part == "..":
            if stack:
                stack.pop()
        elif part == ".":
            continue
        else:
            return "/".join([*stack, *parts[i:]])
    return "/".join(stack)


def _walk_collection(
    path: Path,
    prefix_stack: list[str],
    elements: dict[str, Any],
    relationships: Relationships,
) -> None:
    group = zarr.open_group(str(path), mode="r")
    ome = group.attrs["ome"]
    assert ome.get("type") == "collection", f"Expected an RFC-8 collection node at {path}, got {ome.get('type')!r}."

    edges = ome.get("attributes", {}).get("sp-ops:relationships", {}).get("edges", [])
    for edge_dict in edges:
        resolved = dict(edge_dict)
        resolved["from"] = _resolve_relative(prefix_stack, edge_dict["from"])
        resolved["to"] = _resolve_relative(prefix_stack, edge_dict["to"])
        try:
            relationships.add(Edge.from_dict(resolved))
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed sp-ops:relationships edge {edge_dict!r} at {path}: {e}")

    for node in ome.get("nodes", []):
        name = node["name"]
        node_type = node["type"]
        elem_name = "/".join([*prefix_stack, name])
        node_path_info = node.get("path")
        child_path = path / node_path_info["path"] if node_path_info is not None else path / name

        try:
            if node_type == "collection":
                _walk_collection(child_path, [*prefix_stack, name], elements, relationships)
            elif node_type == "multiscale":
                elements[elem_name] = _read_rfc8_multiscale(child_path)
            elif node_type == "sp-ops:table":
                elements[elem_name] = _read_sp_ops_table(child_path)
            elif node_type == "sp-ops:shapes":
                elements[elem_name] = _read_sp_ops_shapes(child_path)
            elif node_type == "sp-ops:points":
                elements[elem_name] = _read_points(child_path)
            else:
                logger.warning(f"Skipping node {elem_name!r}: unknown RFC-8 node type {node_type!r}.")
        except Exception as e:  # noqa: BLE001 - one bad leaf must not sink the whole store
            logger.warning(f"Skipping node {elem_name!r} at {child_path}: could not be read ({e!r}).")


def _read_sp_ops_table(path: Path) -> AnnData:
    """Read a ``sp-ops:table`` node.

    These leaves are written as plain AnnData-in-zarr, but (unlike spatialdata's own table
    writer) without a ``spatialdata_attrs.version`` stamp, so ``_read_table`` can't dispatch on
    format version. Fall back to reading the AnnData directly.
    """
    try:
        return _read_table(path)
    except AssertionError:
        adata = _read_anndata_zarr(str(path))
        # anndata>=0.13 can report a phantom `None`-keyed layer on a table with no explicit
        # layers, which trips spatialdata's key-uniqueness validation; drop it defensively.
        adata.layers.pop(None, None)
        return TableModel.parse(adata)


def _read_sp_ops_shapes(path: Path) -> GeoDataFrame:
    """Read a ``sp-ops:shapes`` node.

    Same gap as `_read_sp_ops_table`: no ``spatialdata_attrs.version``/``coordinateTransformations``
    on these leaves, so `_read_shapes` can't be used as-is. Fall back to reading the GeoParquet
    file directly.
    """
    try:
        return _read_shapes(path)
    except (AssertionError, KeyError):
        gdf = _read_geoparquet(str(path / "shapes.parquet"))
        return ShapesModel.parse(gdf)


def _channel_names(ome_attributes: dict[str, Any], n_channels: int) -> list[str]:
    sp_channels = ome_attributes.get("sp-ops:channels")
    if (
        isinstance(sp_channels, list)
        and len(sp_channels) == n_channels
        and all(isinstance(c, dict) and "name" in c for c in sp_channels)
    ):
        return [str(c["name"]) for c in sp_channels]
    return [f"channel{i}" for i in range(n_channels)]


def _read_rfc8_multiscale(path: Path) -> Raster_T:
    """Read one RFC-8 ``multiscale`` node into a spatialdata Image or Labels DataArray/DataTree.

    Caveats (vertical slice, not general RFC-8 support):

    - Axis names are lowercased and singleton axes are dropped (sp-ops spec D6), following
      the *declared* `multiscales[0]["axes"]`, not `sp-ops:axis`/`sp-ops:rounds` (multi-round
      processed images are read as a plain image without their `round` provenance metadata).
    - A raw, single-channel array with no `c` axis at all (as ``sp-ops`` writes for ``raw``
      channels, see layout.md D5) gets a synthetic size-1 `c` axis prepended.
    - Only the per-dataset `coordinateTransformations` used to align pyramid levels are read;
      the well-level ``scene`` registration transforms (RFC-5) are not applied here, so every
      element gets an `Identity` transform to spatialdata's default "global" coordinate system
      rather than being placed in the shared `well` frame.
    - A node whose declared shape doesn't fit any spatialdata raster model (e.g. an RGBA
      rendering with axes `(y, x, c)` marked as a `labels` node, as `biohub_example`'s QC
      overlays are) is still read, but as an Image with `c` moved to the front, not as Labels.
    """
    group = zarr.open_group(str(path), mode="r")
    ome = group.attrs["ome"]
    multiscale = ome["multiscales"][0]
    ome_attributes = ome.get("attributes", {})

    axes_full = [ax["name"].lower() for ax in multiscale["axes"]]
    datasets = multiscale["datasets"]
    arrays = [da.from_zarr(str(path / dataset["path"])) for dataset in datasets]

    keep = [i for i, ax in enumerate(axes_full) if arrays[0].shape[i] != 1 or ax in ("y", "x")]
    dims = tuple(axes_full[i] for i in keep)
    indexer = tuple(slice(None) if i in keep else 0 for i in range(len(axes_full)))
    arrays = [arr[indexer] for arr in arrays]

    is_labels = "labels" in ome_attributes and "c" not in dims

    if not is_labels:
        if "c" not in dims:
            dims = ("c", *dims)
            arrays = [arr[None, ...] for arr in arrays]
        # Canonicalize to the sp-ops D6 axis order (round, t, c, z, y, x). Usually a no-op --
        # needed for e.g. an RGBA rendering declared as (y, x, c).
        canonical = tuple(sorted(dims, key=_D6_AXIS_ORDER.index))
        if dims != canonical:
            order = sorted(range(len(dims)), key=lambda i: _D6_AXIS_ORDER.index(dims[i]))
            dims = canonical
            arrays = [arr.transpose(order) for arr in arrays]

    channels = _channel_names(ome_attributes, arrays[0].shape[dims.index("c")]) if "c" in dims else None

    if len(arrays) > 1:
        result: DataArray | DataTree = dask_arrays_to_datatree(arrays, dims=dims, channels=channels)
    else:
        result = DataArray(
            arrays[0], name="image", dims=dims, coords={"c": channels} if channels is not None else {}
        )
    _set_transformations(result, {DEFAULT_COORDINATE_SYSTEM: Identity()})
    return compute_coordinates(result)
