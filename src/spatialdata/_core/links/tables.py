"""Links between a table and the elements it annotates.

From table format 0.3 on, a table has one column of instance ids per annotated element and the mapping *element →
column* is stated by the edges of the SpatialData object. Tables in the previous format carry the
``region``/``region_key``/``instance_key`` metadata instead. The helpers here give the join code a single view over
both: for every annotated element, the instance ids of the rows that annotate it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from spatialdata._core.links.edges import INDEX, VALUE, Edge
from spatialdata.models import Labels2DModel, Labels3DModel, TableModel, get_model, get_table_keys
from spatialdata.models._table_conversion import convert_table_v02_to_v03

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData

__all__ = [
    "TableLink",
    "annotated_elements",
    "convert_legacy_table",
    "get_table_links",
    "resolve_instance_ids",
    "right_key_for_element",
    "table_instance_keys",
    "table_instance_keys_any_format",
]


@dataclass(frozen=True)
class TableLink:
    """One link from a table to an element: the ``obs`` column holding the instance ids and the key on the element."""

    element: str
    column: str
    right: Literal["index", "value"]

    def to_edge(self, table_name: str, cardinality: str | None = "1:1") -> Edge:
        return Edge(
            from_=table_name,
            to=self.element,
            method="join",
            on={"left": self.column, "right": self.right},
            status="computed",
            cardinality=cardinality,
        )


def get_table_links(sdata: SpatialData, table_name: str) -> dict[str, TableLink]:
    """Return the links from a table to elements, keyed by element name, as stated by ``sdata.links``."""
    links: dict[str, TableLink] = {}
    for edge in sdata.links.for_element(table_name, direction="from"):
        if edge.method != "join" or not isinstance(edge.to, str) or edge.on["right"] not in (INDEX, VALUE):
            continue
        links[edge.to] = TableLink(element=edge.to, column=edge.on["left"], right=edge.on["right"])  # type: ignore[arg-type]
    return links


def table_instance_keys(sdata: SpatialData, table_name: str) -> dict[str, str]:
    """Return the mapping *element name → column of ``table.obs`` holding its instance ids*, from ``sdata.links``.

    Pass it to the functions that receive a table detached from its SpatialData object (they cannot look up the links
    themselves); it is empty, and therefore ignored, for tables carrying the legacy annotation metadata.
    """
    return {element: link.column for element, link in get_table_links(sdata, table_name).items()}


def table_instance_keys_any_format(
    table: AnnData, *, sdata: SpatialData | None = None, table_name: str | None = None
) -> dict[str, str]:
    """Return the mapping *element name → column of ``table.obs`` holding its instance ids*, whatever the format.

    For a table carrying the legacy annotation metadata every annotated element shares the same ``instance_key``
    column; for a table in table format 0.3 the columns are read from the links of the SpatialData object.
    """
    if TableModel.has_region_annotation(table):
        region, _, instance_key = get_table_keys(table)
        return dict.fromkeys([region] if isinstance(region, str) else region, instance_key)
    if sdata is not None and table_name is not None:
        return table_instance_keys(sdata, table_name)
    return {}


def right_key_for_element(sdata: SpatialData, element_name: str) -> Literal["index", "value"]:
    """Return the key a table column joins on for an element: pixel values for labels, the index otherwise."""
    element = sdata.get(element_name)
    if element is None:
        return INDEX  # type: ignore[return-value]
    return VALUE if get_model(element) in (Labels2DModel, Labels3DModel) else INDEX  # type: ignore[return-value]


def _positional_ids(values: pd.Series, mask: np.ndarray) -> pd.Series:
    """Return the instance ids of the selected rows, indexed by row position and with plain numpy dtypes."""
    positions = np.flatnonzero(mask)
    selected = values.iloc[positions]
    dtype = selected.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        selected = selected.astype(dtype.categories.dtype)
    elif pd.api.types.is_extension_array_dtype(dtype):
        selected = selected.astype("int64") if pd.api.types.is_integer_dtype(dtype) else selected.astype(object)
    return pd.Series(selected.to_numpy(), index=positions)


def _instance_ids_from_columns(table: AnnData, columns: Mapping[str, str]) -> dict[str, pd.Series]:
    ids: dict[str, pd.Series] = {}
    for element, column in columns.items():
        if column not in table.obs.columns:
            raise ValueError(f"The link column {column!r} for the element {element!r} is not in `table.obs`.")
        values = table.obs[column]
        ids[element] = _positional_ids(values, values.notna().to_numpy())
    return ids


def _instance_ids_from_region_annotation(table: AnnData) -> dict[str, pd.Series]:
    region, region_key, instance_key = get_table_keys(table)
    regions = [region] if isinstance(region, str) else list(region)
    region_column = table.obs[region_key].astype(str).to_numpy()
    # the declared regions, plus the ones found in the region key column: the two can disagree, since the column can
    # be edited without updating the metadata (see `SpatialData.update_annotated_regions_metadata`)
    regions += [r for r in pd.unique(region_column).tolist() if r not in regions]
    values = table.obs[instance_key]
    return {r: _positional_ids(values, region_column == r) for r in regions}


def resolve_instance_ids(
    table: AnnData,
    *,
    sdata: SpatialData | None = None,
    table_name: str | None = None,
    instance_keys: Mapping[str, str] | None = None,
) -> dict[str, pd.Series]:
    """
    Return, for every element annotated by the table, the instance ids of the rows annotating it.

    Each series is indexed by the row position in ``table.obs`` and contains no missing values.

    Parameters
    ----------
    table
        The table.
    sdata
        The SpatialData object whose links describe the table; needed for tables in format 0.3 unless
        ``instance_keys`` is given.
    table_name
        The name of the table in ``sdata``.
    instance_keys
        Explicit mapping *element name → column of ``table.obs``* holding the instance ids; overrides ``sdata``.

    Raises
    ------
    ValueError
        If the table is in format 0.3 and neither ``instance_keys`` nor ``sdata`` and ``table_name`` are given.
    """
    if TableModel.has_region_annotation(table):
        return _instance_ids_from_region_annotation(table)
    if instance_keys is not None:
        return _instance_ids_from_columns(table, instance_keys)
    if sdata is not None and table_name is not None:
        links = get_table_links(sdata, table_name)
        return _instance_ids_from_columns(table, {element: link.column for element, link in links.items()})
    raise ValueError(
        "The table is in table format 0.3: the columns linking it to the elements are described by the links of the "
        "SpatialData object. Pass `sdata` and `table_name`, or `instance_keys` (a mapping from element name to the "
        "column of `table.obs` holding its instance ids)."
    )


def annotated_elements(
    table: AnnData,
    *,
    sdata: SpatialData | None = None,
    table_name: str | None = None,
    non_empty: bool = False,
) -> list[str]:
    """
    Return the names of the elements annotated by the table (an empty list if they cannot be determined).

    Parameters
    ----------
    table
        The table.
    sdata
        The SpatialData object whose links describe the table; needed for tables in table format 0.3.
    table_name
        The name of the table in ``sdata``.
    non_empty
        If `True`, only the elements that the table actually has rows for are returned; by default every declared
        element is returned, even if no row annotates it.
    """
    if non_empty:
        ids = resolve_instance_ids(table, sdata=sdata, table_name=table_name)
        return [element for element, element_ids in ids.items() if len(element_ids) > 0]
    if TableModel.has_region_annotation(table):
        region, _, _ = get_table_keys(table)
        return [region] if isinstance(region, str) else list(region)
    if sdata is not None and table_name is not None:
        return list(get_table_links(sdata, table_name))
    return []


def convert_legacy_table(sdata: SpatialData, table_name: str, table: AnnData) -> AnnData:
    """
    Convert a table with ``region``/``region_key``/``instance_key`` metadata to format 0.3 and register its edges.

    The table is converted in place; the edges previously going out of ``table_name`` are replaced.
    """
    region, _, _ = get_table_keys(table)
    regions = [region] if isinstance(region, str) else list(region)
    labels = [r for r in regions if right_key_for_element(sdata, r) == VALUE]
    converted, edges = convert_table_v02_to_v03(table, table_name, labels=labels, copy=False)
    for edge in sdata.links.for_element(table_name, direction="from"):
        sdata.links.remove(edge)
    for edge in edges:
        sdata.links.add(
            edge.from_,
            edge.to,
            on=(edge.on["left"], edge.on["right"]),
            cardinality=edge.cardinality,
            validate=False,
        )
    return converted
