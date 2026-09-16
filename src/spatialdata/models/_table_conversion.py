"""Conversion of tables between table format 0.2 (``region``/``region_key``/``instance_key``) and 0.3 (links)."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

import pandas as pd
from anndata import AnnData

from spatialdata._core.links.edges import INDEX, VALUE, Edge
from spatialdata.models.models import ATTRS_KEY, TableModel, get_table_keys

__all__ = ["convert_table_v02_to_v03", "convert_table_v03_to_v02"]

TABLE_FORMAT_V03 = "0.3"


def _nullable_dtype(ids: pd.Series) -> str:
    dtype = ids.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return "Int64"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "string"
    raise TypeError(f"Unsupported dtype {ids.dtype} for the instance ids; expected integers or strings.")


def convert_table_v02_to_v03(
    table: AnnData,
    table_name: str,
    *,
    labels: Collection[str] = (),
    copy: bool = True,
) -> tuple[AnnData, list[Edge]]:
    """
    Convert a table from table format 0.2 to 0.3.

    In format 0.2 a table describes what it annotates with the ``region``/``region_key``/``instance_key`` metadata: the
    ``region_key`` column says which element each row annotates and the ``instance_key`` column which instance of it.
    In format 0.3 the table has **one column per annotated element**, named after the element, holding the instance
    ids (``NA`` for the rows that do not annotate that element); which column links to which element is stated by the
    edges of the SpatialData object. The two legacy columns are removed and the metadata is reduced to the version.

    Parameters
    ----------
    table
        The table to convert; it must carry the ``region``/``region_key``/``instance_key`` metadata.
    table_name
        The name of the table in the SpatialData object, used as the source of the returned edges.
    labels
        Names of the annotated elements that are labels (segmentation masks): their edges join on the pixel values
        (``"value"``); all the other elements join on their index.
    copy
        If `True` (default) the input table is left untouched and a copy is converted.

    Returns
    -------
    The converted table and the edges to add to ``sdata.links``, one per annotated element (declared ``1:1``).

    Raises
    ------
    ValueError
        If the table has no ``region``/``region_key``/``instance_key`` metadata, or if ``obs`` already has a column
        named like one of the annotated elements.
    """
    if not TableModel.has_region_annotation(table):
        raise ValueError(
            "The table has no `region`/`region_key`/`instance_key` metadata: it is either already in table format 0.3 "
            "or it does not annotate any element."
        )
    TableModel.validate(table)
    region, region_key, instance_key = get_table_keys(table)
    regions = [region] if isinstance(region, str) else list(region)

    table = table.copy() if copy else table
    obs = table.obs
    clashes = [r for r in regions if r in obs.columns and r not in (region_key, instance_key)]
    if clashes:
        raise ValueError(
            f"Cannot create the link column(s) {clashes}: `obs` already has column(s) with these names. Rename them "
            "first."
        )

    ids = obs[instance_key]
    target_dtype = _nullable_dtype(ids)
    region_column = obs[region_key].astype(str)
    link_columns = {}
    for r in regions:
        values = ids.where(region_column == r).to_numpy(dtype=object)
        link_columns[r] = pd.Series(pd.array(values, dtype=target_dtype), index=obs.index)

    obs = obs.drop(columns=[region_key, instance_key])
    for r in regions:
        obs[r] = link_columns[r]
    table.obs = obs
    table.uns[ATTRS_KEY] = {TableModel.VERSION_KEY: TABLE_FORMAT_V03}

    edges = [
        Edge(
            from_=table_name,
            to=r,
            method="join",
            on={"left": r, "right": VALUE if r in labels else INDEX},
            status="computed",
            cardinality="1:1",
        )
        for r in regions
    ]
    return table, edges


def convert_table_v03_to_v02(
    table: AnnData,
    edges: Iterable[Edge | Mapping[str, Any]],
    table_name: str,
    *,
    region_key: str = "region",
    instance_key: str = "instance_id",
    copy: bool = True,
) -> AnnData:
    """
    Convert a table from table format 0.3 back to 0.2.

    The per-element columns of instance ids are collapsed into a ``region_key`` column (the element annotated by each
    row) and an ``instance_key`` column (the instance id), and the ``region``/``region_key``/``instance_key`` metadata
    is restored. Format 0.2 can only represent rows that annotate exactly one element.

    Parameters
    ----------
    table
        The table to convert; it must not carry ``region``/``region_key``/``instance_key`` metadata.
    edges
        The edges of the SpatialData object (or their dictionaries); only the key joins from ``table_name`` to a single
        element on its index or pixel values are used.
    table_name
        The name of the table in the SpatialData object.
    region_key
        Name of the column that will hold the annotated element of each row.
    instance_key
        Name of the column that will hold the instance id of each row.
    copy
        If `True` (default) the input table is left untouched and a copy is converted.

    Returns
    -------
    The converted table.

    Raises
    ------
    ValueError
        If the table carries ``region``/``region_key``/``instance_key`` metadata, if no edge links the table to an
        element, if a link column is missing, if ``obs`` already has the ``region_key`` or ``instance_key`` column, or
        if a row annotates zero or several elements.
    """
    if TableModel.has_region_annotation(table):
        raise ValueError("The table already carries `region`/`region_key`/`instance_key` metadata (table format 0.2).")
    parsed = [e if isinstance(e, Edge) else Edge.from_dict(e) for e in edges]
    links = {
        e.to: e.on["left"]
        for e in parsed
        if e.from_ == table_name and e.method == "join" and isinstance(e.to, str) and e.on["right"] in (INDEX, VALUE)
    }
    if not links:
        raise ValueError(f"No edge links the table {table_name!r} to an element.")
    obs = table.obs
    missing = [column for column in links.values() if column not in obs.columns]
    if missing:
        raise ValueError(f"The link column(s) {missing} are not in `obs`.")
    for column in (region_key, instance_key):
        if column in obs.columns and column not in links.values():
            raise ValueError(f"`obs` already has a column named {column!r}; choose another name.")

    not_na = pd.DataFrame({element: obs[column].notna() for element, column in links.items()}, index=obs.index)
    n_links = not_na.sum(axis=1)
    if (n_links > 1).any():
        rows = obs.index[n_links > 1].tolist()
        raise ValueError(
            f"Table format 0.2 cannot represent rows annotating several elements; offending rows: {rows[:10]}"
            f"{' ...' if len(rows) > 10 else ''}."
        )
    if (n_links == 0).any():
        rows = obs.index[n_links == 0].tolist()
        raise ValueError(
            f"Table format 0.2 cannot represent rows that annotate no element; offending rows: {rows[:10]}"
            f"{' ...' if len(rows) > 10 else ''}."
        )

    table = table.copy() if copy else table
    obs = table.obs
    region_column = not_na.idxmax(axis=1)
    instance_values = pd.Series(pd.NA, index=obs.index, dtype=object)
    for element, column in links.items():
        mask = not_na[element]
        instance_values[mask] = obs.loc[mask, column].to_numpy(dtype=object)
    link_dtypes = {str(obs[column].dtype) for column in links.values()}
    instance_dtype = "int64" if link_dtypes <= {"Int64"} else "str"

    obs = obs.drop(columns=list(links.values()))
    obs[region_key] = pd.Categorical(region_column)
    obs[instance_key] = instance_values.astype(instance_dtype)
    table.obs = obs

    regions = sorted(region_column.unique().tolist())
    table.uns[ATTRS_KEY] = {
        TableModel.REGION_KEY: regions[0] if len(regions) == 1 else regions,
        TableModel.REGION_KEY_KEY: region_key,
        TableModel.INSTANCE_KEY: instance_key,
    }
    return TableModel.parse(table)
