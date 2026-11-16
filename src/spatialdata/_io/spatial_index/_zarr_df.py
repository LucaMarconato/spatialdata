"""Minimal reader/writer for the ``df:`` and ``enum:`` Zarr conventions.

- https://github.com/scverse/zarr-convention-dataframe
- https://github.com/scverse/zarr-convention-enum
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import zarr

DF_CONVENTION = {
    "uuid": "1f1b36d2-3185-6da8-91f6-9a0823af2d33",
    "schema_url": "https://raw.githubusercontent.com/scverse/zarr-convention-dataframe/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/scverse/zarr-convention-dataframe/blob/v1/README.md",
    "name": "df:",
    "description": "A collection of 1d-interpreted arrays to be used as a dataframe",
}
ENUM_CONVENTION = {
    "uuid": "c906c423-56ed-413c-9943-7b2ff52d18f2",
    "schema_url": "https://raw.githubusercontent.com/scverse/zarr-convention-categorical/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/scverse/zarr-convention-categorical/blob/v1/README.md",
    "name": "enum:",
    "description": "Dictionary-encoded enum array: integer codes indexing into a values array.",
}


def _has_convention(attrs: dict[str, Any], convention: dict[str, Any]) -> bool:
    return any(c.get("uuid") == convention["uuid"] for c in attrs.get("zarr_conventions", []))


def _numpy_dtype(t: pa.DataType) -> Any:
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return str
    return t.to_pandas_dtype()


class ZarrDataFrameWriter:
    """Create an empty ``df:`` group with a fixed schema and a fixed number of rows, then fill it by row ranges."""

    def __init__(
        self,
        group: zarr.Group,
        schema: pa.Schema,
        n_rows: int,
        chunk_length: int,
        shard_length: int | None,
        categories: dict[str, Sequence[str]],
    ) -> None:
        self.group = group
        self.schema = schema
        self.n_rows = n_rows
        names = list(schema.names)
        group.attrs.update({"zarr_conventions": [DF_CONVENTION], "df:shape": [n_rows, len(names)]})
        group.create_array("columns", data=np.asarray(names, dtype=str), chunks=(max(1, len(names)),))
        self.arrays: dict[str, zarr.Array] = {}
        chunks = (min(chunk_length, max(n_rows, 1)),)
        shards = None
        if shard_length is not None:
            shard = max(chunks[0], (shard_length // chunks[0]) * chunks[0])
            shards = (shard,)
        for i, field in enumerate(schema):
            name = f"column_{i}"
            if pa.types.is_dictionary(field.type):
                cats = list(categories[field.name])
                g = group.create_group(name)
                g.attrs.update({"zarr_conventions": [ENUM_CONVENTION], "enum:ordered": False, "enum:codes": ["codes"]})
                code_dtype = np.int16 if len(cats) < 2**15 else np.int32
                g.create_array("enum_values", data=np.asarray(cats, dtype=str), chunks=(max(1, len(cats)),))
                self.arrays[field.name] = g.create_array(
                    "codes", shape=(n_rows,), dtype=code_dtype, chunks=chunks, shards=shards, fill_value=-1
                )
            else:
                dtype = _numpy_dtype(field.type)
                self.arrays[field.name] = group.create_array(
                    name, shape=(n_rows,), dtype=dtype, chunks=chunks, shards=shards
                )

    def write(self, start: int, table: pa.Table) -> None:
        stop = start + table.num_rows
        for field in self.schema:
            col = table.column(field.name).combine_chunks()
            if pa.types.is_dictionary(field.type):
                values = col.indices.to_numpy(zero_copy_only=False)
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                values = np.asarray(col.to_pylist(), dtype=object)
            else:
                values = col.to_numpy(zero_copy_only=False)
            self.arrays[field.name][start:stop] = values


def read_zarr_df(
    group: zarr.Group,
    start: int | None = None,
    stop: int | None = None,
    columns: Sequence[str] | None = None,
    rows: np.ndarray | None = None,
) -> pd.DataFrame:
    """Read rows [start, stop) (or the given sorted row indices) of a ``df:`` group into pandas."""
    attrs = group.attrs.asdict()
    if not _has_convention(attrs, DF_CONVENTION):
        raise ValueError("Not a df: group.")
    names = [str(n) for n in group["columns"][:]]
    selected = names if columns is None else list(columns)
    out: dict[str, Any] = {}
    for name in selected:
        node = group[f"column_{names.index(name)}"]
        if isinstance(node, zarr.Group):
            g_attrs = node.attrs.asdict()
            assert _has_convention(g_attrs, ENUM_CONVENTION)
            codes_arr = node[g_attrs["enum:codes"][0]]
            codes = codes_arr.get_coordinate_selection(rows) if rows is not None else codes_arr[start:stop]
            cats = [str(v) for v in node["enum_values"][:]]
            out[name] = pd.Categorical.from_codes(codes, categories=cats)
        else:
            out[name] = node.get_coordinate_selection(rows) if rows is not None else node[start:stop]
    return pd.DataFrame(out)


def zarr_df_columns(group: zarr.Group) -> list[str]:
    return [str(n) for n in group["columns"][:]]


def zarr_df_n_rows(group: zarr.Group) -> int:
    return int(group.attrs["df:shape"][0])
