"""Element relationship edges (sp-ops spec assumption A6, decision D9).

A generic, in-memory edge list linking elements of a :class:`~spatialdata.SpatialData` object,
as an alternative to (or complement of) the legacy ``region``/``region_key``/``instance_key``
table annotation triple. See the sp-ops specification's ``joinable-components.md`` for the
edge JSON shape this mirrors.

This module is purely in-memory: it is not (yet) serialized to or read from a Zarr store.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata.models import Labels2DModel, Labels3DModel, get_model

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData

__all__ = ["Edge", "Relationships"]

_CARDINALITY_RE = re.compile(r"^(n|\d+):(n|\d+)$")

#: Sentinel used on one side of ``on`` to mean "the pixel value of a Labels element", following
#: the sp-ops spec's convention (see ``joinable-components.md``: "`value` on a labels element
#: means the pixel value").
_VALUE = "value"


@dataclass(frozen=True)
class Edge:
    """One relationship edge between two elements of a SpatialData object.

    Parameters
    ----------
    from_
        Name of the source element.
    to
        Name of the target element.
    method
        ``"join"`` for a key join (column-to-column, or column-to-labels-pixel-value), or
        ``"sjoin"`` for a spatial join (geometry predicate).
    on
        For ``method="join"``: ``{"left": <column or "value">, "right": <column or "value">}``,
        where ``"left"`` refers to ``from_`` and ``"right"`` to ``to``, and the literal string
        ``"value"`` means the pixel value of a Labels element on that side.
        For ``method="sjoin"``: ``{"predicate": <a geopandas sjoin predicate, e.g. "within">}``.
    status
        ``"computed"`` if the join column/result already exists, ``"suggested"`` if a reader
        would have to compute it.
    cardinality
        Optional cardinality of the edge, e.g. ``"1:1"``, ``"n:1"``, ``"1:n"``, ``"n:n"``.
    """

    from_: str
    to: str
    method: Literal["join", "sjoin"]
    on: dict[str, str]
    status: Literal["computed", "suggested"] = "computed"
    cardinality: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("join", "sjoin"):
            raise ValueError(f"Edge.method must be 'join' or 'sjoin', got {self.method!r}.")
        if self.status not in ("computed", "suggested"):
            raise ValueError(f"Edge.status must be 'computed' or 'suggested', got {self.status!r}.")
        if self.cardinality is not None and not _CARDINALITY_RE.match(self.cardinality):
            raise ValueError(
                f"Edge.cardinality must look like '1:1', 'n:1', '1:n' or 'n:n', got {self.cardinality!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the sp-ops spec's edge JSON shape."""
        d: dict[str, Any] = {
            "from": self.from_,
            "to": self.to,
            "method": self.method,
            "on": dict(self.on),
            "status": self.status,
        }
        if self.cardinality is not None:
            d["cardinality"] = self.cardinality
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        """Deserialize from the sp-ops spec's edge JSON shape."""
        return cls(
            from_=d["from"],
            to=d["to"],
            method=d.get("method", "join"),
            on=dict(d["on"]),
            status=d.get("status", "computed"),
            cardinality=d.get("cardinality"),
        )


class Relationships:
    """A list-like container of :class:`Edge` objects."""

    def __init__(self, edges: list[Edge] | None = None) -> None:
        self._edges: list[Edge] = list(edges) if edges is not None else []

    def add(self, edge: Edge) -> None:
        """Append an edge to the list."""
        self._edges.append(edge)

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._edges)

    def __getitem__(self, index: int) -> Edge:
        return self._edges[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationships):
            return NotImplemented
        return self._edges == other._edges

    def __repr__(self) -> str:
        return f"Relationships({self._edges!r})"

    def for_element(self, name: str, direction: Literal["from", "to", "either"] = "either") -> Relationships:
        """Return the edges that touch ``name``, restricted to the given ``direction``."""
        if direction == "from":
            edges = [e for e in self._edges if e.from_ == name]
        elif direction == "to":
            edges = [e for e in self._edges if e.to == name]
        elif direction == "either":
            edges = [e for e in self._edges if e.from_ == name or e.to == name]
        else:
            raise ValueError(f"direction must be 'from', 'to' or 'either', got {direction!r}.")
        return Relationships(edges)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all edges to the sp-ops spec's edge JSON shape (the ``edges`` list)."""
        return [e.to_dict() for e in self._edges]

    @classmethod
    def from_dicts(cls, edges: list[dict[str, Any]]) -> Relationships:
        """Deserialize from a list of sp-ops spec edge dicts."""
        return cls([Edge.from_dict(e) for e in edges])

    def check(self, sdata: SpatialData) -> list[str]:
        """Validate every edge against ``sdata``, without raising.

        Returns a list of human-readable problem descriptions (empty if everything is fine).
        A ``status="suggested"`` edge is not, by itself, a problem.
        """
        problems: list[str] = []
        for i, edge in enumerate(self._edges):
            for side, name in (("from", edge.from_), ("to", edge.to)):
                if name not in sdata:
                    problems.append(
                        f"edge {i} ({edge.from_!r} -> {edge.to!r}): {side!r} element {name!r} "
                        "not found in the SpatialData object."
                    )
            if edge.cardinality is not None and not _CARDINALITY_RE.match(edge.cardinality):
                problems.append(f"edge {i} ({edge.from_!r} -> {edge.to!r}): malformed cardinality {edge.cardinality!r}.")
        return problems

    def resolve(self, sdata: SpatialData, edge: Edge) -> pd.DataFrame | GeoDataFrame:
        """Compute the join or spatial join described by ``edge`` against ``sdata``.

        Returns a merged :class:`pandas.DataFrame` for ``method="join"`` (one row per matched
        pair), or a :class:`geopandas.GeoDataFrame` for ``method="sjoin"``.
        """
        from_elem = sdata[edge.from_]
        to_elem = sdata[edge.to]
        if edge.method == "join":
            return _resolve_join(from_elem, to_elem, edge.on)
        return _resolve_sjoin(from_elem, to_elem, edge.on)


def _is_labels(element: Any) -> bool:
    if not isinstance(element, (DataArray, DataTree)):
        return False
    try:
        return get_model(element) in (Labels2DModel, Labels3DModel)
    except (ValueError, TypeError):
        return False


def _labels_unique_values(element: DataArray | DataTree) -> np.ndarray:
    if isinstance(element, DataTree):
        data = next(iter(element["scale0"].values())).data
    else:
        data = element.data
    values = np.unique(np.asarray(data))
    return values[values != 0]


def _as_dataframe(element: Any) -> pd.DataFrame:
    if isinstance(element, AnnData):
        return element.obs
    if isinstance(element, DaskDataFrame):
        return element.compute()
    if isinstance(element, (pd.DataFrame, GeoDataFrame)):
        return element
    raise TypeError(f"Cannot resolve a key join against an element of type {type(element)}.")


def _resolve_join(from_elem: Any, to_elem: Any, on: dict[str, str]) -> pd.DataFrame:
    left_key, right_key = on["left"], on["right"]

    if left_key == _VALUE and _is_labels(from_elem):
        left_df = pd.DataFrame({_VALUE: _labels_unique_values(from_elem)})
        right_df = _as_dataframe(to_elem)
        return left_df.merge(right_df, left_on=_VALUE, right_on=right_key)
    if right_key == _VALUE and _is_labels(to_elem):
        left_df = _as_dataframe(from_elem)
        right_df = pd.DataFrame({_VALUE: _labels_unique_values(to_elem)})
        return left_df.merge(right_df, left_on=left_key, right_on=_VALUE)

    left_df = _as_dataframe(from_elem)
    right_df = _as_dataframe(to_elem)
    return left_df.merge(right_df, left_on=left_key, right_on=right_key)


def _as_geodataframe(element: Any) -> GeoDataFrame:
    if isinstance(element, GeoDataFrame):
        return element
    df = _as_dataframe(element)
    if "x" in df.columns and "y" in df.columns:
        return GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
    raise TypeError(f"Cannot resolve a spatial join against an element of type {type(element)}.")


def _resolve_sjoin(from_elem: Any, to_elem: Any, on: dict[str, str]) -> GeoDataFrame:
    predicate = on.get("predicate", "intersects")
    left_gdf = _as_geodataframe(from_elem)
    right_gdf = _as_geodataframe(to_elem)
    return gpd.sjoin(left_gdf, right_gdf, predicate=predicate)
