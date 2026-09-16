"""Description of a link between elements of a SpatialData object.

The links of a :class:`~spatialdata.SpatialData` object are stored as a JSON list of edges in
``sdata.attrs["spatialdata_link"]``; each edge says which two elements are linked and how (a key join between two
columns, or a spatial join). This module holds the edge description and its validation and must stay importable
without the rest of :mod:`spatialdata._core`, since the IO formats use it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["ELEMENT", "INDEX", "LINK_ATTRS_KEY", "VALUE", "Edge", "validate_link_spec"]

#: Key of ``SpatialData.attrs`` holding the list of edges.
LINK_ATTRS_KEY = "spatialdata_link"

#: ``on`` sentinel: the index of a dataframe-like element (``obs_names`` for tables).
INDEX = "index"
#: ``on`` sentinel: the pixel values of a labels element.
VALUE = "value"
#: ``on`` sentinel: the target elements themselves are the instances (the column holds element names).
ELEMENT = "element"

_METHODS = ("join", "sjoin")
_STATUSES = ("computed", "suggested")
_EDGE_KEYS = frozenset({"from", "to", "method", "on", "status", "cardinality"})
_REQUIRED_EDGE_KEYS = ("from", "to", "method", "on")
_CARDINALITY_RE = re.compile(r"^(n|\d+):(n|\d+)$")


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


@dataclass(frozen=True)
class Edge:
    """One link between two elements of a SpatialData object.

    Parameters
    ----------
    from_
        Name of the source element. For ``n:1`` links this is the finer side (e.g. transcripts pointing to cells).
    to
        Name of the target element, or a tuple of element names when ``on["right"] == "element"`` (the rows of
        ``from_`` refer to whole elements, one element per row).
    method
        ``"join"`` for a key join between two columns, ``"sjoin"`` for a spatial join.
    on
        For ``method="join"``: ``{"left": <key>, "right": <key>}``, where ``"left"`` refers to ``from_`` and
        ``"right"`` to ``to``. A key is a column name, ``"index"`` (the index of a dataframe-like element, or the
        ``obs_names`` of a table), ``"value"`` (the pixel values of a labels element) or ``"element"`` (the target
        elements themselves). For ``method="sjoin"``: ``{"predicate": <geopandas predicate>}``.
    status
        ``"computed"`` if the columns named in ``on`` exist, ``"suggested"`` if the link still has to be computed
        (spatial joins are resolved lazily when queried).
    cardinality
        Optional declared cardinality, e.g. ``"1:1"``, ``"n:1"``, ``"1:n"``, ``"n:n"``. A declaration only: it does not
        imply that rows are in the same order.
    """

    from_: str
    to: str | tuple[str, ...]
    method: Literal["join", "sjoin"] = "join"
    on: Mapping[str, str] = field(default_factory=dict)
    status: Literal["computed", "suggested"] = "computed"
    cardinality: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.to, list):
            object.__setattr__(self, "to", tuple(self.to))
        if isinstance(self.on, Mapping):
            object.__setattr__(self, "on", dict(self.on))
        problems = self._problems()
        if problems:
            raise ValueError("; ".join(problems))

    def _problems(self) -> list[str]:
        problems: list[str] = []
        if not _is_name(self.from_):
            problems.append("'from' must be a non-empty string")
        if isinstance(self.to, tuple):
            if not self.to or not all(_is_name(t) for t in self.to):
                problems.append("'to' must be a non-empty string or a non-empty list of non-empty strings")
        elif not _is_name(self.to):
            problems.append("'to' must be a non-empty string or a non-empty list of non-empty strings")
        if self.method not in _METHODS:
            problems.append(f"'method' must be one of {list(_METHODS)}, got {self.method!r}")
        if self.status not in _STATUSES:
            problems.append(f"'status' must be one of {list(_STATUSES)}, got {self.status!r}")
        if self.cardinality is not None and not (
            isinstance(self.cardinality, str) and _CARDINALITY_RE.match(self.cardinality)
        ):
            problems.append(f"'cardinality' must look like '1:1', 'n:1', '1:n' or 'n:n', got {self.cardinality!r}")
        if not isinstance(self.on, Mapping):
            problems.append("'on' must be a mapping")
            return problems
        if self.method == "join":
            missing = [side for side in ("left", "right") if side not in self.on]
            if missing:
                problems.append(f"'on' must contain 'left' and 'right' for method 'join' (missing {missing})")
            for side in ("left", "right"):
                if side in self.on and not _is_name(self.on[side]):
                    problems.append(f"'on[{side!r}]' must be a non-empty string")
            if isinstance(self.to, tuple) and self.on.get("right") != ELEMENT:
                problems.append("'to' can list several elements only when on['right'] == 'element'")
        elif self.method == "sjoin":
            if not _is_name(self.on.get("predicate")):
                problems.append("'on' must contain a non-empty 'predicate' for method 'sjoin'")
            if isinstance(self.to, tuple):
                problems.append("'to' must be a single element for method 'sjoin'")
        return problems

    @property
    def targets(self) -> tuple[str, ...]:
        """The target element names (one, unless the rows of ``from_`` refer to whole elements)."""
        return self.to if isinstance(self.to, tuple) else (self.to,)

    @property
    def elements(self) -> tuple[str, ...]:
        """All element names touched by the edge: the source first, then the targets."""
        return (self.from_, *self.targets)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape stored in ``SpatialData.attrs``."""
        d: dict[str, Any] = {
            "from": self.from_,
            "to": list(self.to) if isinstance(self.to, tuple) else self.to,
            "method": self.method,
            "on": dict(self.on),
            "status": self.status,
        }
        if self.cardinality is not None:
            d["cardinality"] = self.cardinality
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Edge:
        """Deserialize from the JSON shape stored in ``SpatialData.attrs``."""
        if not isinstance(d, Mapping):
            raise ValueError(f"an edge must be a mapping, got {type(d).__name__}")
        missing = [k for k in _REQUIRED_EDGE_KEYS if k not in d]
        if missing:
            raise ValueError(f"missing keys {', '.join(repr(k) for k in missing)}")
        unknown = sorted(set(d) - _EDGE_KEYS)
        if unknown:
            raise ValueError(f"unknown keys {', '.join(repr(k) for k in unknown)}")
        to = d["to"]
        if isinstance(to, list):
            to = tuple(to)
        return cls(
            from_=d["from"],
            to=to,
            method=d["method"],
            on=d["on"],
            status=d.get("status", "computed"),
            cardinality=d.get("cardinality"),
        )


def validate_link_spec(spec: Any, element_names: Iterable[str] | None = None) -> list[Edge]:
    """Validate the value stored in ``sdata.attrs["spatialdata_link"]``.

    Parameters
    ----------
    spec
        The value to validate: a list of edge dictionaries.
    element_names
        If given, every element referenced by an edge must be among these names.

    Returns
    -------
    The parsed edges.

    Raises
    ------
    ValueError
        If ``spec`` is not a list of valid edges, or references unknown elements. All problems are reported at once.
    """
    prefix = f"attrs[{LINK_ATTRS_KEY!r}] is not a valid link specification"
    if not isinstance(spec, list):
        raise ValueError(f"{prefix}: expected a list of edges, got {type(spec).__name__}")
    edges: list[Edge] = []
    problems: list[str] = []
    for i, item in enumerate(spec):
        try:
            edges.append(Edge.from_dict(item))
        except ValueError as e:
            problems.append(f"edge {i}: {e}")
    if problems:
        raise ValueError(f"{prefix} ({'; '.join(problems)})")
    if element_names is not None:
        names = set(element_names)
        for i, edge in enumerate(edges):
            for name in edge.elements:
                if name not in names:
                    problems.append(f"edge {i} ({edge.from_!r} -> {edge.to!r}): element {name!r} not found")
        if problems:
            raise ValueError(f"{prefix} ({'; '.join(problems)})")
    return edges
