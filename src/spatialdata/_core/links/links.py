"""Live view over the links stored in ``SpatialData.attrs["spatialdata_link"]``."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from spatialdata._core.links.edges import LINK_ATTRS_KEY, Edge, validate_link_spec

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData

__all__ = ["Links"]


class Links:
    """Links between the elements of a :class:`~spatialdata.SpatialData` object.

    This is a view: edges are read from and written to ``sdata.attrs["spatialdata_link"]``, a JSON list persisted
    with the other attrs, so nothing is cached here and the view cannot go out of sync with the attrs.
    """

    def __init__(self, sdata: SpatialData) -> None:
        self._sdata = sdata

    @property
    def _raw(self) -> list[Any]:
        raw = self._sdata.attrs.get(LINK_ATTRS_KEY, [])
        if not isinstance(raw, list):
            raise ValueError(
                f"attrs[{LINK_ATTRS_KEY!r}] must be a list of edges, got {type(raw).__name__}; "
                "see `SpatialData.validate()`"
            )
        return raw

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self) -> Iterator[Edge]:
        return iter([Edge.from_dict(d) for d in self._raw])

    def __getitem__(self, index: int) -> Edge:
        if not isinstance(index, int):
            raise TypeError(f"edges are indexed by integer position, got {type(index).__name__}")
        return Edge.from_dict(self._raw[index])

    def __repr__(self) -> str:
        raw = self._sdata.attrs.get(LINK_ATTRS_KEY, [])
        if not isinstance(raw, list):
            return f"Links: invalid specification in attrs[{LINK_ATTRS_KEY!r}] (see `SpatialData.validate()`)"
        if len(raw) == 0:
            return "Links: 0 edges"
        rows: list[tuple[str, str, str, str]] = []
        names: set[str] = set()
        for i, d in enumerate(raw):
            try:
                edge = Edge.from_dict(d)
            except ValueError as e:
                rows.append((f"[{i}] <invalid edge: {e}>", "", "", ""))
                continue
            names.update(edge.elements)
            on = edge.on["predicate"] if edge.method == "sjoin" else f"{edge.on['left']} ↔ {edge.on['right']}"
            to = ", ".join(edge.targets)
            rows.append((f"[{i}] {edge.from_} ──({on})──▶ {to}", edge.method, edge.status, edge.cardinality or ""))
        width = max(len(r[0]) for r in rows)
        header = f"Links: {len(raw)} edge{'s' if len(raw) != 1 else ''} over {len(names)} elements"
        lines = [f"  {left:<{width}}  {method:<5} {status:<9} {card}".rstrip() for left, method, status, card in rows]
        return "\n".join([header, *lines])

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return a deep copy of the edges in their JSON form."""
        return copy.deepcopy(self._raw)

    @property
    def element_names(self) -> set[str]:
        """Names of all the elements touched by an edge."""
        return {name for edge in self for name in edge.elements}

    def add(
        self,
        from_: str,
        to: str | Sequence[str],
        *,
        on: Mapping[str, str] | tuple[str, str] | None = None,
        method: Literal["join", "sjoin"] = "join",
        predicate: str | None = None,
        status: Literal["computed", "suggested"] | None = None,
        cardinality: str | None = None,
        validate: bool = True,
    ) -> Edge:
        """Add an edge.

        Parameters
        ----------
        from_
            Name of the source element (the finer side for ``n:1`` links).
        to
            Name of the target element; a list of names when ``on`` is ``(<column>, "element")``, i.e. when the
            rows of ``from_`` refer to whole elements.
        on
            For ``method="join"``, the two keys as ``(left, right)`` or ``{"left": ..., "right": ...}``; each key is a
            column name, ``"index"``, ``"value"`` (labels pixel values) or ``"element"``.
        method
            ``"join"`` (key join) or ``"sjoin"`` (spatial join, see ``predicate``).
        predicate
            The geopandas predicate of a spatial join, e.g. ``"within"`` or ``"intersects"``.
        status
            Defaults to ``"computed"`` for key joins and ``"suggested"`` for spatial joins.
        cardinality
            Declared cardinality, e.g. ``"1:1"`` or ``"n:1"``.
        validate
            If `True`, every element named by the edge must exist in the object.

        Returns
        -------
        The added edge.
        """
        raw = self._raw
        if on is None:
            on_dict: dict[str, str] = {}
        elif isinstance(on, Mapping):
            on_dict = dict(on)
        elif isinstance(on, tuple | list) and len(on) == 2:
            on_dict = {"left": on[0], "right": on[1]}
        else:
            raise ValueError("`on` must be a (left, right) pair or a mapping with 'left' and 'right'")
        if predicate is not None:
            if method != "sjoin":
                raise ValueError("`predicate` can only be used with method='sjoin'")
            on_dict["predicate"] = predicate
        if status is None:
            status = "computed" if method == "join" else "suggested"
        to_normalized: str | tuple[str, ...] = to if isinstance(to, str) else tuple(to)
        try:
            edge = Edge(
                from_=from_, to=to_normalized, method=method, on=on_dict, status=status, cardinality=cardinality
            )
        except ValueError as e:
            raise ValueError(f"invalid edge: {e}") from None
        if validate:
            missing = [name for name in edge.elements if name not in self._sdata]
            if missing:
                raise ValueError(
                    f"element(s) {missing} not found in the SpatialData object "
                    "(pass validate=False to add the edge anyway)"
                )
        if LINK_ATTRS_KEY not in self._sdata.attrs:
            self._sdata.attrs[LINK_ATTRS_KEY] = raw
        raw.append(edge.to_dict())
        return edge

    def remove(self, edge: int | Edge) -> Edge:
        """Remove an edge, given by position or by value, and return it."""
        raw = self._raw
        if isinstance(edge, Edge):
            matches = [i for i, d in enumerate(raw) if Edge.from_dict(d) == edge]
            if not matches:
                raise ValueError(f"edge not found: {edge}")
            index = matches[0]
        else:
            index = edge
        removed = Edge.from_dict(raw[index])
        del raw[index]
        if not raw:
            del self._sdata.attrs[LINK_ATTRS_KEY]
        return removed

    def for_element(self, name: str, direction: Literal["from", "to", "either"] = "either") -> list[Edge]:
        """Return the edges touching ``name`` as source (``"from"``), as target (``"to"``) or on either side."""
        if direction not in ("from", "to", "either"):
            raise ValueError(f"direction must be 'from', 'to' or 'either', got {direction!r}")
        edges = []
        for edge in self:
            is_from = edge.from_ == name
            is_to = name in edge.targets
            selected = {"from": is_from, "to": is_to, "either": is_from or is_to}[direction]
            if selected:
                edges.append(edge)
        return edges

    def validate(self) -> list[Edge]:
        """Validate the stored edges and check that the elements they reference exist.

        Returns
        -------
        The parsed edges.

        Raises
        ------
        ValueError
            If the stored value is malformed or references unknown elements; all problems are reported at once.
        """
        element_names = [name for _, name, _ in self._sdata._gen_elements(include_tables=True)]
        return validate_link_spec(self._sdata.attrs.get(LINK_ATTRS_KEY, []), element_names=element_names)
