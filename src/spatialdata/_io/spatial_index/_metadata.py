"""Metadata of the ``spatial_index`` Zarr convention (RFC 3: tiled and multiscale points).

See https://spatialdata-spec.scverse.org/rfcs/0003-tiled-points.html.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

SPATIAL_INDEX_KEY = "spatial_index"
SPATIAL_INDEX_VERSION = "0.1.0-draft"
SPATIAL_INDEX_SPEC_URL = "https://spatialdata-spec.scverse.org/rfcs/0003-tiled-points.html"
SPATIAL_INDEX_CONVENTION = {
    "name": SPATIAL_INDEX_KEY,
    "spec_url": SPATIAL_INDEX_SPEC_URL,
    "description": "Tiled and multiscale points",
}
P_COLUMN = "__spatialdata_p__"
# p is drawn as an integer in [0, 2**P_BITS) and stored as the float32 value p_int / 2**P_BITS, which is exact
P_BITS = 24

Storage = Literal["parquet", "zarr"]
FeatureIndexLayout = Literal["dense", "sparse"]


@dataclass(frozen=True)
class LevelGrid:
    """The grid of one level: origin, tile size and number of tiles per axis, and the fraction P of the level."""

    origin: tuple[float, ...]
    tile_size: tuple[float, ...]
    grid_shape: tuple[int, ...]
    P: float

    @property
    def n_tiles(self) -> int:
        return int(np.prod(self.grid_shape))

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": [float(v) for v in self.origin],
            "tile_size": [float(v) for v in self.tile_size],
            "grid_shape": [int(v) for v in self.grid_shape],
            "P": float(self.P),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LevelGrid:
        return cls(
            origin=tuple(float(v) for v in d["origin"]),
            tile_size=tuple(float(v) for v in d["tile_size"]),
            grid_shape=tuple(int(v) for v in d["grid_shape"]),
            P=float(d["P"]),
        )

    def tile_coords(self, coords: np.ndarray) -> np.ndarray:
        """Tile coordinates (n, d) of positions (n, d), clipped to the grid."""
        origin = np.asarray(self.origin, dtype=np.float64)
        tile_size = np.asarray(self.tile_size, dtype=np.float64)
        t = np.floor((coords.astype(np.float64) - origin) / tile_size).astype(np.int64)
        return np.clip(t, 0, np.asarray(self.grid_shape, dtype=np.int64) - 1)

    def tile_ids(self, coords: np.ndarray) -> np.ndarray:
        """C-order tile ids of positions (n, d)."""
        t = self.tile_coords(coords)
        return np.ravel_multi_index(tuple(t.T), self.grid_shape).astype(np.int64)

    def tiles_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
        """Ids (sorted) of the tiles that intersect the closed box [box_min, box_max]."""
        origin = np.asarray(self.origin, dtype=np.float64)
        tile_size = np.asarray(self.tile_size, dtype=np.float64)
        shape = np.asarray(self.grid_shape, dtype=np.int64)
        lo = np.floor((np.asarray(box_min, dtype=np.float64) - origin) / tile_size).astype(np.int64)
        hi = np.floor((np.asarray(box_max, dtype=np.float64) - origin) / tile_size).astype(np.int64)
        if np.any(hi < 0) or np.any(lo >= shape):
            return np.empty(0, dtype=np.int64)
        lo = np.clip(lo, 0, shape - 1)
        hi = np.clip(hi, 0, shape - 1)
        ranges = [np.arange(a, b + 1) for a, b in zip(lo, hi, strict=True)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        ids = np.ravel_multi_index(tuple(m.ravel() for m in mesh), tuple(shape))
        return np.sort(ids.astype(np.int64))


@dataclass(frozen=True)
class SpatialIndexMetadata:
    """The ``spatial_index`` convention attributes of a points element."""

    axes: tuple[str, ...]
    levels: tuple[LevelGrid, ...]
    feature_keys: tuple[str, ...] = ()
    version: str = SPATIAL_INDEX_VERSION
    # implementation-specific extras (not in the RFC); kept in a separate key so that they are easy to spot
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def L(self) -> int:
        return len(self.levels) - 1

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "axes": list(self.axes),
            "feature_keys": list(self.feature_keys),
            "levels": [lvl.to_dict() for lvl in self.levels],
        }
        if self.extra:
            d["spatialdata_extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpatialIndexMetadata:
        md = cls(
            axes=tuple(d["axes"]),
            levels=tuple(LevelGrid.from_dict(lvl) for lvl in d["levels"]),
            feature_keys=tuple(d.get("feature_keys", [])),
            version=d["version"],
            extra=dict(d.get("spatialdata_extra", {})),
        )
        md.validate()
        return md

    def validate(self) -> None:
        if self.version != SPATIAL_INDEX_VERSION:
            raise ValueError(f"Unsupported spatial_index version {self.version!r}; expected {SPATIAL_INDEX_VERSION!r}.")
        if len(self.levels) == 0:
            raise ValueError("spatial_index: at least level 0 must be present.")
        d = len(self.axes)
        for lvl in self.levels:
            if not (len(lvl.origin) == len(lvl.tile_size) == len(lvl.grid_shape) == d):
                raise ValueError("spatial_index: origin, tile_size and grid_shape must have one entry per axis.")
        if self.levels[0].P != 1.0:
            raise ValueError("spatial_index: P of level 0 must be 1.")
        Ps = [lvl.P for lvl in self.levels]
        if any(b >= a for a, b in zip(Ps[:-1], Ps[1:], strict=True)):
            raise ValueError("spatial_index: the fractions P must strictly decrease with the level.")
        for fine, coarse in zip(self.levels[:-1], self.levels[1:], strict=True):
            if not refines(fine, coarse):
                raise ValueError("spatial_index: the grid of each level must refine the grid of the next level.")

    def band_of(self, p: np.ndarray) -> np.ndarray:
        """Band of each p: the largest l with p < P(l)."""
        Ps = np.asarray([lvl.P for lvl in self.levels], dtype=np.float64)
        # number of levels with p < P(l), minus 1 (P(0) = 1 > p always)
        return (np.searchsorted(-Ps, -np.asarray(p, dtype=np.float64), side="right") - 1).astype(np.int64)

    def coarsest_level_for_budget(self, b: float) -> int:
        """The coarsest level l with P(l) >= b (see "Reading with a budget")."""
        level = 0
        for i, lvl in enumerate(self.levels):
            if lvl.P >= b:
                level = i
        return level


def refines(fine: LevelGrid, coarse: LevelGrid, rtol: float = 1e-9) -> bool:
    """Whether every tile of ``fine`` lies inside a single tile of ``coarse``."""
    for o_f, s_f, n_f, o_c, s_c, n_c in zip(
        fine.origin, fine.tile_size, fine.grid_shape, coarse.origin, coarse.tile_size, coarse.grid_shape, strict=True
    ):
        ratio = s_c / s_f
        if abs(ratio - round(ratio)) > rtol * max(1.0, ratio):
            return False
        shift = (o_f - o_c) / s_c
        # the boundaries of the coarse tiles must be boundaries of fine tiles
        if abs((o_f - o_c) / s_f - round((o_f - o_c) / s_f)) > rtol * max(1.0, abs(shift)):
            return False
        if o_c > o_f + rtol * abs(s_f) or o_c + s_c * n_c < o_f + s_f * n_f - rtol * abs(s_f):
            return False
    return True


def merge_rule(level0: LevelGrid, n_levels: int | None = None) -> list[LevelGrid]:
    """Grids of all levels from the grid of level 0 with the merge rule of RFC 3 and the default rates.

    With ``n_levels=None`` the rule is applied until a single tile remains.
    """
    levels = [level0]
    v0 = float(np.prod(level0.tile_size))
    while n_levels is None or len(levels) < n_levels:
        prev = levels[-1]
        A = [i for i, n in enumerate(prev.grid_shape) if n > 1]
        if not A:
            # a single tile is left: the merge rule stops here
            break
        m = min(prev.tile_size[i] for i in A)
        M = {i for i in A if prev.tile_size[i] < 2 * m}
        tile_size = tuple(s * 2 if i in M else s for i, s in enumerate(prev.tile_size))
        grid_shape = tuple(math.ceil(n / 2) if i in M else n for i, n in enumerate(prev.grid_shape))
        P = v0 / float(np.prod(tile_size))
        levels.append(LevelGrid(prev.origin, tile_size, grid_shape, P))
    return levels
