"""The metadata of the spatial index: ``spatial_index`` in ``zarr.json``, and the Parquet sidecar.

- ``zarr.json``: the ``spatial_index`` Zarr convention (version, axes, feature keys, p_bits, levels, ...), the same in
  both formats, plus, in Parquet only, ``"parquet": "spatial_index.parquet.json"``.
- ``spatial_index.parquet.json`` (Parquet only): for each table (``spatial`` and ``feature/<key>``) its part files in
  order (relative to the element), the first row of each (``first_row[i]`` is the first row of ``files[i]``; the last
  entry is the number of rows) and, for a feature-major table, the global category list (``categories``) whose
  codes the feature index uses. In Zarr the global list is the ``enum_values`` of the feature column.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import zarr

from spatialdata._io.spatial_index._metadata import (
    PARQUET_SIDECAR,
    SPATIAL_INDEX_CONVENTION,
    SPATIAL_INDEX_KEY,
    SpatialIndexMetadata,
)


def write_spatial_index(element_path: Path, md: SpatialIndexMetadata, parquet_sidecar: dict[str, Any] | None) -> None:
    """Write ``spatial_index`` into ``zarr.json`` and, for Parquet, the sidecar next to it."""
    element_path = Path(element_path)
    if parquet_sidecar is not None:
        (element_path / PARQUET_SIDECAR).write_text(json.dumps(parquet_sidecar, indent=1, ensure_ascii=False) + "\n")
        md = replace(md, parquet=PARQUET_SIDECAR)
    group = zarr.open_group(element_path, mode="a")
    attrs = group.attrs.asdict()
    conventions = [c for c in attrs.get("zarr_conventions", []) if c.get("name") != SPATIAL_INDEX_KEY]
    conventions.append(SPATIAL_INDEX_CONVENTION)
    group.attrs.update({"zarr_conventions": conventions, SPATIAL_INDEX_KEY: md.to_dict()})


def update_spatial_index(element_path: Path, **changes: Any) -> SpatialIndexMetadata:
    """Change fields of ``spatial_index`` (e.g. ``table``) in ``zarr.json``."""
    group = zarr.open_group(Path(element_path), mode="a")
    md = replace(SpatialIndexMetadata.from_dict(group.attrs[SPATIAL_INDEX_KEY]), **changes)
    group.attrs[SPATIAL_INDEX_KEY] = md.to_dict()
    return md


def read_metadata(element_path: str | Path, fs: Any = None) -> tuple[SpatialIndexMetadata, dict[str, Any] | None]:
    """The index metadata (read as plain JSON) and the Parquet sidecar (``None`` for the Zarr format).

    ``fs``: an fsspec filesystem for a remote element (``element_path`` is then its URL).
    """

    def read_json(rel: str) -> Any:
        if fs is None:
            p = Path(element_path) / rel
            return json.loads(p.read_text()) if p.exists() else None
        try:
            return json.loads(fs.cat_file(f"{str(element_path).rstrip('/')}/{rel}"))
        except FileNotFoundError:
            return None

    zj = read_json("zarr.json") or {}
    attrs = zj.get("attributes", {})
    if SPATIAL_INDEX_KEY not in attrs:
        raise ValueError(f"{element_path} has no {SPATIAL_INDEX_KEY} metadata.")
    md = SpatialIndexMetadata.from_dict(attrs[SPATIAL_INDEX_KEY])
    sidecar = read_json(md.parquet) if md.parquet is not None else None
    return md, sidecar
