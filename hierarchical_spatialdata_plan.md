# Design Plan: Hierarchical SpatialData with Flat Zarr Layout

## Core idea

Eliminate the typed subdirectory structure (`images/`, `labels/`, etc.). Store every element at a
flat path relative to the zarr root, encoding the element type in each element's `.zattrs`. Allow
`/` in element names to express hierarchy. Reading a subfolder returns a SpatialData scoped to that
prefix.

---

## 1. Internal storage — single `_elements` dict

**Before** (`spatialdata.py:115–131`):

```python
# __init__ signature
def __init__(
    self,
    images: dict[str, Raster_T] | None = None,
    labels: dict[str, Raster_T] | None = None,
    points: dict[str, DaskDataFrame] | None = None,
    shapes: dict[str, GeoDataFrame] | None = None,
    tables: dict[str, AnnData] | None = None,
    ...
):
    self._images: Images = Images(shared_keys=self._shared_keys)
    self._labels: Labels = Labels(shared_keys=self._shared_keys)
    self._points: Points = Points(shared_keys=self._shared_keys)
    self._shapes: Shapes = Shapes(shared_keys=self._shared_keys)
    self._tables: Tables = Tables(shared_keys=self._shared_keys)
```

**After**:

```python
# __init__ signature — unified dict
def __init__(
    self,
    elements: dict[str, SpatialElement | AnnData] | None = None,
    attrs: Mapping[Any, Any] | None = None,
) -> None:
    self._path: Path | None = None
    self._elements: dict[str, SpatialElement | AnnData] = {}
    self.attrs = attrs or {}

    if elements:
        for k, v in elements.items():
            self[k] = v  # validation happens in __setitem__

# Typed views — computed properties, no separate storage
@property
def images(self) -> dict[str, Raster_T]:
    return {k: v for k, v in self._elements.items()
            if get_model(v) in (Image2DModel, Image3DModel)}

@property
def labels(self) -> dict[str, Raster_T]:
    return {k: v for k, v in self._elements.items()
            if get_model(v) in (Labels2DModel, Labels3DModel)}

@property
def points(self) -> dict[str, DaskDataFrame]:
    return {k: v for k, v in self._elements.items()
            if get_model(v) == PointsModel}

@property
def shapes(self) -> dict[str, GeoDataFrame]:
    return {k: v for k, v in self._elements.items()
            if get_model(v) == ShapesModel}

@property
def tables(self) -> dict[str, AnnData]:
    return {k: v for k, v in self._elements.items()
            if get_model(v) == TableModel}
```

---

## 2. Element access — `__getitem__` / `__setitem__`

**Before** (`spatialdata.py:2289–2356`):
- `sdata["image"]` searches all typed containers sequentially
- No `/` support; key must be unique across all containers

**After**:

```python
def __getitem__(self, key: str) -> SpatialElement | AnnData | SpatialData:
    # Exact match — return element
    if key in self._elements:
        return self._elements[key]

    # Prefix match — return sub-SpatialData view
    prefix = key.rstrip("/") + "/"
    sub_elements = {
        k[len(prefix):]: v
        for k, v in self._elements.items()
        if k.startswith(prefix)
    }
    if sub_elements:
        sub = SpatialData(elements=sub_elements)
        if self._path is not None:
            sub._path = self._path / key  # points at zarr subfolder
        return sub

    raise KeyError(key)

def __setitem__(self, key: str, value: SpatialElement | AnnData) -> None:
    # Validate key — must not contain ".."
    if ".." in key.split("/"):
        raise KeyError(f"Element name must not contain '..': {key!r}")
    get_model(value)  # validates element type, raises if invalid
    self._elements[key] = value

def __delitem__(self, key: str) -> None:
    if key in self._elements:
        del self._elements[key]
    else:
        # Delete an entire prefix group
        prefix = key.rstrip("/") + "/"
        keys_to_delete = [k for k in self._elements if k.startswith(prefix)]
        if not keys_to_delete:
            raise KeyError(key)
        for k in keys_to_delete:
            del self._elements[k]
```

Usage:

```python
# Before
sdata["dapi"]                          # image from images/dapi
sdata["cells"]                         # labels from labels/cells

# After — flat
sdata["dapi"]                          # element at my_data.zarr/dapi
sdata["cells"]                         # element at my_data.zarr/cells

# After — hierarchical
sdata["sample1/dapi"]                  # element at my_data.zarr/sample1/dapi
sdata["sample1/cells"]                 # element at my_data.zarr/sample1/cells

sub = sdata["sample1"]                 # SpatialData with {"dapi": ..., "cells": ...}
sub["dapi"]                            # same as sdata["sample1/dapi"]
sub._path                              # → my_data.zarr/sample1
```

---

## 3. Zarr on-disk layout

**Before** — type-based subdirectories:
```
my_data.zarr/
├── .zattrs          # {"spatialdata_attrs": {"version": "0.2"}}
├── images/
│   ├── dapi/        # OME-Zarr multiscale image
│   └── he/
├── labels/
│   └── cells/
├── shapes/
│   └── rois/
└── tables/
    └── cell_table/
```

**After** — flat with element type in `.zattrs`:
```
my_data.zarr/
├── .zattrs          # {"spatialdata_attrs": {"version": "0.3"}}
├── dapi/            # .zattrs: {"spatialdata_attrs": {"element_type": "image"}}
├── he/              # .zattrs: {"spatialdata_attrs": {"element_type": "image"}}
├── cells/           # .zattrs: {"spatialdata_attrs": {"element_type": "labels"}}
├── rois/            # .zattrs: {"spatialdata_attrs": {"element_type": "shapes"}}
└── cell_table/      # .zattrs: {"spatialdata_attrs": {"element_type": "table"}}

# — or hierarchical:
my_data.zarr/
├── .zattrs
├── sample1/
│   ├── dapi/        # {"spatialdata_attrs": {"element_type": "image"}}
│   ├── he/          # {"spatialdata_attrs": {"element_type": "image"}}
│   └── cells/       # {"spatialdata_attrs": {"element_type": "labels"}}
└── sample2/
    └── dapi/        # {"spatialdata_attrs": {"element_type": "image"}}
```

Intermediate group folders (`sample1/`) have no `element_type` in `.zattrs` — that is how the
reader distinguishes a structural folder from an element.

---

## 4. Read path — `io_zarr.py`

**Before** (`io_zarr.py:125–235`):
- Hard-coded iteration over 5 named groups (`images`, `labels`, `points`, `shapes`, `tables`)
- Element type inferred from folder name

**After**:

```python
def _scan_for_elements(
    group: zarr.Group,
    root_path: str,
    prefix: str = "",
) -> dict[str, tuple[str, str]]:
    """
    Recursively scan zarr group, yielding element_name -> (element_type, abs_path).
    Stops recursing into a sub-group once it is identified as an element.
    """
    found: dict[str, tuple[str, str]] = {}
    for name in group:
        if name.startswith("."):
            continue
        sub = group[name]
        elem_name = f"{prefix}{name}" if prefix else name
        attrs = sub.attrs.asdict()
        sdata_attrs = attrs.get("spatialdata_attrs", {})
        element_type = sdata_attrs.get("element_type")

        if element_type is not None:
            # Leaf: this is a SpatialData element
            abs_path = os.path.join(root_path, sub.path)
            found[elem_name] = (element_type, abs_path)
        elif hasattr(sub, "keys"):
            # Structural group: recurse
            found.update(_scan_for_elements(sub, root_path, prefix=f"{elem_name}/"))
    return found


def read_zarr(store: str | Path | UPath | zarr.Group, ...) -> SpatialData:
    root_group = zarr.open_group(store, mode="r")
    elements: dict[str, SpatialElement | AnnData] = {}

    for elem_name, (element_type, abs_path) in _scan_for_elements(root_group, ...).items():
        if element_type == "image":
            elements[elem_name] = _read_multiscale(abs_path, "image", ...)
        elif element_type == "labels":
            elements[elem_name] = _read_multiscale(abs_path, "labels", ...)
        elif element_type == "points":
            elements[elem_name] = _read_points(abs_path)
        elif element_type == "shapes":
            elements[elem_name] = _read_shapes(abs_path)
        elif element_type == "table":
            elements[elem_name] = _read_table(abs_path)

    sdata = SpatialData(elements=elements)
    sdata.path = store
    return sdata
```

Partial read — just open a sub-group:

```python
# Reads only elements under my_data.zarr/sample1 (names relative to opened root)
sub = SpatialData.read("my_data.zarr/sample1")
# sub._elements == {"dapi": ..., "cells": ...}

# Equivalent via zarr group:
root = zarr.open_group("my_data.zarr", mode="r")
sub = SpatialData.read(root["sample1"])
```

---

## 5. Write path — `io_zarr.py`

**Before** (`spatialdata.py:1103`, `io_zarr.py:238`):
- `_get_groups_for_element()` creates `root["images"]["dapi"]`
- Element type determines which subgroup to write into

**After**:

```python
def _write_element_flat(
    zarr_root: zarr.Group,
    element_name: str,       # e.g. "dapi" or "sample1/dapi"
    element: SpatialElement | AnnData,
    ...
) -> None:
    # zarr.require_group handles "/" transparently, creating intermediate groups
    elem_group = zarr_root.require_group(element_name)

    # Tag the group so the reader can identify the type
    element_type_str = _get_element_type_string(element)  # "image", "labels", ...
    elem_attrs = dict(elem_group.attrs)
    elem_attrs.setdefault("spatialdata_attrs", {})["element_type"] = element_type_str
    elem_group.attrs.update(elem_attrs)

    # Delegate to existing type-specific writers (unchanged internally)
    if element_type_str in ("image", "labels"):
        _write_multiscale(element, elem_group, ...)
    elif element_type_str == "shapes":
        _write_shapes(element, elem_group, ...)
    elif element_type_str == "points":
        _write_points(element, elem_group, ...)
    elif element_type_str == "table":
        _write_table(element, elem_group, ...)
```

---

## 6. `__repr__` — tree structure

**Before** (`spatialdata.py:1938`): flat per-type sections.

**After**: builds a prefix tree and renders with box-drawing characters.

Example output — flat elements:
```
SpatialData object at my_data.zarr
├── dapi [Image2D] DataArray[czyx] (1, 1024, 1024)
├── he [Image2D] DataArray[czyx] (3, 1024, 1024)
├── cells [Labels2D] DataArray[yx] (1024, 1024)
├── rois [Shapes] GeoDataFrame (10, 2)
└── cell_table [Table] AnnData (1000, 50)
```

Example output — hierarchical elements:
```
SpatialData object at my_data.zarr
├── rois [Shapes] GeoDataFrame (10, 2)
├── sample1/ (3 elements)
│   ├── dapi [Image2D] DataArray[czyx] (1, 1024, 1024)
│   ├── he [Image2D] DataArray[czyx] (3, 1024, 1024)
│   └── cells [Labels2D] DataArray[yx] (1024, 1024)
└── sample2/ (2 elements)
    ├── dapi [Image2D] DataArray[czyx] (1, 1024, 1024)
    └── cells [Labels2D] DataArray[yx] (1024, 1024)
```

Implementation sketch:

```python
def _gen_repr(self) -> str:
    from collections import defaultdict

    # Split elements into flat (no "/") and grouped (has "/", keyed by first component)
    tree: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    flat: list[tuple[str, Any]] = []

    for name, elem in sorted(self._elements.items(), key=lambda x: _natural_keys(x[0])):
        parts = name.split("/", 1)
        if len(parts) == 1:
            flat.append((name, elem))
        else:
            tree[parts[0]].append((parts[1], elem))

    lines = ["SpatialData object" + (f" at {self.path}" if self.path else "")]
    all_entries: list[tuple[str, Any]] = flat + [(k, v) for k, v in sorted(tree.items())]

    for i, entry in enumerate(all_entries):
        is_last = i == len(all_entries) - 1
        branch = "└──" if is_last else "├──"
        child_indent = "    " if is_last else "│   "

        if entry[0] in tree:
            # Folder entry
            folder = entry[0]
            children = tree[folder]
            lines.append(f"{branch} {folder}/ ({len(children)} elements)")
            for j, (child_name, child_elem) in enumerate(children):
                child_branch = "└──" if j == len(children) - 1 else "├──"
                lines.append(f"{child_indent}{child_branch} {child_name} {_format_element(child_elem)}")
        else:
            name, elem = entry
            lines.append(f"{branch} {name} {_format_element(elem)}")

    return "\n".join(lines)
```

---

## 7. Backward compatibility

The reader detects the version from root `.zattrs` and dispatches accordingly:

```python
if sdata_version == "0.3":
    # new flat layout — scan recursively for element_type markers
    ...
elif sdata_version in ("0.1", "0.2"):
    # old typed-subdir layout — existing reader unchanged
    ...
```

The old typed-dict constructor is kept as a deprecated factory:

```python
@classmethod
def from_typed_dicts(
    cls,
    images: dict[str, Raster_T] | None = None,
    labels: dict[str, Raster_T] | None = None,
    points: dict[str, DaskDataFrame] | None = None,
    shapes: dict[str, GeoDataFrame] | None = None,
    tables: dict[str, AnnData] | None = None,
    **kwargs,
) -> SpatialData:
    """Backward-compatible constructor. Deprecated in v0.3."""
    import warnings
    warnings.warn(
        "from_typed_dicts() is deprecated. Use SpatialData(elements={...}) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    elements: dict[str, SpatialElement | AnnData] = {}
    for k, v in (images or {}).items():
        elements[k] = v
    for k, v in (labels or {}).items():
        elements[k] = v
    for k, v in (points or {}).items():
        elements[k] = v
    for k, v in (shapes or {}).items():
        elements[k] = v
    for k, v in (tables or {}).items():
        elements[k] = v
    return cls(elements=elements, **kwargs)
```

---

## 8. Summary of files to change

| File | Change |
|---|---|
| `_core/spatialdata.py` | Single `_elements` dict; new `__getitem__`/`__setitem__`/`__delitem__`; typed views as computed properties; new `_gen_repr` with tree rendering |
| `_core/_elements.py` | Can be removed or repurposed as a thin validation helper |
| `_io/io_zarr.py` | Replace typed group iteration with `_scan_for_elements`; new `_write_element_flat` |
| `_io/io_raster.py`, `io_points.py`, `io_shapes.py`, `io_table.py` | Minor: writers receive a pre-created group (caller no longer wraps them in `images/`-style parent) |
| `_io/format.py` | Add version `"0.3"` and document the new layout |

---

## Open questions

- Should `sdata["sample1"]` when `sample1` has no elements raise `KeyError` or return an empty
  `SpatialData`?
- Should writing `sdata["sample1/image"] = img` immediately persist to disk if `sdata.path` is set
  (like the current `write_element` incremental API), or remain lazy?
- Tables annotate regions by element name; should the region reference store the full path
  (`"sample1/cells"`) or remain relative to the SpatialData root?
- Is depth limited (only one level of `/`)? Or is arbitrary nesting (`a/b/c/image`) allowed?
- Should `sdata.images` return only top-level images, or all images at any depth?
