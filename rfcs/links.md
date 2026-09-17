# RFC: instance-level linking for SpatialData (mulink-compatible)

## Authors and acknowledgements

_Draft._ The list of authors will be completed before the RFC is formally
presented for review. Both the authors and the acknowledgements will include
the people who worked on element relationships during the hackathons where
this design took shape — the scverse proteomics hackathon (Berlin, 2026), the
scverse hackathon in Padua (2026) and the Bioconductor hackathon in Venice
(2026) — as well as the participants in the MuData feature-mapping discussion
(see References).

## Summary

`SpatialData` should be able to answer relationship queries between the
**instances** of its elements — "the transcripts of these cells", "the cells
inside this ROI", "the rows of this table that annotate that segmentation" —
through one declarative, persisted description of how elements relate and one
ergonomic query API. Today a single relationship is encoded (a table
annotating an element through the `region`/`region_key`/`instance_key` triple);
every other relationship lives in ad-hoc columns and manual joins.

The proposal separates three concerns that are easy to conflate:

1. **Schema** — _which_ elements are linked and _how_ (join on which columns,
   spatial predicate, cardinality). Stored as a JSON list of edges in
   `sdata.attrs["spatialdata_link"]` (in memory; serialized with the rest of
   `attrs`), validated by a new container-format version.
2. **Encoding** — the actual per-instance relationships. Stored as **columns on
   the elements themselves** (a table's `obs`, a `GeoDataFrame`'s columns, a
   points dataframe's columns; labels use their pixel values). A table that
   annotates several elements gets one column per element, `NA` where a row
   does not annotate that element.
3. **Execution** — running a query across one or more hops. The default
   backend uses spatialdata's existing join operations and geopandas spatial
   joins, with a breadth-first traversal over the (small) element-level graph.
   Two optional backends can execute the same schema: **mulink + MuData**
   (materialize the instance graph as a sparse `obsp` matrix and traverse it
   with mulink) and **DuckDB** (register element columns as views and run the
   joins as SQL).

For comparison, MuData/mulink fold 1 and 2 into one object — a sparse
adjacency matrix in `varp` (an `obsp` variant is planned, not implemented)
whose key name and shape are the only schema — and mulink implements 3 on top
of it. Here the schema and the encoding
stay separate so that the same description can be executed by any backend, and
so that the encoding stays in the data structures users already know
(dataframe columns), not in a P×P matrix.

MuData only supports `AnnData` modalities. To make shapes, points and labels
participate one could (a) turn everything into `AnnData` (not possible today;
may or may not become an option), (b) wrap each element in a proxy `AnnData`, or
(c) define a small **container protocol** — "what are the instances, what is the
data, how to subset" — that non-AnnData elements implement. This RFC takes (c).
The protocol is used by _every_ backend inside spatialdata (it is how the
native backend reads ids and subsets shapes, points and labels uniformly);
its role for MuData is specific to the mulink + MuData backend, where it is the
concrete artifact to propose so that MuData can host non-AnnData modalities.

## Decisions

| #   | Topic                   | Decision                                                                                                                                                                                                                                                                                                                              |
| --- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Edge description        | One JSON shape per edge, `from/to/method/on/status/cardinality` (as in [LucaMarconato/spatialdata#3](https://github.com/LucaMarconato/spatialdata/pull/3), branch `vibecoded-experiment/sp-ops-relationships`). `add_group([...])` is Python sugar that expands 1:1 id-sharing groups into star edges; nothing new on disk            |
| 2   | APIs                    | `sdata.links` accessor (schema, validation, backends, graph export) and `sdata.subset_by(...)` (query)                                                                                                                                                                                                                                |
| 3   | Storage of the schema   | `sdata.attrs["spatialdata_link"]`, in memory, serialized with `attrs` → **container format V03 ("0.3")**                                                                                                                                                                                                                              |
| 4   | Storage of the encoding | Columns on elements; **table format V03** replaces `region`/`region_key`/`instance_key` with one instance-id column per annotated element. Table metadata keeps only a version: which column links to which element is stated by edges in `attrs["spatialdata_link"]`, the single place for all links (details in _Table format V03_) |
| 5   | `subset_by` return      | Filtered `SpatialData`; elements with no link path are dropped and listed; `keep_unlinked=True` passes them through untouched                                                                                                                                                                                                         |
| 6   | Validation              | On write (V03 format), when an edge is added (unless `validate=False`), and on demand via a public `sdata.validate()` that also runs the existing element/table validations. Malformed links found on read (e.g. edge to a missing element) are tolerated with a warning                                                              |
| 7   | Direction               | Undirected transitive traversal by default; `direction="up"\|"down"` restricts to coarser/finer (mulink's ancestors/descendants)                                                                                                                                                                                                      |
| 8   | Instance identity       | Every element needs an identity: `index`, a named column, or pixel values for labels. Identity values should be unique; on duplicates a **warning** is raised, the first row is used, and the data is not modified                                                                                                                    |
| 9   | Execution backend       | Default `"native"` (spatialdata joins + geopandas `sjoin`; no mudata/mulink needed). Optional `"mulink"` (MuData `obsp` + mulink traversal) and `"duckdb"` backends, each behind its own optional dependency                                                                                                                          |
| 10  | Dependencies            | Core: none new. Extras: `spatialdata[links-mulink]` adds `mulink` (which brings `mudata`, `networkx`, `matplotlib`); `spatialdata[links-duckdb]` adds `duckdb`. Development installs the forks `LucaMarconato/mulink` and, if needed, `LucaMarconato/mudata`                                                                          |

### Why one column per annotated element (decision 4)

Today a table annotating several elements needs two `obs` columns that act
together as a multi-index: `region_key` (which element) and `instance_key`
(which instance in it). Every consumer must know to combine them, and a single
row can never annotate two elements. The new design removes the multi-index:

- there is no `region` metadata and no `region_key` column;
- the former `instance_key` column keeps its values but is **named after the
  element it annotates** (by default; any name is allowed since the edge names
  the column);
- a table may have several such columns, one per annotated element; each column
  is one link to one external element; rows that do not refer to that element
  are `NA` (nullable `Int64`/`string` dtypes, which anndata round-trips);
- no `where` clause is needed to run a query.

The cost is a possibly wider table (one column per annotated element instead
of two columns total). This is a design change with respect to the current
spatialdata table model; conversion in both directions is provided.

## Scope: which links are covered

Terminology: an **Instance** is an entity of interest (a cell, a transcript, a
label id, a sample); an **Element** is a collection of instances of one type
(`GeoDataFrame`, `DaskDataFrame`, `DataArray`/`DataTree`, `AnnData`); a
**Container** holds elements (`SpatialData`, `MuData`).

Links can cross these levels: a row of a table may refer to an instance of an
element (a cell), to an element (an image, a table), or to a container (a
`SpatialData` object of a multi-sample study).

| Link                                             | Example                                                                                                                                                                                             | In this RFC?                                                                                                      |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| instance ↔ instance, same `SpatialData`          | table row ↔ cell shape; transcript → cell; cell → ROI; label id ↔ table row                                                                                                                         | yes                                                                                                               |
| instance ↔ element (element as instance)         | a table with one row per image, or one row per table: `to` lists the elements (or an element type), `on.right = "element"`, the column holds element names; the element counts as a single instance | yes                                                                                                               |
| n:m produced by a spatial join                   | overlapping polygons with `predicate="intersects"`                                                                                                                                                  | yes for querying (`status="suggested"` edges); no for storing: an n:m result cannot be materialized into a column |
| instance ↔ container                             | a table with one row per `SpatialData` object                                                                                                                                                       | no (possible future update; needs a container of containers)                                                      |
| instance ↔ instance across `SpatialData` objects | cells of replicate 1 ↔ cells of replicate 2                                                                                                                                                         | no (possible future update)                                                                                       |
| stored n:m links                                 | overlap between two segmentations as a junction table or matrix                                                                                                                                     | no (possible future update; needs an array-backed encoding, not columns)                                          |
| var-axis links                                   | image channels ↔ table `var_names`                                                                                                                                                                  | no (possible future update; mulink's native case, same accessor)                                                  |

## Background and constraints

This RFC is part of a cross-project scverse effort (see References): a
feature-level DAG for MuData (mulink), an element-relationship prototype from
the scverse Padua hackathon, and an edge-list prototype in a spatialdata fork
developed during the Bioconductor Venice hackathon.

| Constraint                                                                                                                                                                                                                                                                                                                                                                                                                                              | How it is addressed                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mulink's graph lives on the var axis only (`mdata.varp[key]`)                                                                                                                                                                                                                                                                                                                                                                                           | Native backend does not need mulink; the mulink backend uses `obsp` and an obs-axis `QueryAccessor` added in the mulink fork                                                                                                             |
| `MuData` assumes `AnnData` modalities (`_check_duplicated_names`, `update`, repr, IO access `.obs/.var/.obs_names`; `mudata/_core/mudata.py` ~lines 100–130, 210–235)                                                                                                                                                                                                                                                                                   | Container protocol defined in spatialdata; `to_mudata()` works today for tables-only graphs; full support needs a mudata fork PR (see _Development workflow_)                                                                            |
| MuData's `axis` says which axis modalities share: `axis=0` — observations shared, variables separate (the default: same cells, different feature sets); `axis=1` — variables shared, observations separate (same features, different sample sets); `axis=-1` — both shared. A graph over instances of different elements needs _neither_ shared (every element brings its own instances and its own columns); call that `axis=-2`, which does not exist | Prefix instance ids with the element name (`"table-12034"`) so that `axis=0` with pairwise-disjoint `obs_names` behaves like "observations separate"; propose `axis=-2` in the mudata fork                                               |
| `sdata.attrs` is JSON-only, persisted by `SpatialData.write_attrs` (`src/spatialdata/_core/spatialdata.py` ~line 1777)                                                                                                                                                                                                                                                                                                                                  | Schema stored as JSON there; container format V03 validates the reserved key                                                                                                                                                             |
| `attrs` is passed by reference through `subset()`/`query()`                                                                                                                                                                                                                                                                                                                                                                                             | `subset_by` deep-copies `attrs`; the same change is proposed for the other query functions (follow-up)                                                                                                                                   |
| Neither `mudata` nor `mulink` is a dependency (`pyproject.toml`)                                                                                                                                                                                                                                                                                                                                                                                        | Optional extras; the native backend has no new dependency                                                                                                                                                                                |
| A P×P instance adjacency is a poor fit for 1:1 groups and n:1 columns                                                                                                                                                                                                                                                                                                                                                                                   | Native backend traverses the element graph and joins columns; the instance matrix is built only by the mulink backend                                                                                                                    |
| Points are dask-backed and may lack stable ids: the default index restarts per partition and changes with repartitioning                                                                                                                                                                                                                                                                                                                                | Recommendation: use a stable id **column** as identity. `index` is accepted only when it is known to be unique (dask `known_divisions`, or verified once); otherwise `index_element()` writes one. Duplicates warn and use the first row |

## Design

### Schema: edge description

`attrs["spatialdata_link"]` is a JSON list of edges:

```jsonc
{"from": "transcripts", "to": "cell_boundaries", "method": "join",
 "on": {"left": "cell_id", "right": "index"}, "status": "computed", "cardinality": "n:1"}
{"from": "cell_boundaries", "to": "rois", "method": "sjoin",
 "on": {"predicate": "within"}, "status": "suggested"}
```

- `on.left`/`on.right` ∈ `"index"` | `"value"` (labels pixel values) |
  `<column>`; for tables `"index"` is `obs_names`, a column is `obs[col]`.
- Direction: `from` is the finer/child side, `to` the coarser/parent side
  for `n:1` (mulink's precursor→protein convention); irrelevant for `1:1`.
- `cardinality` is a declaration, not a guarantee. A declared `1:1` says
  nothing about row **order**; `add_group(..., ordered=True)` additionally
  promises matching order. `check()` verifies both and warns on mismatch.
- `status="suggested"` marks a spatial join that is declared but not yet
  computed; it is resolved lazily on first query and cached. `materialize()`
  writes the result as a column on `from` and rewrites the edge to
  `method="join", status="computed"`.
- All edges live in this list, including table → element edges; table
  metadata does not repeat them (single source of truth).
- Edges that reference missing elements are skipped with a warning.

### Encoding: table format V03

Table metadata `uns["spatialdata_attrs"]` changes from
`{"region", "region_key", "instance_key"}` to `{"version": "0.3"}`. Which
`obs` column links to which element is stated **only** by edges in
`attrs["spatialdata_link"]` (`from=<table>, on.left=<column>, to=<element>,
on.right="index"|"value"`). A table outside a `SpatialData` therefore carries
its id columns but not their meaning; the meaning is a property of the
container. API changes (`src/spatialdata/models/models.py`):

- `TableModel.parse(adata)` takes no annotation arguments. `region`/
  `region_key`/`instance_key` are still accepted with a `DeprecationWarning`:
  they are stored temporarily in `uns` and turned into columns + edges when
  the table enters a `SpatialData` (`SpatialData.__init__`, `sdata["t"] = ...`).
- `get_table_keys()` is kept for backward compatibility: it still returns
  the triple for tables in the old format (and for tables converted back with
  `convert_table_v03_to_v02`) and raises a clear error on V03 tables, pointing
  to `sdata.links`. `join_spatialelement_table`, `match_*`,
  `filter_by_table_query`, `validate_table_in_spatialdata` read the table's
  edges from `sdata.links` (they already receive `sdata`) — one code path
  instead of region filtering + instance matching.
- `src/spatialdata/_io/format.py`: `TablesFormatV03(FormatV05)`, version
  `"0.3"`; `src/spatialdata/_io/io_table.py` no longer writes `region`/
  `region_key`/`instance_key` group attributes.

Conversion (`src/spatialdata/models/_table_conversion.py`):

- `convert_table_v02_to_v03(table) -> (table, edges)`: one region → the
  `instance_key` column is renamed to the element name; several regions → one
  nullable column per region (`region` name, `Int64`/`string`), filled from
  `instance_key` where `region_key == region`; the `region_key` and
  `instance_key` columns are **removed** so no duplicated information remains;
  `uns` reduced to the version; one edge per region is returned.
- `convert_table_v03_to_v02(table, edges)`: inverse — rebuilds `region_key`
  and `instance_key` columns and the old `uns` triple from the table's edges
  (a row annotating two elements cannot be represented and raises).
- `read_zarr` converts V02 tables automatically in memory and appends the
  resulting edges to `attrs["spatialdata_link"]`; writing back produces V03.
  No user call is needed.

### Encoding: container protocol

```python
class InstanceCollection(Protocol):
    ids: pd.Index                       # identity values, one per instance (unique; see decision 8); len(ids) = n
    def column(self, name: str) -> pd.Series      # join-key values aligned to ids ("index"/"value"/column)
    def subset(self, ids: pd.Index) -> Element     # same element type, only instances in ids

# examples
TableAdapter(adata, identity="index")             # ids = obs_names;      column("cell_id") = obs["cell_id"]
ShapesAdapter(gdf, identity="index")              # ids = gdf.index;      column("roi") = gdf["roi"]
PointsAdapter(ddf, identity="transcript_id")      # ids = a stable column (recommended); column("cell_id") lazy
LabelsAdapter(labels, identity="value")           # ids = unique non-zero pixel values (cached)
ElementAdapter("he_image")                        # ids = Index(["he_image"]): the element is one instance
```

Adapters live in `src/spatialdata/_core/links/adapters.py`. Subsetting reuses
`_get_masked_element` (points/shapes; `src/spatialdata/_core/query/relational_query.py`
~line 210), `filter_label_pixels` logic from `join_spatialelement_table` (labels),
and boolean masks (tables). `get_element_instances` (same module ~line 65)
provides label ids. `ElementAdapter` backs instance ↔ element links: an
element is a single instance whose id is its name, and `subset` returns the
whole element or nothing. Duplicate identity values: warn, keep the first row.

Column naming. `sdata.links.index_element(name, column="_sd_index")` writes a
stable identity column (0..n−1 in the current row order) when an element has
no usable identity; the default name can be changed and an existing column is
never overwritten (error unless `overwrite=True`). A materialized spatial join
(`materialize(edge, column=None)`) writes to a column **named after the target
element** by default — the same convention as table columns — so an element
involved in several spatial joins accumulates one column per target
(`cell_boundaries["rois"]`, `cell_boundaries["tiles"]`); a name clash raises
unless `column=` is given.

### Execution: backends

**`native` (default).** Build the element-level graph from the edges (a few
nodes). For a query `(element, ids)`: BFS over the element graph up to `depth`
and honouring `direction`; at each hop compute the matched ids on the
neighbour with a pandas `isin`/merge on the two columns (`sjoin` for suggested
edges, in the target coordinate system, reusing the `by.sjoin(values)` logic of
`_aggregate_shapes`, `src/spatialdata/_core/operations/aggregate.py` ~lines
334 and 446; pixel lookup for labels). 1:1 groups are a single `isin`. Nothing
needs an instance-level matrix; memory is bounded by the columns touched.

**`mulink`** (extra `links-mulink`). `sdata.links.to_mudata()` builds a private
`MuData`: one modality per element (`InstanceCollection` values once the mudata
fork accepts them; real `AnnData` for tables), `obs_names` prefixed with the
element name, `obsmap` = offsets, `obsp["links"]` = sparse adjacency built from
the edges. Queries use `mdata.link.query.descendants/ancestors(axis="obs")`
from the mulink fork. Useful when the instance graph itself is wanted (export,
graph algorithms, MuData tooling), and as a bridge to mulink users.

**`duckdb`** (extra `links-duckdb`). Each element's identity and join columns
are registered as zero-copy views (`conn.register(df)`; parquet views for
points). Each edge becomes a junction view (`SELECT l.id, r.id FROM ... JOIN
...`). `check()` is SQL (`COUNT(*) WHERE id NOT IN (...)`, cardinality via
`GROUP BY`); a multi-hop query is one SQL join along the element path; the
schema exports an ER diagram from `information_schema` (graphviz). Attractive
for very wide/long tables, for users who already think in SQL, and as a
cross-`SpatialData` execution engine later.

Selection: `sdata.links.backend = "native" | "mulink" | "duckdb"` (per
`SpatialData` object, default `"native"`); the schema and encoding are
identical for all three, and the acceptance tests run on each.

A note on n:m. The mulink backend (a sparse matrix) and the DuckDB backend (a
junction view) could execute arbitrary n:m relationships, but the schema and
encoding of this RFC cannot _store_ one: a column holds one target per row.
n:m relationships therefore appear only as the result of a `suggested`
spatial join (computed at query time, never materialized). A stored n:m
encoding is a possible future update.

### MuData and mulink: assessment

|                  | mulink                                                                                                               | MuData                                                                                                                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What it gives    | ~200 lines: BFS ancestors/descendants over a CSR adjacency, modality filtering, a `.link` accessor pattern, plotting | The container: aligned global index, `obsmap`/`varmap` offsets, `pull_obs`/`push_obs`, `update()`, `.h5mu`/zarr IO, ecosystem tooling                                                                     |
| Fit for this RFC | Good for traversal; needs an obs axis (small PR); its per-vertex BFS loop needs batching                             | Hosts the instance graph naturally _if_ modalities could be non-AnnData and obs/var could both be separate; today neither holds                                                                           |
| Cost             | Optional; pure functions importable                                                                                  | Brings its own object model: `update()`/name handling assume `AnnData`; a 10M-row global `obs` for transcripts is heavy                                                                                   |
| Verdict          | Use as an optional backend and for interop; do not make the core depend on it                                        | Target backend for interop; not a core dependency. The protocol and an `axis=-2` mode ("neither observations nor variables shared", see _Background and constraints_) are the two asks to bring to MuData |

The native backend exists so that neither library is required to link
spatialdata elements; the mulink backend exists so that the same schema can be
handed to the MuData ecosystem without translation.

The JSON edge description is independent of spatialdata. mulink could adopt it
to describe _its_ relationships (stored in `mdata.uns`, next to the `varp`/
`obsp` matrix that encodes them), which would give scverse one way of stating
"these two axes are linked through these keys" regardless of whether the
container is a `SpatialData` or a `MuData`, and regardless of which backend
executes the query.

### Container format V03

`src/spatialdata/_io/format.py`:

- `SpatialDataContainerFormatV03(FormatV05)`, version `"0.3"`;
  `attrs_to_dict()` as V02.
- New `validate_attrs(attrs)` declared next to the other validation methods in
  the format classes: no-op for V01/V02; V03 validates `attrs["spatialdata_link"]`
  against the edge schema (reuse `Edge.from_dict`). Called from
  `SpatialData.write_attrs` before `zarr_group.attrs.put`, and by
  `sdata.validate()`.
- Registries: `CurrentSpatialDataContainerFormat = V03`; extend
  `SpatialDataContainerFormatType`, `SpatialDataContainerFormats["0.3"]`,
  `ContainerFormatValidElements[V03]` (V02's element formats plus
  `TablesFormatV03`); `src/spatialdata/_io/io_zarr.py` (~line 181) accepts
  `"0.3"`. A V02 store is read and converted in memory (tables V02→V03);
  a V03 store with a malformed key warns and keeps the raw JSON.

### Accessor and query

`sdata.links` (`src/spatialdata/_core/links/accessor.py`): `add(...)`,
`add_group(..., ordered=False)`, `remove(...)`, `check()`, `materialize(...)`,
`graph()` → `networkx.DiGraph`, `to_mermaid()`/`to_dot()`, `pl.graph()`,
`to_mudata()`, `backend`, `index_element(...)`.

`sdata.subset_by(element, ids, *, depth="all", direction=None,
keep_unlinked=False, filter_label_pixels=None) -> SpatialData`: resolves
`(element, ids)` through the active backend, subsets each reached element via
its adapter, filters tables annotating kept elements with
`_filter_table_by_elements` (`relational_query.py` ~line 130), and returns
`SpatialData(..., attrs=deepcopy(sdata.attrs))`. It never depends on mudata or
mulink. `SpatialData.subset()` (by element names) is unchanged.

## Status

Implemented so far, as stacked pull requests on the fork (each one based on
the previous, the first one on `rfc-link`, which carries this document):

| PR                                                          | Branch                           | What landed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [#10](https://github.com/LucaMarconato/spatialdata/pull/10) | `links/01-schema-container-v03`  | The **edge description** (`Edge`, `validate_link_spec`) and the **`sdata.links` view** over `attrs["spatialdata_link"]` (`add`, `remove`, `for_element`, `validate`, `to_dicts`, `repr`), `sdata.validate()`, and **container format 0.3**, which reserves the key: validated on write, tolerated with a warning when a stored value is malformed; 0.2 stores treat the key as plain attrs.                                                                                                                                                                                                                                                                                    |
| [#11](https://github.com/LucaMarconato/spatialdata/pull/11) | `links/02-table-format-v03`      | **Table format 0.3** (`TablesFormatV03`): the table metadata holds only the version, one column of instance ids per annotated element. The two **converters** (`convert_table_v02_to_v03` returning the table and its edges, `convert_table_v03_to_v02` the inverse), reading and writing of 0.3 tables, and `TableModel.has_region_annotation()` so every reader of the legacy triple keeps working and fails clearly on 0.3 tables.                                                                                                                                                                                                                                          |
| [#12](https://github.com/LucaMarconato/spatialdata/pull/12) | `links/03-table-links-migration` | The **compatibility layer** `_core/links/tables.py` (`resolve_instance_ids`, `annotated_elements`, `table_instance_keys[_any_format]`, `get_table_links`, `right_key_for_element`, `convert_legacy_table`) and the migration of every consumer of the legacy triple to it: the five joins and `join_spatialelement_table` (which gains `instance_keys=`), `match_sdata_to_table`, `get_element_annotators`, `get_values`, `_filter_table_by_elements`, `filter_by_coordinate_system`, `concatenate` (links carried over, suffixes applied), `rasterize_bins`, the tiles dataloader. `SpatialData.annotated_elements()` and a links-aware `set_table_annotates_spatialelement`. |

Everything above is backwards compatible: `CurrentTablesFormat` is still 0.2,
a table keeps its `region`/`region_key`/`instance_key` metadata unless
`convert_table_v02_to_v03` (or `convert_legacy_table`) is called on it, and
the existing test suite passes unchanged.

### Next

1. **Flip the default to table format 0.3** (`links/04-table-format-default`).
   `CurrentTablesFormat = TablesFormatV03`; convert legacy tables
   automatically on `read_zarr` and when a table is added to a `SpatialData`
   object (the hook is a one-line `on_set` on `Tables`, prototyped and then
   reverted in PR 3); `TableModel.parse(region=…, region_key=…,
instance_key=…)` keeps working with a `DeprecationWarning`. This is the
   first **user-visible** change of the series — `obs["region"]` and
   `obs["instance_id"]` are replaced by one column per annotated element — so
   it also carries the migration of the fixtures and of the ~186 assertions
   on the legacy columns across 13 test files (`tests/conftest.py`,
   `tests/core/query/*`, `tests/core/operations/*`, `tests/io/test_multi_table.py`,
   `tests/models/test_models.py`, …). Keep the exclusive-join and
   `match_rows` semantics under test while doing it.
2. `links/05-protocol-adapters` — `InstanceCollection`, the per-element
   adapters and `ElementAdapter` (instance ↔ element links).
3. `links/06-native-backend` — the element graph and the native traversal.
4. `links/07-accessor-subset-by` — the rest of `sdata.links` (`add_group`,
   `check`, `graph`/`to_mermaid`, `index_element`), `subset_by`, and the
   deep copy of `attrs` on subsetting.
5. `links/08-spatial-join` — suggested edges, `sjoin`/pixel lookup,
   `materialize`.
6. `links/09-mulink-backend` — `to_mudata()` and the mulink backend
   (tables-only until mudata accepts non-AnnData modalities).
7. `links/10-duckdb-backend`.

Known gaps to close along the way: `SpatialData.subset()`/`query()` still
pass `attrs` by reference (the links are therefore shared with the subset);
`get_annotated_regions` raises on 0.3 tables and callers should move to
`sdata.annotated_elements()`; `concatenate` refuses to mix the two formats
in one table.

## Development workflow

Personal forks, stacked pull requests against each fork's `main`; nothing is
opened upstream. PR 1 targets `rfc-link`; each following PR targets the
previous branch. The `Edge` dataclass in `_core/relationships.py` of
[LucaMarconato/spatialdata#3](https://github.com/LucaMarconato/spatialdata/pull/3)
(branch `vibecoded-experiment/sp-ops-relationships`) is the reference shape;
the `vibecoded-experiment/*` stack itself is not a base.

- mulink (`https://github.com/LucaMarconato/mulink`): `obs-axis/01-query-axis`
  (`QueryAccessor(axis=...)` over `obs_names/obsp/obsmap`),
  `obs-axis/02-batched-reachability` (one BFS from a virtual source instead of
  one per query vertex).
- mudata (fork of `scverse/mudata`, item 9, needed for the full mulink backend
  with non-AnnData modalities): accept `InstanceCollection`-like objects in
  `.mod` (`obs`, `var`, `obs_names`, `var_names`, `n_obs`, `n_vars`, `shape`,
  `__getitem__`) and an `axis=-2` mode.

## Verification

- Unit: table V02↔V03 converters (single region, several regions, a row
  annotating two elements raises on the way back); `TableModel.parse`
  deprecation path (legacy arguments become columns + edges when the table
  enters a `SpatialData`); container V03 registry and `validate_attrs`; adapters per
  element type (duplicate identities warn and keep the first row; `DataTree`
  labels); `Edge` roundtrip; native traversal on toy graphs; 1:1 groups with
  matching and non-matching order (`check()` reports both).
- Invariant: for a converted legacy table, `subset_by(table, ids)` equals the
  pre-change `join_spatialelement_table(..., how="inner")` restricted to `ids`.
- Roundtrip: `write` → `read_zarr` preserves `attrs["spatialdata_link"]` and
  writes `spatialdata_attrs.version == "0.3"`; a V02 store reads and converts.
- Acceptance: the Appendix use cases as integration tests on a fixture built
  from `spatialdata.datasets.cells()` (94 cells; images `he_aligned`,
  `he_image`, `morphology_focus`; labels `cell_labels`, `nucleus_labels`,
  `tissue_labels`; points `transcripts`; shapes `cell_boundaries`,
  `nucleus_boundaries`; `table`) **plus components added by the fixture** to
  exercise the whole machinery:
    - `rois` (two large polygons): each cell falls within at most one ROI — a
      **containment** relationship (`predicate="within"`, n:1), which can be
      computed by `sjoin` and stored as a column;
    - `tiles` (a regular grid whose tiles straddle cell boundaries): a cell can
      intersect several tiles — an **overlap** relationship
      (`predicate="intersects"`, n:m), which `sjoin` computes at query time but
      which the column encoding cannot store (`materialize` must refuse it);
    - a second segmentation, `cell_labels` shifted by a few pixels: two labels
      elements whose ids do not correspond, linked by a spatial join declared
      with `status="suggested"`, i.e. written into the schema _before_ any
      join is computed, so that the lazy resolution and caching path is tested;
    - a two-sample table with `NA` columns; an `image_metadata` table with one
      row per image (instance ↔ element links); a points element with a
      non-unique index (duplicate-identity warning).
      Run against each available backend.
- Performance smoke test: 1M-point `n:1` edge, subset 10k cells, seconds not
  minutes, on the native backend.

## References

- mulink: https://github.com/lucas-diedrich/mulink (`src/mulink/query.py`,
  `src/mulink/basic.py`; design notes in `docs/rfc.md`)
- MuData feature-mapping discussion: https://github.com/scverse/mudata/issues/111
- scverse Padua hackathon issue on element relationships:
  https://github.com/scverse/2026_04_hackathon_padua/issues/6
- Element-relationship prototype from the Bioconductor Venice hackathon
  (throwaway, useful for API ergonomics only):
  https://github.com/HelenaLC/VeniceInterop/pull/2
- Edge-list prototype in the spatialdata fork:
  https://github.com/LucaMarconato/spatialdata/pull/3
- DuckDB experiments (zero-copy registration of `obs`, parquet views, junction
  tables, orphan checks in SQL, ER diagram from `information_schema`):
  https://github.com/LucaMarconato/spatialdata/blob/duckdb/scripts/hackathon_duckdb.py
- Forks used for development: https://github.com/LucaMarconato/spatialdata,
  https://github.com/LucaMarconato/mulink
- spatialdata code reused: `get_element_instances`, `_get_masked_element`,
  `_filter_table_by_elements`, `join_spatialelement_table`,
  `filter_by_table_query` (`src/spatialdata/_core/query/relational_query.py`);
  `TableModel` (`src/spatialdata/models/models.py`); `_aggregate_shapes`
  (`src/spatialdata/_core/operations/aggregate.py`); formats
  (`src/spatialdata/_io/format.py`, `io_table.py`, `io_zarr.py`);
  `spatialdata.datasets.cells`

---

## Appendix — mock notebook (not runnable; outputs simulated)

All examples use `spatialdata.datasets.cells()`, a small Xenium Prime region
with 94 cells. Numbers other than the cell count are illustrative.

```python
import spatialdata as sd
sdata = sd.datasets.cells()
sdata
```

```
SpatialData object
├── Images
│     ├── 'he_aligned': DataTree[cyx] (3, 2048, 2048), (3, 1024, 1024), ...
│     ├── 'he_image': DataTree[cyx] ...
│     └── 'morphology_focus': DataTree[cyx] ...
├── Labels
│     ├── 'cell_labels': DataTree[yx] (2048, 2048), ...
│     ├── 'nucleus_labels': DataTree[yx] ...
│     └── 'tissue_labels': DataTree[yx] ...
├── Points
│     └── 'transcripts': DataFrame with shape: (<Delayed>, 5) (2D points)
├── Shapes
│     ├── 'cell_boundaries': GeoDataFrame shape: (94, 1) (2D shapes)
│     └── 'nucleus_boundaries': GeoDataFrame shape: (91, 1) (2D shapes)
└── Tables
      └── 'table': AnnData (94, 5001)
with coordinate systems:
    ▸ 'global', with elements: he_aligned (Images), ... transcripts (Points), cell_boundaries (Shapes), ...
```

### UC1 — reading a store written before this RFC

The dataset was written with table format 0.2 (`region="cell_labels"`,
`region_key="region"`, `instance_key="cell_id"`). On read the table is converted
to format 0.3 and the corresponding edge appears; nothing has to be called.

```python
sdata["table"].obs.head(2)
```

```
              cell_labels  transcript_counts  ...
cell_00001          1                 312  ...
cell_00002          2                 287  ...
```

```python
sdata["table"].uns["spatialdata_attrs"]
```

```
{'version': '0.3'}
```

```python
sdata.links
```

```
Links: 1 edge over 2 elements  (backend: native)
  [0] table ──(cell_labels ↔ value)──▶ cell_labels        join  computed  1:1
```

The conversion renamed the `cell_id` column to `cell_labels`, dropped the
`region` column, and appended this edge to `attrs["spatialdata_link"]`. A
freshly constructed `SpatialData` with no legacy metadata and no edges shows
`Links: 0 edges` — relationships are never inferred from names or shapes.

### UC2 — declaring id-sharing elements, and looking at the graph

```python
sdata.links.add_group(
    ["cell_boundaries", "nucleus_boundaries", ("cell_labels", "value"), ("nucleus_labels", "value"),
     ("table", "cell_labels")],
    ordered=False,            # 1:1 declared; matching row order is *not* promised
)
sdata.links
```

```
Links: 5 edges over 5 elements  (backend: native)
  [0] table ──(cell_labels ↔ value)──▶ cell_labels               join  computed  1:1
  [1] cell_boundaries ──(index ↔ index)──▶ nucleus_boundaries    join  computed  1:1
  [2] cell_boundaries ──(index ↔ value)──▶ cell_labels           join  computed  1:1
  [3] cell_boundaries ──(index ↔ value)──▶ nucleus_labels        join  computed  1:1
  [4] cell_boundaries ──(index ↔ cell_labels)──▶ table           join  computed  1:1
```

```python
print(sdata.links.to_mermaid())
```

```
graph LR
  table -- "cell_labels ↔ value (1:1)" --> cell_labels
  cell_boundaries -- "index ↔ index (1:1)" --> nucleus_boundaries
  cell_boundaries -- "index ↔ value (1:1)" --> cell_labels
  cell_boundaries -- "index ↔ value (1:1)" --> nucleus_labels
  cell_boundaries -- "index ↔ cell_labels (1:1)" --> table
```

```python
sdata.links.pl.graph()          # matplotlib/networkx drawing; sdata.links.to_dot() for graphviz
```

### UC3 — cheap validation before trusting the graph

```python
sdata.links.check()
```

```
edge                                     declared  observed  coverage from→to / to→from  order    notes
[0] table → cell_labels                  1:1       1:1       100.0% / 100.0%             n/a
[1] cell_boundaries → nucleus_boundaries 1:1       1:1*       96.8% / 100.0%             differs  3 cells without nucleus (*partial)
[2] cell_boundaries → cell_labels        1:1       1:1       100.0% / 100.0%             n/a
[3] cell_boundaries → nucleus_labels     1:1       1:1*       96.8% / 100.0%             n/a
[4] cell_boundaries → table              1:1       1:1       100.0% / 100.0%             same
```

`check()` compares id sets only (never geometry or `X`). It reports observed
cardinality, coverage in both directions, whether row order matches when it
can be compared, `<NA>` counts for table columns, and duplicate identities
(`transcripts: 148 202 rows, index unique`). A group declared with
`ordered=True` whose order differs produces a warning here.

### UC4 — transcripts in cells: a foreign-key column already on the points

```python
sdata["transcripts"].head(3)
```

```
         x        y  feature_name  cell_id   qv
0  1021.3    455.8         EPCAM        1  40.0
1  1022.9    460.1          KRT8        1  38.2
2  1198.0    512.6          CD3E        0  21.7      # 0 == unassigned
```

```python
sdata.links.add("transcripts", "cell_labels", on=("cell_id", "value"), cardinality="n:1")
```

Two motivating queries — a subset of cells with everything linked to them, and
the transcripts of those cells:

```python
sub = sdata.subset_by("table", ids=["cell_00001", "cell_00002", "cell_00017"])
sub
```

```
SpatialData object
├── Labels
│     ├── 'cell_labels': DataTree[yx] ...          # pixels ∉ ids set to 0 (filter_label_pixels=True)
│     └── 'nucleus_labels': DataTree[yx] ...
├── Points
│     └── 'transcripts': DataFrame with shape: (<Delayed>, 5) (2D points)   # 4 731 rows
├── Shapes
│     ├── 'cell_boundaries': GeoDataFrame shape: (3, 1) (2D shapes)
│     └── 'nucleus_boundaries': GeoDataFrame shape: (3, 1) (2D shapes)
└── Tables
      └── 'table': AnnData (3, 5001)
Dropped (no link path from 'table'): he_aligned, he_image, morphology_focus (Images), tissue_labels (Labels)
```

```python
sub["transcripts"].compute()["cell_id"].unique(), sub["table"].obs_names.tolist()
```

```
(array([1, 2, 17]), ['cell_00001', 'cell_00002', 'cell_00017'])
```

### UC5 — cells in ROIs: a spatial join declared now, computed on demand

```python
rois = gpd.GeoDataFrame({"name": ["tumor", "stroma"]}, geometry=[Polygon(...), Polygon(...)])
sdata["rois"] = sd.models.ShapesModel.parse(rois)
sdata.links.add("cell_boundaries", "rois", method="sjoin", predicate="within")
sdata.links[-1]
```

```
Edge(from='cell_boundaries', to='rois', method='sjoin', on={'predicate': 'within'}, status='suggested')
```

The element graph now has this shape (arrows point from finer to coarser):

```
transcripts ──n:1──▶ cell_labels ◀──1:1──▶ cell_boundaries ──sjoin (suggested)──▶ rois
                          ▲                     │   ▲
                          │1:1                  │1:1│1:1
                        table              nucleus_boundaries / nucleus_labels
```

```python
tumor = sdata.subset_by("rois", ids=[0])
tumor
```

```
Resolving suggested edge cell_boundaries → rois (sjoin, within) in coordinate system 'global' ... 0.1s
SpatialData object
├── Labels        'cell_labels', 'nucleus_labels' (pixel-filtered)
├── Points        'transcripts': (<Delayed>, 5)              # via cell_labels ← cell_boundaries ← rois
├── Shapes        'cell_boundaries' (41), 'nucleus_boundaries' (40), 'rois' (1)
└── Tables        'table': AnnData (41, 5001)
```

Traversal: `rois → cell_boundaries → {cell_labels → transcripts, table,
nucleus_*}`, three hops, without naming any of them. The same containment
could be declared against `tissue_labels` (a labels element) with
`method="sjoin"`; the native backend then uses a pixel lookup instead of
geopandas.

```python
sdata.links.materialize(-1, column="roi")     # write the result as a column; the edge becomes a plain join
sdata["cell_boundaries"].head(2)
```

```
                 geometry  roi
1   POLYGON ((...))          0
2   POLYGON ((...))          0
```

```python
sdata.links[-1]
```

```
Edge(from='cell_boundaries', to='rois', method='join', on={'left': 'roi', 'right': 'index'}, status='computed', cardinality='n:1')
```

### UC6 — direction and depth (advanced)

```python
# only what these cells *contain* (finer): transcripts, nuclei — not the ROI
sdata.subset_by("table", ids=["cell_00001", "cell_00002"], direction="down")
# only coarser context: the ROI(s) these cells fall in
sdata.subset_by("table", ids=["cell_00001", "cell_00002"], direction="up", depth=2)
```

```
Links.query('table', 2 ids, direction='up', depth=2) → {'cell_labels': 2, 'cell_boundaries': 2, 'rois': 1}
```

### UC7 — one table annotating several samples, and converting formats

```python
sdata_multi["cells"].obs[["s1_cells", "s2_cells"]].head(3)
```

```
   s1_cells  s2_cells
0         1      <NA>
1         2      <NA>
2      <NA>         1
```

```python
sdata_multi.links
```

```
Links: 2 edges over 3 elements  (backend: native)
  [0] cells ──(s1_cells ↔ index)──▶ s1_cells        join  computed  1:1
  [1] cells ──(s2_cells ↔ index)──▶ s2_cells        join  computed  1:1
```

```python
sdata_multi.subset_by("s2_cells", ids=[1]).tables["cells"].obs
```

```
   s1_cells  s2_cells
2      <NA>         1
```

Rows with `<NA>` in a column take no part in that column's edge. A legacy
table annotating the same two elements converts and converts back:

```python
from spatialdata.models import convert_table_v02_to_v03, convert_table_v03_to_v02
old = legacy_table.obs.head(3); old
```

```
   region    cell_id
0  s1_cells        1
1  s1_cells        2
2  s2_cells        1
```

```python
new, edges = convert_table_v02_to_v03(legacy_table)   # region_key/instance_key columns removed, one column per region
convert_table_v03_to_v02(new, edges).obs.equals(legacy_table.obs)
```

```
True
```

### UC8 — persistence, validation, container format 0.3

```python
sdata.write("cells_linked.zarr")
back = sd.read_zarr("cells_linked.zarr")
back.attrs["spatialdata_link"][:2]
```

```
[{'from': 'table', 'to': 'cell_labels', 'method': 'join',
  'on': {'left': 'cell_labels', 'right': 'value'}, 'status': 'computed', 'cardinality': '1:1'},
 {'from': 'cell_boundaries', 'to': 'nucleus_boundaries', 'method': 'join',
  'on': {'left': 'index', 'right': 'index'}, 'status': 'computed', 'cardinality': '1:1'}]
```

Every link, including the table's, is in this one list.

```python
zarr.open_group("cells_linked.zarr", mode="r").attrs["spatialdata_attrs"]["version"]
```

```
'0.3'
```

```python
sdata.attrs["spatialdata_link"].append({"from": "x"})   # malformed
sdata.validate()
```

```
ValueError: attrs['spatialdata_link'] is not a valid link specification (edge 5: missing keys 'to', 'method', 'on')
```

```python
sdata.links.add("cell_boundaries", "does_not_exist", on=("index", "index"))
```

```
ValueError: element 'does_not_exist' not found (pass validate=False to add the edge anyway)
```

### UC9 — the same schema on other backends

```python
sdata.links.backend = "duckdb"          # requires spatialdata[links-duckdb]
sdata.links.check()                     # same table as UC3, computed as SQL
sdata.links.sql("SELECT COUNT(*) FROM transcripts t JOIN cell_labels__table j ON t.cell_id = j.cell_labels WHERE j.table IN ('cell_00001')")
```

```
┌──────────┐
│ count(*) │
│  1 566   │
└──────────┘
```

```python
sdata.links.er_diagram("schema.png")    # graphviz ER diagram from information_schema
```

```python
sdata.links.backend = "mulink"          # requires spatialdata[links-mulink] (forks of mulink and mudata)
mdata = sdata.links.to_mudata()
mdata
```

```
MuData object with n_obs × n_vars = 148 673 × 5001   axis=0 (disjoint obs_names)
  obsmap: 'table', 'cell_boundaries', 'nucleus_boundaries', 'cell_labels', 'nucleus_labels', 'transcripts', 'rois'
  obsp:   'links'
  7 modalities
    table:              94 × 5001
    cell_boundaries:    94 × 0        (InstanceCollection[GeoDataFrame])
    transcripts:   148 202 × 0        (InstanceCollection[DaskDataFrame])
    ...
```

```python
mdata.link.query.descendants(["table-cell_00001", "table-cell_00002"], key="links", axis="obs",
                             include_mods=["transcripts"]).obsmap["transcripts"].sum()
```

```
3 131
```

Same rows as `sdata.subset_by("table", ids=[...], direction="down")["transcripts"]`.
`subset_by` on the default backend never touches mudata or mulink; the
`mulink` backend is for users who want the instance graph as a MuData object.

### UC10 — a table with one row per image (instance ↔ element)

```python
meta = ad.AnnData(obs=pd.DataFrame({"image": ["he_aligned", "he_image", "morphology_focus"],
                                    "stain": ["H&E", "H&E", "DAPI"], "microns_per_pixel": [0.5, 0.5, 0.2125]}))
sdata["image_metadata"] = sd.models.TableModel.parse(meta)
sdata.links.add("image_metadata", ["he_aligned", "he_image", "morphology_focus"], on=("image", "element"))
sdata.links[-1]
```

```
Edge(from='image_metadata', to=['he_aligned', 'he_image', 'morphology_focus'], method='join',
     on={'left': 'image', 'right': 'element'}, status='computed', cardinality='1:1')
```

```python
sdata.subset_by("image_metadata", ids=sdata["image_metadata"].obs.query("stain == 'H&E'").index)
```

```
SpatialData object
├── Images
│     ├── 'he_aligned': DataTree[cyx] ...
│     └── 'he_image': DataTree[cyx] ...
└── Tables
      └── 'image_metadata': AnnData (2, 0)
Dropped (no link path from 'image_metadata'): morphology_focus (Images), cell_labels, ... (Labels), transcripts (Points), ...
```

Each target element is a single instance identified by its name; the whole
element is kept or dropped. The same works with `to="tables"` (all tables) to
annotate tables with a table.

### Not in this RFC (direction only)

```python
# stored n:m overlap between two segmentations — needs an array-backed encoding, not columns
sdata.links.add("cells_v1", "cells_v2", method="overlap", min_fraction=0.5)
# image channels ↔ table vars — mulink's native var-axis case, same accessor
sdata.links.add("morphology_focus", "table", on=("c", "var_names"), axis="var")
```
