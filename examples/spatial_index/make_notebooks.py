"""Generate the two example notebooks of the spatial index (RFC 3 points, RFC 4 shapes); run them with nbconvert."""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


points = [
    md("""
# Tiled and multiscale points (RFC 3): indexing an existing SpatialData object

This notebook starts from a SpatialData object as it is distributed today (the 10x Atera kidney dataset of
[spatialdata-db](https://www.spatialdata-db.com/collections/10x-atera-whole-transcriptome/), 1.19 billion transcripts)
and writes its points again with the spatial index of
[RFC 3](https://spatialdata-spec.scverse.org/rfcs/0003-tiled-points.html): a random priority $p$ per point, levels
as bands, a tile-major table and a feature-major table, and their offsets. It then reads them back by region, level,
budget and gene, and compares with reading without an index.

The API is experimental (draft PR on `LucaMarconato/spatialdata`, branch `tiled-points-spatial-index`).
"""),
    code("""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spatialdata as sd
from spatialdata._io.spatial_index import PointsSpatialIndex, write_points_with_spatial_index

DATA = Path("/Volumes/Scratch/tiled-index-data")
SOURCE = DATA / "spatialdata_db_atera_kidney.zarr"  # points, shapes and table downloaded from spatialdata-db
OUT = DATA / "kidney.zarr"
"""),
    md("## The SpatialData object as distributed"),
    code("""
sdata = sd.read_zarr(SOURCE, selection=("points", "shapes", "tables"))  # images were not downloaded
sdata
"""),
    code("""
points = sdata["transcripts"]
print(points.npartitions, "partitions")
points.dtypes
"""),
    md("""
The feature column (`feature_name`) is a plain string column here; the builder turns it into a categorical with a
global category list (RFC 3: "integer codes with the category list stored once").

## A new store with the other elements

The index is written as a new points element; here we put it in a new store, next to the shapes and the table, and
leave the downloaded dataset untouched. `sd.build_points_spatial_index(sdata, name)` would instead rewrite the
element in place (or under `new_element_name`).
"""),
    code("""
if not OUT.exists():
    sd.SpatialData(shapes=dict(sdata.shapes), tables=dict(sdata.tables)).write(OUT)
"""),
    md("""
## Building the index

Both formats of RFC 3: Parquet (points format 0.3, the default) and Zarr (0.3-experimental), with the same layout.
The grid has all the axes of the points (x, y and a single tile along z); the tile size is chosen so that the
median non-empty tile of level 0 holds about 10,000 points; the levels follow the merge rule and the default rates;
p is stored as 16-bit integers; Parquet row groups hold several tiles (at least 250,000 rows).

The builder works out of core: it reads the points partition by partition, scatters them to temporary buckets and
sorts each bucket in memory.
"""),
    code("""
stats_by_storage = {}
for storage in ["parquet", "zarr"]:
    name = f"transcripts_{storage}"
    path = OUT / "points" / name
    if not (path / "zarr.json").exists() or "spatial_index" not in (path / "zarr.json").read_text():
        t0 = time.time()
        md, stats = write_points_with_spatial_index(
            points, path, storage=storage, tmp_dir=DATA / "tmp", n_workers=6
        )
        stats_by_storage[storage] = stats
        print(storage, f"{time.time() - t0:.0f} s")
sd.SpatialData.read(OUT).write_consolidated_metadata()
"""),
    code("""
index = PointsSpatialIndex(OUT / "points" / "transcripts_parquet")
md = index.metadata
for l, lv in enumerate(md.levels):
    print(l, "tile", [round(s, 1) for s in lv.tile_size], "grid", lv.grid_shape, "P", round(lv.P, 6))
print("p_bits", md.p_bits)
"""),
    md("The element reads as a normal points element (tile-major order, with the extra column `__spatialdata_p__`):"),
    code("""
sdata_tiled = sd.read_zarr(OUT)
sdata_tiled["transcripts_parquet"].head()
"""),
    md("""
## Reading by region, level and budget

`query(box_min, box_max, level=...)` reads the bands $L, \\dots, l$ of the tiles in view. With a budget $b$ it reads
the points with $p < b$: a uniform sample, which is a prefix of each tile.
"""),
    code("""
lv0 = md.levels[0]
origin = np.array(lv0.origin)
extent = np.array(lv0.tile_size) * np.array(lv0.grid_shape)
center = origin + extent / 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, box, kw) in zip(axes, [
    ("whole slide, 200k points", (None, None), {"budget": 200_000 / index.n_rows}),
    ("1 mm, coarsest level that fits 200k", (center - 500, center + 500), {"budget": 200_000 / 3e7}),
    ("100 um, all the points", (center - 50, center + 50), {}),
]):
    t0 = time.time()
    df = index.query(*box, columns=["x", "y", "feature_name"], **kw)
    ax.scatter(df.x, df.y, s=0.2, c=df.feature_name.cat.codes, cmap="tab20", rasterized=True)
    ax.set_title(f"{title}\\n{len(df):,} points in {time.time() - t0:.2f} s")
    ax.set_aspect("equal")
plt.show()
"""),
    md("## Reading genes: feature-major vs tile-major"),
    code("""
counts = index.feature_ranges("feature_name", range(len(index.categories("feature_name"))))
s, e = counts
cats = index.categories("feature_name")
per_gene = (e - s).reshape(md.n_levels, len(cats)).sum(axis=0)
rare = cats[int(np.argsort(per_gene)[len(cats) // 10])]
common = cats[int(np.argmax(per_gene))]
print("rare:", rare, per_gene[cats.index(rare)], "common:", common, per_gene[cats.index(common)])
"""),
    code("""
for gene in [rare, common]:
    for use_feature_index in [True, False]:
        t0 = time.time()
        df = index.query(center - 1000, center + 1000, features=[gene], use_feature_index=use_feature_index,
                         columns=["x", "y", "feature_name"])
        print(f"{gene:>12} feature-major={use_feature_index!s:5} {len(df):>9,} points {time.time() - t0:6.2f} s")
"""),
    md("""
## Without an index

For comparison, dask with Parquet filters on the original element (the row groups of the original are not sorted
spatially, so statistics prune little).
"""),
    code("""
import dask.dataframe as dd

src = SOURCE / "points" / "transcripts" / "points.parquet"
lo, hi = center - 50, center + 50
t0 = time.time()
n = len(dd.read_parquet(src, columns=["x", "y"], filters=[("x", ">=", lo[0]), ("x", "<=", hi[0]),
                                                        ("y", ">=", lo[1]), ("y", "<=", hi[1])]))
print(f"no index: {n:,} points in {time.time() - t0:.1f} s")
t0 = time.time()
n = len(index.query(lo, hi, columns=["x", "y"]))
print(f"index:    {n:,} points in {time.time() - t0:.2f} s")
"""),
    md("""
## Zarr storage

The same index with the points as a `df:` Zarr group (`spatial/points`, sharded), read the same way.
"""),
    code("""
zindex = PointsSpatialIndex(OUT / "points" / "transcripts_zarr")
t0 = time.time()
df = zindex.query(center - 500, center + 500, budget=0.05, columns=["x", "y", "feature_name"])
print(f"{len(df):,} points in {time.time() - t0:.2f} s")
"""),
    md("""
## In napari

With the napari-spatialdata branch `tiled-points-spatial-index`, the indexed element loads only the points in view,
sampled down to `config.POINT_THRESHOLD`, and reloads them when the camera moves:

```python
from napari_spatialdata import Interactive
Interactive(sd.read_zarr(OUT))
```
"""),
]

shapes = [
    md("""
# Tiled shapes and their table (RFC 4) on the Atera kidney dataset

[RFC 4](https://spatialdata-spec.scverse.org/rfcs/0004-tiled-shapes.html) sorts the cells by the tile of their
centroid, on a grid of their own nested with the grid of the points, writes the geometry as GeoArrow, and stores the expression matrix as
CSR and CSC. This notebook adds this index to the cells of the store made in the points notebook
(`kidney_tiled.zarr`), reorders the table to the order of the cells, and reads cells and gene values by region.
"""),
    code("""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import spatialdata as sd
from spatialdata._io.spatial_index import PointsSpatialIndex, ShapesSpatialIndex, build_shapes_spatial_index

DATA = Path("/Volumes/Scratch/tiled-index-data")
OUT = DATA / "kidney.zarr"
sdata = sd.read_zarr(OUT)
sdata
"""),
    code("""
table = sdata["table"]
print(table)
print("layers already present in spatialdata-db:", list(table.layers))
"""),
    md("""
spatialdata-db already ships a CSC copy (`layers["csc_counts"]`); RFC 4 names it `X_csc`. The builder writes
`X_csc` (and leaves `csc_counts`) and reorders the rows of the table to the order of the cells. Its grid is nested
with the grid of the transcripts: a power of 2 times their tile size, the smallest one where more than half of the
cells lie within the tile of their centroid (RFC 4, "When to use it").

RFC 4 applies to a table that annotates the shapes. In this dataset the table annotates the cell *labels*
(`region="cell_labels"`, `instance_key="label_id"`), and the boundaries are matched through the column `cell_id`:
making the table annotate the boundaries is up to the user, done here first.
"""),
    code("""
if "spatial_index" not in zarr.open_group(OUT / "shapes" / "cell_boundaries").attrs:
    # make the table annotate the cell boundaries (the user's responsibility, RFC 4)
    from spatialdata.models import TableModel

    t = sdata["table"]
    t.obs["region"] = pd.Categorical(["cell_boundaries"] * t.n_obs)
    t.uns[TableModel.ATTRS_KEY].update({"region": "cell_boundaries", "instance_key": "cell_id"})
    sdata.delete_element_from_disk("table")
    sdata.write_element("table")
    t0 = time.time()
    build_shapes_spatial_index(sdata, "cell_boundaries", table_name="table", nest_with_points="transcripts_parquet")
    print(f"{time.time() - t0:.0f} s")
index = ShapesSpatialIndex(OUT / "shapes" / "cell_boundaries", OUT / "tables" / "table")
print(index.metadata.levels[0], "margin", index.margin, "within tile", index.metadata.extra["fraction_within_tile"])
"""),
    md("""
The grid is the one of the transcripts, so one tile request can load both. Note the margin: the largest overhang of
a cell outside its tile is about 30 um, 4-5 tiles of 6.7 um, so every box query is grown by that much (RFC 4 takes
the maximum over all the shapes; one large cell sets it for all).
"""),
    code("""
pts = PointsSpatialIndex(OUT / "points" / "transcripts_parquet")
print(pts.metadata.levels[0].tile_size, index.metadata.levels[0].tile_size)
"""),
    md("## Cells in view, colored by a gene read from the CSC layer"),
    code("""
lv0 = index.metadata.levels[0]
center = np.array(lv0.origin) + np.array(lv0.tile_size) * np.array(lv0.grid_shape) / 2
lo, hi = center - 150, center + 150
gene = "C7"  # a gene expressed in kidney stroma
t0 = time.time()
cells = index.query(lo, hi)
t1 = time.time()
values = index.gene_values(gene, cells["__row__"].to_numpy())
t2 = time.time()
print(f"{len(cells)} cells in {t1 - t0:.2f} s, {gene} in {t2 - t1:.2f} s")
transcripts = pts.query(lo, hi, features=[gene], columns=["x", "y"])
ax = cells.plot(column=values, cmap="viridis", figsize=(7, 7), legend=True)
ax.scatter(transcripts.x, transcripts.y, s=1, c="red")
ax.set_title(f"{gene}: cells (color) and transcripts (red)")
plt.show()
"""),
    md("""
## Zoomed out

RFC 4 has a single level for now: zoomed out, a client shows fewer cells by other means (aggregation, or the
transcripts alone). napari-spatialdata loads the cells only when at most `config.TILED_SHAPES_MAX` are in view.
"""),
    md("""
## In napari

With the napari-spatialdata branch, the cell boundaries load by view when fewer than `config.TILED_SHAPES_MAX` cells
are in view, and `layer.metadata["_tiled_loader"].color_by_gene(gene)` colors them from the CSC layer.
"""),
]

for name, cells in [("tiled_points_atera.ipynb", points), ("tiled_shapes_atera.ipynb", shapes)]:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
