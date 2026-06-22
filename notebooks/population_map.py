# population_map.py
# ==================
# Produces three maps:
#   1. population_map.png            — population density + pharmacy locations
#   2. accessibility_map.png         — current walking accessibility + pharmacies
#   3. accessibility_optimised_map.png — optimised pharmacy locations
#
# Run: python population_map.py
# Requires: map_utils.py in the same directory

import numpy as np
import pandas as pd
import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import box
from map_utils import (
    PATHS, MUNICIPALITIES_EXTENDED,
    SCALEBAR_TRANSPARENT, add_municipal_labels, style_legend, set_map_bounds,
)

# ── 1. Load all data ──────────────────────────────────────────────────────────

# Population grid
pop_df = pd.read_csv(PATHS["csv_population"])
pop_gdf = gpd.GeoDataFrame(
    pop_df,
    geometry=gpd.points_from_xy(pop_df["lon"], pop_df["lat"]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

# Pharmacies — current and relocated
pharmacies_gdf = gpd.read_file(PATHS["shp_pharmacies"]).to_crs(epsg=3857)

df_relo = pd.read_csv(PATHS["csv_pharm_relo"])
pharmacies_relo = gpd.GeoDataFrame(
    df_relo,
    geometry=gpd.points_from_xy(df_relo["new_lon"], df_relo["new_lat"]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

# Study area boundary (boundary .shp has incorrect metadata — force to EPSG:4326)
gdf_boundary_raw = gpd.read_file(PATHS["shp_boundary"])
gdf_boundary_raw = gdf_boundary_raw.set_crs("EPSG:4326", allow_override=True)
gdf_3857 = gdf_boundary_raw.to_crs(epsg=3857)

# Derive a clean bounding box from population extent (+ 1500 m / 600 m buffer)
p_minx, p_miny, p_maxx, p_maxy = pop_gdf.total_bounds
study_box = box(p_minx - 1500, p_miny - 600, p_maxx + 1500, p_maxy + 600)
boundary_gdf = gpd.GeoDataFrame(geometry=[study_box], crs="EPSG:3857")

# Water bodies (coordinates are Swiss Grid / EPSG:2056 — override and reproject)
# water_national = gpd.read_file(PATHS["shp_water"])
# water_national = water_national.set_crs(epsg=2056, allow_override=True)
# water_gdf = gpd.clip(water_national.to_crs(epsg=3857), boundary_gdf)

print(f"Population points:  {len(pop_gdf)}")
print(f"Pharmacies:         {len(pharmacies_gdf)}")
print(f"Relocated:          {len(pharmacies_relo)}")
# print(f"Water features:     {len(water_gdf)}")

# Accessibility raster
ai_df = pd.read_csv(PATHS["csv_accessibility"])
ai_gdf = gpd.GeoDataFrame(
    ai_df,
    geometry=gpd.points_from_xy(ai_df["lon"], ai_df["lat"]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

# ── Shared constants ──────────────────────────────────────────────────────────
SIZE_MULTIPLIER = 14

BOUNDARY_STYLE = dict(
    facecolor="none", edgecolor="#555555", linewidth=1.0,
    linestyle="-", alpha=0.3, zorder=3,
)

ANNOTATION_BBOX = dict(boxstyle="square,pad=0.5", fc="white", ec="#cccccc", lw=0.5)
ANNOTATION_ARROW = dict(arrowstyle="->", color="#555555", lw=1, connectionstyle="arc3,rad=0")
ANNOTATION_TEXT = dict(fontsize=9, color="#333333", fontfamily="sans-serif", zorder=6)


def _pharmacy_legend(ax, pharm_bins, pharm_labels, anchor_y, marker_color="green"):
    """Build and add a pharmacy capacity legend to ax."""
    handles = [
        Line2D(
            [0], [0], marker="P", color="none",
            markerfacecolor=marker_color, markeredgecolor="none",
            markeredgewidth=0.2, markersize=upper, label=label,
        )
        for upper, label in zip(pharm_bins, pharm_labels)
    ]
    legend = ax.legend(
        handles=handles, title="Pharmacy Capacity",
        loc="upper right", bbox_to_anchor=(1.0, anchor_y),
        frameon=True, fontsize=9, title_fontsize=10, alignment="left",
    )
    style_legend(legend)
    return legend


def _finish_map(ax, crop_gdf):
    """Crop, basemap, scalebar, labels."""
    set_map_bounds(ax, crop_gdf)
    ax.margins(0)
    cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerNoLabels, alpha=1, zorder=1)
    ax.add_artist(ScaleBar(**SCALEBAR_TRANSPARENT))
    ax.set_axis_off()
    add_municipal_labels(ax, MUNICIPALITIES_EXTENDED)


# ── 2. Map 1: Population distribution + pharmacy locations ───────────────────
print("\n── Map 1: Population ──")

pop_classifier = mapclassify.NaturalBreaks(pop_gdf["population"], k=8)
cap_classifier  = mapclassify.NaturalBreaks(pharmacies_gdf["capacity"], k=3)
pharm_bins = cap_classifier.bins

# Build pharm range labels
pharm_labels, lower = [], pharmacies_gdf["capacity"].min()
for upper in pharm_bins:
    pharm_labels.append(f"{lower:,.0f} – {upper:,.0f}")
    lower = upper

fig, ax = plt.subplots(figsize=(12, 12))

pop_gdf.plot(
    ax=ax, column="population", scheme="NaturalBreaks", k=10,
    cmap="Reds", marker="s", markersize=24, edgecolor="darkred",
    linewidth=0.3, legend=False, alpha=0.85, zorder=2,
)
gdf_3857.plot(ax=ax, **BOUNDARY_STYLE)
pharmacies_gdf.plot(
    ax=ax, color="green", marker="P",
    markersize=pharmacies_gdf["capacity"] * SIZE_MULTIPLIER,
    edgecolor="white", linewidth=0.5, zorder=4,
)

_finish_map(ax, boundary_gdf)

# Population legend
pop_bins = pop_classifier.bins
labels, lower = [], pop_gdf["population"].min()
for upper in pop_bins:
    labels.append(f"{lower:,.0f} – {upper:,.0f}")
    lower = upper

colors = plt.cm.Reds(np.linspace(0.2, 1, len(labels)))
pop_legend = ax.legend(
    handles=[mpatches.Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(colors, labels)],
    title="Population distribution",
    loc="upper right", bbox_to_anchor=(1.0, 1.0),
    frameon=True, fontsize=9, title_fontsize=10, alignment="left",
)
style_legend(pop_legend)
ax.add_artist(pop_legend)

pharm_leg = _pharmacy_legend(ax, pharm_bins, pharm_labels, anchor_y=0.80)
ax.add_artist(pharm_leg)

# Horgen Drogerie annotation
ax.annotate(
    "Schinzenhof Apotheke Drogerie\nis not shown as it is not in the official dataset",
    xy=(957022.09, 5984627.56), xytext=(954100, 5983000),
    bbox=ANNOTATION_BBOX, arrowprops=ANNOTATION_ARROW,
    ha="center", va="center", **ANNOTATION_TEXT,
)

# Densification annotation (dual-arrow)
dense_text, dense_t1, dense_t2 = (949500, 5985400), (953200, 5988500), (955300, 5984800)
ax.annotate(
    "Densification potential lies within\ncurrently less populated residential areas",
    xy=dense_t1, xytext=dense_text, xycoords="data",
    bbox=ANNOTATION_BBOX, arrowprops=ANNOTATION_ARROW,
    ha="left", va="top", **ANNOTATION_TEXT,
)
ax.annotate("", xy=dense_t2, xytext=dense_text,
            arrowprops=ANNOTATION_ARROW, zorder=5)

plt.savefig(PATHS["out_population"], dpi=300, bbox_inches="tight")
print(f"Saved → {PATHS['out_population']}")
plt.show()


# ── 3. Map 2: Current accessibility distribution ─────────────────────────────
print("\n── Map 2: Accessibility (current) ──")

ai_gdf["Ai_scaled"] = ai_gdf["Ai_walk_correct"] * 1000
ai_classifier = mapclassify.NaturalBreaks(ai_gdf["Ai_scaled"], k=8)

# Reuse same pharmacy capacity legend
cap_classifier  = mapclassify.NaturalBreaks(pharmacies_gdf["capacity"], k=3)
pharm_bins = cap_classifier.bins
pharm_labels, lower = [], pharmacies_gdf["capacity"].min()
for upper in pharm_bins:
    pharm_labels.append(f"{lower:,.0f} – {upper:,.0f}")
    lower = upper

fig, ax = plt.subplots(figsize=(12, 12))

ai_gdf.plot(
    ax=ax, column="Ai_scaled", scheme="NaturalBreaks", k=6,
    cmap="YlOrRd", marker="s", markersize=24, edgecolor="black",
    linewidth=0.3, legend=False, alpha=0.85, zorder=2,
)
gdf_3857.plot(ax=ax, **BOUNDARY_STYLE)
pharmacies_gdf.plot(
    ax=ax, color="green", marker="P",
    markersize=pharmacies_gdf["capacity"] * SIZE_MULTIPLIER,
    edgecolor="white", linewidth=0.5, zorder=4,
)

_finish_map(ax, boundary_gdf)

# Accessibility legend
ai_bins = ai_classifier.bins
labels, lower = [], ai_gdf["Ai_scaled"].min()
for upper in ai_bins:
    labels.append(f"{lower:,.1f} – {upper:,.1f}")
    lower = upper

colors = plt.cm.YlOrRd(np.linspace(0.2, 1, len(labels)))
ai_legend = ax.legend(
    handles=[mpatches.Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(colors, labels)],
    title="Accessibility index (×10³)",
    loc="upper right", bbox_to_anchor=(1.0, 1.0),
    frameon=True, fontsize=9, title_fontsize=10, alignment="left",
)
style_legend(ai_legend)
ax.add_artist(ai_legend)

pharm_leg = _pharmacy_legend(ax, pharm_bins, pharm_labels, anchor_y=0.80)
ax.add_artist(pharm_leg)

# Horgen annotation
ax.annotate(
    "Schinzenhof Apotheke Drogerie\nhas no influence on the accessibility index",
    xy=(957022.09, 5984627.56), xytext=(954100, 5983000),
    bbox=ANNOTATION_BBOX, arrowprops=ANNOTATION_ARROW,
    ha="center", va="center", **ANNOTATION_TEXT,
)

# Accessibility desert annotation
dense_text, dense_t1, dense_t2 = (949600, 5985600), (953200, 5988500), (954800, 5985900)
ax.annotate(
    "Even a densely populated area lacks\nwalkable accessibility to the service.\n"
    "In the whole study area,\nover 6,500 people are forced\nto use a vehicle.",
    xy=dense_t1, xytext=dense_text, xycoords="data",
    bbox=ANNOTATION_BBOX,
    ha="left", va="top", **ANNOTATION_TEXT,
)
ax.annotate("", xy=dense_t2, xytext=dense_text,
            arrowprops=ANNOTATION_ARROW, zorder=5)

plt.savefig(PATHS["out_accessibility"], dpi=300, bbox_inches="tight")
print(f"Saved → {PATHS['out_accessibility']}")
plt.show()


# ── 4. Map 3: Optimised pharmacy locations ───────────────────────────────────
print("\n── Map 3: Accessibility (optimised) ──")

ai_gdf["Ai_scaled"] = ai_gdf["Ai_walk_optimum"] * 1000

# Force zero as its own break; natural breaks on positive values only
positive_vals = ai_gdf.loc[ai_gdf["Ai_scaled"] > 0, "Ai_scaled"]
custom_bins = [0.0] + list(mapclassify.NaturalBreaks(positive_vals, k=5).bins) \
    if not positive_vals.empty else [0.0]
ai_classifier = mapclassify.UserDefined(ai_gdf["Ai_scaled"], bins=custom_bins)

cap_classifier  = mapclassify.NaturalBreaks(pharmacies_relo["capacity"], k=3)
pharm_bins = cap_classifier.bins
pharm_labels, lower = [], pharmacies_relo["capacity"].min()
for upper in pharm_bins:
    pharm_labels.append(f"{lower:,.0f} – {upper:,.0f}")
    lower = upper

fig, ax = plt.subplots(figsize=(12, 12))

ai_gdf.plot(
    ax=ax, column="Ai_scaled",
    scheme="UserDefined", classification_kwds={"bins": custom_bins},
    cmap="YlOrRd", marker="s", markersize=24, edgecolor="black",
    linewidth=0.3, legend=False, alpha=0.85, zorder=2,
)
gdf_3857.plot(ax=ax, **BOUNDARY_STYLE)
pharmacies_relo.plot(
    ax=ax, color="blue", marker="P",
    markersize=pharmacies_relo["capacity"] * SIZE_MULTIPLIER,
    edgecolor="white", linewidth=0.5, zorder=4,
)

_finish_map(ax, boundary_gdf)

# Accessibility legend — label zero bin explicitly
labels, lower = [], ai_gdf["Ai_scaled"].min()
for upper in custom_bins:
    labels.append("0.0 (No Accessibility)" if (lower == 0.0 and upper == 0.0)
                  else f"{lower:,.1f} – {upper:,.1f}")
    lower = upper

colors = plt.cm.YlOrRd(np.linspace(0.1, 1, len(labels)))
ai_legend = ax.legend(
    handles=[mpatches.Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(colors, labels)],
    title="Accessibility index (×10³)",
    loc="upper right", bbox_to_anchor=(1.0, 1.0),
    frameon=True, fontsize=9, title_fontsize=10, alignment="left",
)
style_legend(ai_legend)
ax.add_artist(ai_legend)

pharm_leg = _pharmacy_legend(ax, pharm_bins, pharm_labels, anchor_y=0.80, marker_color="blue")
ax.add_artist(pharm_leg)

# Residual zero-access annotation
dense_text, dense_t2 = (950600, 5983600), (955435.71, 5984506.54)
ax.annotate(
    "There are only 124 individuals\nliving in areas with 0 accessibility.",
    xy=(953200, 5988500), xytext=dense_text, xycoords="data",
    bbox=ANNOTATION_BBOX,
    ha="left", va="top", **ANNOTATION_TEXT,
)
ax.annotate("", xy=dense_t2, xytext=dense_text,
            arrowprops=ANNOTATION_ARROW, zorder=5)

plt.savefig(PATHS["out_optimised"], dpi=300, bbox_inches="tight")
print(f"Saved → {PATHS['out_optimised']}")
plt.show()
