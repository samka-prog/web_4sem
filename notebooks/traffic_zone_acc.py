# traffic_zone_acc.py
# ====================
# Produces two accessibility difference maps:
#   1. accessibility_difference_map.png  — RdBu_r diverging scheme
#   2. accessibility_difference_map1.png — custom livability color ramp + annotation
#
# Run: python traffic_zone_acc.py
# Requires: map_utils.py in the same directory

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import TwoSlopeNorm
from matplotlib_scalebar.scalebar import ScaleBar
from map_utils import (
    PATHS, MUNICIPALITIES, CMAP_LIVABILITY,
    SCALEBAR_SOLID, add_municipal_labels, set_map_bounds,
)

# ── 1. Load spatial layers ────────────────────────────────────────────────────
gdf_study_zones = gpd.read_file(PATHS["shp_zones"])
if gdf_study_zones.crs is None:
    gdf_study_zones = gdf_study_zones.set_crs("EPSG:4326")

# ── 2. Load and aggregate accessibility metrics ───────────────────────────────
df_access = pd.read_csv(PATHS["csv_accessibility"])
df_access_zonal = (
    df_access
    .groupby("ID")[["Ai_z_norm_correct", "Ai_z_norm_optimized"]]
    .first()
    .reset_index()
)

# ── 3. Merge and reproject ────────────────────────────────────────────────────
gdf_mapped = gdf_study_zones.merge(df_access_zonal, on="ID", how="inner")
gdf_mapped_3857 = gdf_mapped.to_crs(epsg=3857)

print("Merged zones:", len(gdf_mapped_3857))
print("CRS:", gdf_mapped_3857.crs)

# ── 4. Compute accessibility change ──────────────────────────────────────────
# Positive = gained accessibility, Negative = reduced accessibility
gdf_mapped_3857["Accessibility_Change"] = (
    gdf_mapped_3857["Ai_z_norm_optimized"] - gdf_mapped_3857["Ai_z_norm_correct"]
)

# Quick distribution check
counts, bin_edges = np.histogram(
    gdf_mapped_3857["Accessibility_Change"].dropna(), bins=37
)
for i, (lo, hi, n) in enumerate(zip(bin_edges, bin_edges[1:], counts)):
    print(f"Bin {i+1:02d}: [{lo:.4f}, {hi:.4f}) → {n}")

# Symmetric norm centred at zero
max_abs = max(
    abs(gdf_mapped_3857["Accessibility_Change"].min()),
    abs(gdf_mapped_3857["Accessibility_Change"].max()),
) or 0.001  # guard against all-zero edge case

norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
print(f"\nNorm limits → vmin: {-max_abs:.4f}, vcenter: 0, vmax: {max_abs:.4f}")


# ── Helper: common basemap / scalebar / labels steps ─────────────────────────
def _finish_map(ax, gdf):
    """Add basemap, crop, scalebar, and municipal labels."""
    cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, alpha=1, zorder=1)
    set_map_bounds(ax, gdf)
    ax.set_axis_off()
    ax.add_artist(ScaleBar(**SCALEBAR_SOLID))
    add_municipal_labels(ax, MUNICIPALITIES, color="#222222", weight="bold")


# ── 5b. Map 2 — Custom livability color ramp ─────────────────────────────────
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 11))

gdf_mapped_3857.plot(
    column="Accessibility_Change",
    cmap=CMAP_LIVABILITY,
    norm=norm,
    legend=False,
    ax=ax,
    alpha=0.82,
    edgecolor="#666666",
    linewidth=0.35,
    zorder=2,
)

_finish_map(ax, gdf_mapped_3857)

# Annotation — explain amber zones
ax.annotate(
    "Amber zones had extremely high accessibility\n"
    "due to TOD development logic.\n"
    "A decline here means the whole area\n"
    "is becoming more homogeneous.",
    xy=(953200, 5988500),
    xytext=(951500, 5985000),
    xycoords="data",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", lw=0.6),
    fontsize=9, color="#333333", fontfamily="sans-serif",
    ha="left", va="top", zorder=6,
)

# ── Patch legend (replaces inset colorbar) ────────────────────────────────────
from map_utils import style_legend

n_patches = 5
patch_colors = [CMAP_LIVABILITY(p) for p in np.linspace(0, 1, n_patches)]
patch_labels = ["Reduced", "Slightly reduced", "Neutral", "Slightly improved", "Improved"]

handles = [
    mpatches.Patch(facecolor=c, edgecolor="white", label=l)
    for c, l in zip(patch_colors, patch_labels)
]

legend = ax.legend(
    handles=handles,
    title="Change in Accessibility Score",
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    fontsize=9,
    title_fontsize=10,
    alignment="left",
)
style_legend(legend)

plt.tight_layout()
plt.savefig(PATHS["out_difference1"], dpi=96, bbox_inches="tight")
print(f"Map 2 saved → {PATHS['out_difference1']}")
plt.show()