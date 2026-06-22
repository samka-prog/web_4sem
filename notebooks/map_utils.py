"""
map_utils.py
------------
Shared constants and helper functions used across all mapping notebooks.
Import at the top of each notebook with:  from map_utils import *
"""

from pathlib import Path
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as mcolors

# ── File paths ────────────────────────────────────────────────────────────────
DATA   = Path("/Users/ianasam/Documents/study/project/web_4sem/data")
ASSETS = Path("/Users/ianasam/Documents/study/project/web_4sem/assets")

PATHS = {
    # inputs — all files sit flat inside data/
    "shp_zones":         DATA / "Verkehrszonen_Schweiz_NPVM_2017_EPSG4326_selected_cut.shp",
    "csv_accessibility": DATA / "accesibilities_within_zones.csv",
    "csv_population":    DATA / "population.csv",
    "shp_pharmacies":    DATA / "pharmacies_final_EPSG4326.shp",
    "shp_boundary":      DATA / "case_study_iana_boundary_EPSG2056.shp",
    "csv_pharm_relo":    DATA / "pharmacies_moved_replaced.csv",
    # outputs -> assets/ so index.html can reference them directly
    "out_population":    ASSETS / "population_map.png",
    "out_accessibility": ASSETS / "accessibility_map.png",
    "out_optimised":     ASSETS / "accessibility_optimised_map.png",
    "out_difference":    ASSETS / "accessibility_difference_map.png",
    "out_difference1":   ASSETS / "accessibility_difference_map1.png",
}

# ── Municipal reference labels (EPSG:3857 coordinates) ───────────────────────
# Used by traffic_zone_acc and population_map notebooks.
MUNICIPALITIES = {
    "Kilchberg":        (952400, 5995500),
    "Rüschlikon":       (953400, 5993000),
    "Thalwil":          (954000, 5991000),
    "Horgen":           (957900, 5985000),
    "Oberrieden":       (955900, 5987900),
    "Küsnacht":         (956000, 5994000),
    "Gattikon":         (951500, 5988500),
}

# population_map also includes Langnau am Albis
MUNICIPALITIES_EXTENDED = {
    **MUNICIPALITIES,
    "Langnau am Albis": (950000, 5989000),
}

# ── Custom livability colormap (traffic zone difference maps) ─────────────────
#   rust/amber  -> zones that still need a car
#   ivory       -> no meaningful change
#   green       -> zones gaining walkable access
CMAP_LIVABILITY = mcolors.LinearSegmentedColormap.from_list(
    "livability",
    [
        "#B85C2C",  # deep rust    — strong car-dependency remains
        "#D4855A",  # warm amber   — moderate car-dependency
        "#E8C49A",  # sand         — slight car-dependency
        "#F5F0E8",  # ivory        — neutral / no change
        "#A8C9A5",  # sage green   — slight livability gain
        "#4D9E6B",  # mid green    — moderate gain
        "#1A6B45",  # forest green — strong car-free livability gain
    ],
    N=256,
)

# ── Scalebar presets ──────────────────────────────────────────────────────────
SCALEBAR_SOLID = dict(
    dx=1, units="m", dimension="si-length",
    location="upper left",
    box_color="white", box_alpha=0.65,
    color="#111111",
    length_fraction=0.20, scale_loc="bottom",
    font_properties={"family": "sans-serif", "size": 10, "weight": "medium"},
)

SCALEBAR_TRANSPARENT = dict(
    dx=1, units="m", dimension="si-length",
    location="upper left",
    box_alpha=0.0,
    color=(0, 0, 0, 0.7),
    length_fraction=0.25, scale_loc="bottom",
    font_properties={"family": "sans-serif", "size": 9},
)

# ── Reusable helpers ──────────────────────────────────────────────────────────

def add_municipal_labels(ax, municipalities, fontsize=10,
                         color="#666666", weight="semibold"):
    """Render municipality name labels with a white halo stroke."""
    for name, (x, y) in municipalities.items():
        ax.text(
            x, y, name,
            fontsize=fontsize, color=color,
            fontfamily="sans-serif", weight=weight,
            ha="center", va="center", zorder=5,
            path_effects=[path_effects.withStroke(linewidth=4, foreground="white")],
        )


def style_legend(legend, title_size=10, text_size=9, color="#333333"):
    """Apply consistent font and colour styling to a matplotlib legend."""
    legend.get_title().set_fontproperties(
        {"family": "sans-serif", "weight": "bold", "size": title_size}
    )
    legend.get_title().set_color(color)
    for text in legend.get_texts():
        text.set_fontproperties({"family": "sans-serif", "size": text_size})
        text.set_color(color)


def set_map_bounds(ax, gdf, pad=0):
    """Crop the axes to the tight bounding box of a GeoDataFrame."""
    if gdf.empty:
        return
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
