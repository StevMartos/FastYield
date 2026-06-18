# import FastYield modules
from fastyield.FastYield import load_planet_table
from fastyield.config import planet_types

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

# import numpy modules
import numpy as np



planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

radius = np.array(planet_table["PlanetRadius"].value) # R_earth
mass   = np.array(planet_table["PlanetMass"].value)   # M_earth

mask_im = planet_table["DiscoveryMethod"]=="Imaging"
mask_rv = planet_table["DiscoveryMethod"]=="Radial Velocity"
mask_tr = planet_table["DiscoveryMethod"]=="Transit"
mask_ot = (~mask_im) & (~mask_rv) & (~mask_tr)

mask_cold = np.array(planet_table["PlanetTeff"].value < 250)
mask_temp = np.array(250 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 500) 
mask_warm = np.array(500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1000) 
mask_hot  = np.array(1000 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1500) 
mask_vhot = np.array(1500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 2000) 
mask_uhot = np.array(2000 <= planet_table["PlanetTeff"].value)

plt.figure(figsize=(10, 6), dpi=300)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$M_\oplus$", fontsize=14)
plt.ylabel(r"$R_\oplus$", fontsize=14)
#plt.title(f"FastYield classification: {len(planet_table)} known exoplanets", fontsize=16)
plt.xlim(np.nanmin(mass[mass!=0]), np.nanmax(mass[mass!=0]))
plt.ylim(np.nanmin(radius[radius!=0]), np.nanmax(radius[radius!=0]))

plt.scatter(mass[mask_im & mask_cold], radius[mask_im & mask_cold], c="#6fb0c9", marker="s")
plt.scatter(mass[mask_rv & mask_cold], radius[mask_rv & mask_cold], c="#6fb0c9", marker="o")
plt.scatter(mass[mask_tr & mask_cold], radius[mask_tr & mask_cold], c="#6fb0c9", marker="v")
plt.scatter(mass[mask_ot & mask_cold], radius[mask_ot & mask_cold], c="#6fb0c9", marker="P")

plt.scatter(mass[mask_im & mask_temp], radius[mask_im & mask_temp], c="#7ac87a", marker="s")
plt.scatter(mass[mask_rv & mask_temp], radius[mask_rv & mask_temp], c="#7ac87a", marker="o")
plt.scatter(mass[mask_tr & mask_temp], radius[mask_tr & mask_temp], c="#7ac87a", marker="v")
plt.scatter(mass[mask_ot & mask_temp], radius[mask_ot & mask_temp], c="#7ac87a", marker="P")

plt.scatter(mass[mask_im & mask_warm], radius[mask_im & mask_warm], c="#c8c26f", marker="s")
plt.scatter(mass[mask_rv & mask_warm], radius[mask_rv & mask_warm], c="#c8c26f", marker="o")
plt.scatter(mass[mask_tr & mask_warm], radius[mask_tr & mask_warm], c="#c8c26f", marker="v")
plt.scatter(mass[mask_ot & mask_warm], radius[mask_ot & mask_warm], c="#c8c26f", marker="P")

plt.scatter(mass[mask_im & mask_hot], radius[mask_im & mask_hot], c="#f0a44f", marker="s")
plt.scatter(mass[mask_rv & mask_hot], radius[mask_rv & mask_hot], c="#f0a44f", marker="o")
plt.scatter(mass[mask_tr & mask_hot], radius[mask_tr & mask_hot], c="#f0a44f", marker="v")
plt.scatter(mass[mask_ot & mask_hot], radius[mask_ot & mask_hot], c="#f0a44f", marker="P")

plt.scatter(mass[mask_im & mask_vhot], radius[mask_im & mask_vhot], c="#e36c4a", marker="s")
plt.scatter(mass[mask_rv & mask_vhot], radius[mask_rv & mask_vhot], c="#e36c4a", marker="o")
plt.scatter(mass[mask_tr & mask_vhot], radius[mask_tr & mask_vhot], c="#e36c4a", marker="v")
plt.scatter(mass[mask_ot & mask_vhot], radius[mask_ot & mask_vhot], c="#e36c4a", marker="P")

plt.scatter(mass[mask_im & mask_uhot], radius[mask_im & mask_uhot], c="#c23a3a", marker="s")
plt.scatter(mass[mask_rv & mask_uhot], radius[mask_rv & mask_uhot], c="#c23a3a", marker="o")
plt.scatter(mass[mask_tr & mask_uhot], radius[mask_tr & mask_uhot], c="#c23a3a", marker="v")
plt.scatter(mass[mask_ot & mask_uhot], radius[mask_ot & mask_uhot], c="#c23a3a", marker="P")

plt.minorticks_on()
plt.tick_params(axis='both', labelsize=12)
#plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# LEGEND
ax = plt.gca()
temp_colors_merged = {"Cold (<250 K)":          "#6fb0c9",
                      "Temperate (250–500 K)":  "#7ac87a",
                      "Warm (500–1000 K)":      "#c8c26f",
                      "Hot (1000–1500 K)":      "#f0a44f",
                      "Very-hot (1500–2000 K)": "#e36c4a",
                      "Ultra-hot (≥2000 K)":    "#c23a3a"}
legend_temp = [mlines.Line2D([0], [0], marker='o', linestyle='', markersize=8, label=lbl, markerfacecolor=col, markeredgecolor='none') for lbl, col in temp_colors_merged.items()]
leg1 = ax.legend(handles=legend_temp, title="Temperature bands", loc='lower right', frameon=True)
ax.add_artist(leg1)
legend_methods = [mlines.Line2D([], [], marker='s', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Imaging'), 
                  mlines.Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Radial Velocity'),
                  mlines.Line2D([], [], marker='v', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Transit'),
                  mlines.Line2D([], [], marker='P', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Other')]
ax.legend(handles=legend_methods, title="Discovery method", loc='upper left', frameon=True)

# TYPES ZONES
def build_type_bounds(planet_types, ax):
    """Retourne une liste {name, m1,m2,r1,r2} en agrégeant toutes les bandes Teff."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    by_type = {}
    for key, c in planet_types.items():
        tname = key.split(" ", 1)[1]  # ex: "Sub-Earth"
        d = by_type.setdefault(tname, {
            "m_min": np.inf, "m_max": -np.inf,
            "r_min": np.inf, "r_max": -np.inf
        })
        d["m_min"] = min(d["m_min"], c["mass_min"])
        d["m_max"] = max(d["m_max"], c["mass_max"])
        d["r_min"] = min(d["r_min"], c["radius_min"])
        d["r_max"] = max(d["r_max"], c["radius_max"])

    # borne infinie -> limite d’axe
    bounds = []
    for name, d in by_type.items():
        m1 = max(d["m_min"], xlim[0])
        m2 = min(d["m_max"], xlim[1]) if np.isfinite(d["m_max"]) else xlim[1]
        r1 = max(d["r_min"], ylim[0])
        r2 = min(d["r_max"], ylim[1]) if np.isfinite(d["r_max"]) else ylim[1]
        if np.isfinite([m1,m2,r1,r2]).all() and (m2>m1) and (r2>r1):
            bounds.append({"name": name, "m1": m1, "m2": m2, "r1": r1, "r2": r2})
    return bounds

def draw_grey_highlight(ax, bounds,
                        face_alpha=0.10, edge_alpha=0.9,
                        edge_lw=1.5, label_fs=10):
    """Surligne en gris (au-dessus de tout)."""
    for b in bounds:
        rect = Rectangle((b["m1"], b["r1"]),
                         b["m2"] - b["m1"], b["r2"] - b["r1"],
                         facecolor=(0,0,0,face_alpha),   # gris translucide
                         edgecolor=(0,0,0,edge_alpha),   # contour gris foncé
                         linewidth=edge_lw,
                         zorder=1e6, clip_on=False)
        ax.add_patch(rect)

        # centre géométrique (axes log)
        cx = np.sqrt(b["m1"]*b["m2"])
        cy = np.sqrt(b["r1"]*b["r2"])
        ax.text(cx, cy, b["name"],
                ha="center", va="center",
                fontsize=label_fs, color="black",
                zorder=1e6+1, alpha=0.9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.8)])

bounds = build_type_bounds(planet_types, ax)
draw_grey_highlight(ax, bounds, face_alpha=0.08, edge_alpha=0.85, edge_lw=1.2, label_fs=10)
plt.draw()
plt.tight_layout()
plt.show()