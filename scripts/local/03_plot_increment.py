"""Maps of the optimised initial-state increment (what GraphCast's starting weather was changed by).

Physical sanity check: a credible increment is smooth and synoptic (a trough or moisture
plume upstream of the event), not grid-scale noise, and stays within the EDA spread.

usage: 03_plot_increment.py <run_dir_name>      (under results/pilot/)
"""
import json, sys
from pathlib import Path

import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1]
D = ROOT / "results/pilot" / run
inc = xr.open_dataset(D / "increment.nc").sel(time="t0")
S = json.load(open(D / "summary.json"))
tanh = np.load(D / "increment_tanh_a.npz")
G = 9.80665

fields = [
    ("Δ geopotential height 500 hPa (m)", inc.geopotential.sel(level=500) / G, "RdBu_r"),
    ("Δ temperature 850 hPa (K)", inc.temperature.sel(level=850), "RdBu_r"),
    ("Δ specific humidity 850 hPa (g kg⁻¹)", inc.specific_humidity.sel(level=850) * 1000, "BrBG"),
    ("Δ mean sea-level pressure (hPa)", inc.mean_sea_level_pressure / 100, "RdBu_r"),
    ("Δ wind speed 850 hPa (m s⁻¹)", None, "PuOr"),
]
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(16, 9))
lon = inc.lon.values; lon = np.where(lon > 180, lon - 360, lon)
order = np.argsort(lon); lon = lon[order]; lat = inc.lat.values

for i, (title, f, cmap) in enumerate(fields):
    ax = fig.add_subplot(2, 3, i + 1, projection=proj)
    if f is None:
        u = inc.u_component_of_wind.sel(level=850).values[:, order]
        v = inc.v_component_of_wind.sel(level=850).values[:, order]
        val = np.hypot(u, v)
        m = ax.pcolormesh(lon, lat, val, cmap="Purples", transform=proj, shading="auto")
        sk = (slice(None, None, 12), slice(None, None, 12))
        ax.quiver(lon[sk[1]], lat[sk[0]], u[sk], v[sk], transform=proj, scale=15, width=0.003)
    else:
        val = f.values[:, order]
        vmax = np.nanpercentile(np.abs(val), 99.5) or 1e-9
        m = ax.pcolormesh(lon, lat, val, cmap=cmap, vmin=-vmax, vmax=vmax, transform=proj, shading="auto")
    ax.coastlines(lw=0.5); ax.add_feature(cfeature.BORDERS, lw=0.3)
    ax.plot(6.13, 49.61, "k*", ms=10, transform=proj)
    ax.set_title(title, fontsize=10)
    plt.colorbar(m, ax=ax, shrink=0.75, orientation="horizontal", pad=0.04)

ax = fig.add_subplot(2, 3, 6)
names = [k for k in tanh.files if k not in ("plat", "plon")]
fr = [float(np.mean(np.abs(tanh[k].astype(np.float32)))) for k in names]
mx = [float(np.mean(np.abs(tanh[k].astype(np.float32)) > 0.9)) * 100 for k in names]
y = np.arange(len(names))
ax.barh(y, fr, color="0.5", label="mean |δ| / bound")
ax.barh(y, np.array(mx) / 100, color="tab:red", alpha=0.7, label="share of points at >90 % of bound")
ax.set_yticks(y); ax.set_yticklabels([n.replace("_component_of_wind", "").replace("_", " ") for n in names], fontsize=8)
ax.set_xlim(0, 1); ax.legend(fontsize=8); ax.set_title("How much of the allowed range is used", fontsize=10)

h0, h1 = S["history"][0], S["history"][-1]
bound = f"{S.get('eda_k', '')} × EDA spread" if S.get("bound") == "eda" else f"{S['bound_frac']} × clim std"
fig.suptitle(f"Optimised increment at t0 = {S['t0'][:13]}Z (bound {bound}).  "
             f"Alzette {h0.get('basin_sum', h0.get('alzette_sum')):.0f} → {h1.get('basin_sum', h1.get('alzette_sum')):.0f} mm "
             f"(radar {sum(S.get('truth_basin_mm', S.get('radar_alzette_mm'))):.0f});  window r {h0.get('window_corr', float('nan')):.2f} → "
             f"{h1.get('window_corr', float('nan')):.2f}", fontsize=11)
out = D.parent / f"increment_{run}.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print(out)
