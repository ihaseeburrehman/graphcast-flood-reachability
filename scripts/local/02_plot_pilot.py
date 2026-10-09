"""Pilot figure: radar vs GraphCast control vs optimised, and the Alzette rain over the iterations."""
import json, sys
from pathlib import Path
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1] if len(sys.argv) > 1 else "step6_20210713T12"
d = xr.open_dataset(ROOT / f"results/pilot/{run}/rain_control_vs_optimised.nc")
r = xr.open_dataset(ROOT / "data/radar_target/radar_6h_025deg.nc").precip_6h
S = json.load(open(ROOT / f"results/pilot/{run}/summary.json"))
t = np.datetime64("2021-07-14T18")
win = dict(lat=slice(float(d.lat.min()), float(d.lat.max())), lon=slice(float(d.lon.min()), float(d.lon.max())))

fig = plt.figure(figsize=(13, 7.5))
panels = [("Radar (RMI radclim)", r.sel(time=t).sel(**win)),
          ("GraphCast control", d.rain_control.sel(time=t)),
          ("GraphCast optimised", d.rain_optimised.sel(time=t))]
lev = np.arange(0, 52, 4)
for i, (name, f) in enumerate(panels):
    ax = fig.add_subplot(2, 3, i + 1)
    cf = ax.contourf(f.lon, f.lat, f.clip(min=0), levels=lev, cmap="YlGnBu", extend="max")
    ax.add_patch(plt.Rectangle((5.625, 49.125), 1.0, 1.0, fill=False, ec="red", lw=1.5))
    ax.plot(6.13, 49.61, "k*", ms=9)
    ax.set_title(f"{name}\n6 h to 14 Jul 18 UTC", fontsize=10); ax.set_aspect(1.5)
fig.colorbar(cf, ax=fig.axes[:3], shrink=0.8, label="mm per 6 h")

ax = fig.add_subplot(2, 2, 3)
it = [h["iter"] for h in S["history"]]
alz = np.array([h.get("basin_mm", h.get("alzette_mm")) for h in S["history"]])
labels = ["6 h to 14 Jul 12Z", "6 h to 14 Jul 18Z", "6 h to 15 Jul 00Z"][-alz.shape[1]:]
for j in range(alz.shape[1]):
    l, = ax.plot(it, alz[:, j], "o-", ms=3, label=labels[j])
    ax.axhline(S.get("truth_basin_mm", S.get("radar_alzette_mm"))[j], color=l.get_color(), ls="--", lw=1)
ax.set_xlabel("optimisation iteration"); ax.set_ylabel("Alzette rain (mm per 6 h)")
ax.set_title("Alzette rain: dashed = radar", fontsize=10); ax.legend(fontsize=8)

ax = fig.add_subplot(2, 2, 4)
ax.plot(it, [h["loss"] for h in S["history"]], "k.-")
ax.set_xlabel("optimisation iteration"); ax.set_ylabel("loss (mm²)"); ax.set_yscale("log")
bound = f"{S.get('eda_k')} × EDA spread" if S.get("bound") == "eda" else f"{S['bound_frac']} × clim std (placeholder)"
ax.set_title(f"t0 = {S['t0'][:13]}Z, bound {bound}", fontsize=10)
fig.suptitle("Pilot: GraphCast initial-condition optimisation towards radar (target box in red)", fontsize=12)
out = ROOT / f"results/pilot/pilot_{run}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print(out)
