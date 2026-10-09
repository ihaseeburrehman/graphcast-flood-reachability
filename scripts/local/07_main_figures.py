"""Main figures (draft) for the adopted framing.

Fig 1  Attribution of the missed flood at each lead: as run -> recoverable with better
       initial conditions (within 1 x EDA spread) -> model limit; for the LISFLOOD-FP peak
       (gauge mean) and for basin rain. Bound sensitivity (0.5x, 2x) at 1.8 d.
Fig 2  Three systems vs lead (Alzette, Ahr, Vesdre): GraphCast from 11 ERA5 EDA starts,
       ECMWF HRES and ENS (51), and the optimised envelope.
Fig 3  Null test: correlation with the real vs a displaced (~140 km) storm.
Fig 4  Generality: storm pattern correlation vs lead for all five targets.

Inputs: results/hpc (copied from the HPC), configs/targets.json, truth files in data/.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/hpc"
OUT = ROOT / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)
CFG = json.load(open(ROOT / "configs/targets.json"))
OBS_PEAK = {"Steinsel": 131.5, "Pfaffenthal": 134.5, "Livange": 98.75, "Hesperange": 122.6}   # paper's observed peaks
GAUGES = list(OBS_PEAK)
C_CTL, C_OPT, C_OBS, C_GAP = "0.55", "tab:blue", "k", "tab:red"


def lead_days(t0, target):
    first = datetime.fromisoformat(CFG[target]["valid"][0]) - timedelta(hours=6)
    return (first - t0).total_seconds() / 86400


def summary(run):
    return json.load(open(R / "runs" / run / "summary.json"))


def flood_peak_pct(run, which):
    f = R / "flood" / f"{run}_{which}" / "station_Q.csv"
    if not f.exists():
        return np.nan
    d = pd.read_csv(f)
    return float(np.mean([100 * d[f"{g}_Q"].max() / OBS_PEAK[g] for g in GAUGES]))


def get(h, k):
    return h.get(f"basin_{k}", h.get(f"alzette_{k}"))


# ── collect the Alzette lead series ─────────────────────────────────────────
rows = []
for run in sorted(p.name for p in (R / "runs").glob("lead_2021*")):
    S = summary(run); t0 = datetime.fromisoformat(S["t0"])
    truth = sum(S.get("truth_basin_mm", S.get("radar_alzette_mm")))
    rows.append(dict(run=run, lead=lead_days(t0, "alzette_2021"),
                     rain_ctl=100 * get(S["history"][0], "sum") / truth,
                     rain_opt=100 * get(S["history"][-1], "sum") / truth,
                     flood_ctl=flood_peak_pct(run, "control"), flood_opt=flood_peak_pct(run, "optimised")))
A = pd.DataFrame(rows).sort_values("lead")
A.to_csv(OUT / "alzette_attribution.csv", index=False, float_format="%.1f")
sens = {}
for k, run in (("0.5×", "sens05_20210712T12"), ("2×", "sens20_20210712T12")):
    if (R / "runs" / run).exists():
        S = summary(run); truth = sum(S["truth_basin_mm"])
        sens[k] = dict(rain=100 * get(S["history"][-1], "sum") / truth, flood=flood_peak_pct(run, "optimised"),
                       lead=lead_days(datetime.fromisoformat(S["t0"]), "alzette_2021"))

# ── Fig 1: attribution ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, var, title in ((axes[0], "flood", "Flood peak (LISFLOOD-FP, mean of 4 gauges)"),
                       (axes[1], "rain", "Basin rain in the flood window")):
    d = A.dropna(subset=[f"{var}_ctl"])
    x = np.arange(len(d)); ctl = d[f"{var}_ctl"].values; opt = d[f"{var}_opt"].values
    ax.bar(x, ctl, color=C_CTL, label="captured by GraphCast as run")
    ax.bar(x, np.clip(opt - ctl, 0, None), bottom=ctl, color=C_OPT, label="recoverable with better initial conditions")
    ax.bar(x, np.clip(100 - np.maximum(opt, ctl), 0, None), bottom=np.maximum(opt, ctl), color=C_GAP, alpha=0.25,
           label="remaining: model limit")
    for k, s in sens.items():
        if np.isfinite(s[var]):
            i = int(np.argmin(np.abs(d["lead"].values - s["lead"])))
            ax.plot(i, s[var], marker="v" if k == "0.5×" else "^", color="k", ms=8, ls="none",
                    label=f"bound {k} EDA spread" if var == "flood" else None)
    ax.set_xticks(x); ax.set_xticklabels([f"{l:.1f}" for l in d["lead"]])
    ax.set_xlabel("lead time to the flood-producing rain (days)"); ax.set_title(title, fontsize=10)
    ax.axhline(100, color="k", lw=0.8, ls="--"); ax.set_ylim(0, 110)
axes[0].set_ylabel("% of observed")
axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=3, frameon=False)
fig.suptitle("July 2021, Alzette: how much of the missed flood better observations could recover", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig1_attribution.png", dpi=150); plt.close(fig)

# ── Fig 2: three systems vs lead (2021 basins) ──────────────────────────────
op = pd.read_csv(R / "operational_basin_rain.csv")
radar = {"alzette_2021": ROOT / "data/radar_target/radar_6h_025deg.nc",
         "ahr_2021": ROOT / "data/radar_target/radklim_6h_025deg_wide.nc",
         "vesdre_2021": ROOT / "data/radar_target/radclim_6h_025deg_wide.nc"}
prefix = {"alzette_2021": "lead", "ahr_2021": "ahr", "vesdre_2021": "vesdre"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, tg in zip(axes, ("alzette_2021", "ahr_2021", "vesdre_2021")):
    cells = CFG[tg]["cells"]; valid = [np.datetime64(v) for v in CFG[tg]["valid"]]
    truth = float(xr.open_dataset(radar[tg]).precip_6h.sel(time=valid).sel(lat=cells["lat"], lon=cells["lon"])
                  .mean(("lat", "lon")).sum())
    # GraphCast from 11 ERA5 EDA starts (members_* runs store a window covering all 2021 basins)
    for mr in sorted((R / "runs").glob("members_2021*")):
        ds = xr.open_dataset(mr / "members_rain.nc")
        t0 = datetime.fromisoformat(ds.attrs["t0"])
        sel = [v for v in valid if v in ds.time.values]
        if len(sel) < len(valid):
            continue
        vals = 100 * ds.rain.sel(time=sel).sel(lat=cells["lat"], lon=cells["lon"]).mean(("lat", "lon")).sum("time") / truth
        L = lead_days(t0, tg)
        ax.scatter(np.full(vals.size, L), vals, s=12, color="tab:green", alpha=0.6,
                   label="GraphCast, 11 ERA5 analyses" if mr.name.endswith("13T12") else None)
        ax.plot(L, float(np.median(vals)), "_", color="darkgreen", ms=16, mew=2)
    # ECMWF
    e = op[(op.basin == tg) & op.system.str.startswith("ENS")].copy(); e["pct"] = 100 * e.basin_mm / e.truth_mm
    g = e.groupby("lead_days")["pct"]
    ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color="tab:orange", alpha=0.25,
                    label="ECMWF ENS (51) interquartile")
    ax.plot(g.median().index, g.median().values, color="tab:orange", lw=1.5, label="ECMWF ENS median")
    h = op[(op.basin == tg) & (op.system == "HRES")].sort_values("lead_days")
    ax.plot(h.lead_days, 100 * h.basin_mm / h.truth_mm, "s--", color="tab:brown", ms=4, lw=1, label="ECMWF HRES")
    # optimised envelope
    oo = []
    for run in sorted((R / "runs").glob(f"{prefix[tg]}_2021*")):
        S = summary(run.name); oo.append((lead_days(datetime.fromisoformat(S["t0"]), tg),
                                          100 * get(S["history"][-1], "sum") / sum(S.get("truth_basin_mm", S.get("radar_alzette_mm")))))
    oo.sort(); ax.plot(*zip(*oo), "o-", color=C_OPT, label="GraphCast optimised (≤ 1 × EDA spread)")
    ax.axhline(100, color="k", ls="--", lw=0.8); ax.set_xlim(7.4, 0); ax.set_ylim(0, 160)
    ax.set_title(f"{tg.split('_')[0].capitalize()} (radar {truth:.0f} mm)", fontsize=10)
    ax.set_xlabel("lead time (days)")
axes[0].set_ylabel("basin rain in the flood window (% of radar)"); axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Three independent systems lose the July 2021 flood rain at about 4–5 days", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig2_three_systems.png", dpi=150); plt.close(fig)

# ── Fig 3: null test ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))
pairs = [("lead_20210713T12", "null_shift_20210713T12"), ("lead_20210711T12", "null_shift_20210711T12")]
lab, real, fake, real0, fake0 = [], [], [], [], []
for r_run, n_run in pairs:
    Sr, Sn = summary(r_run), summary(n_run)
    lab.append(f"{lead_days(datetime.fromisoformat(Sr['t0']), 'alzette_2021'):.1f} d")
    real0.append(Sr["history"][0]["window_corr"]); real.append(Sr["history"][-1]["window_corr"])
    fake0.append(Sn["history"][0]["window_corr"]); fake.append(Sn["history"][-1]["window_corr"])
x = np.arange(len(lab)); w = 0.35
ax.bar(x - w / 2, real, w, color=C_OPT, label="optimised towards the real storm")
ax.bar(x + w / 2, fake, w, color="tab:purple", label="optimised towards a storm displaced ~140 km")
ax.scatter(x - w / 2, real0, color="k", marker="_", s=300, zorder=3, label="before optimisation")
ax.scatter(x + w / 2, fake0, color="k", marker="_", s=300, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("lead time"); ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("pattern correlation with its target"); ax.set_ylim(-0.6, 1.25)
ax.legend(fontsize=8, loc="upper center", ncol=1, frameon=False)
ax.set_title("Reachable within analysis uncertainty ≠ predictable", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig3_null_test.png", dpi=150); plt.close(fig)

# ── Fig 4: generality across the five targets ───────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for tg, pre, col in (("alzette_2021", "lead", "tab:blue"), ("ahr_2021", "ahr", "tab:cyan"),
                     ("vesdre_2021", "vesdre", "tab:green"), ("boris_2024", "boris", "tab:orange"),
                     ("valencia_2024", "valencia", "tab:red")):
    pts = []
    for run in sorted((R / "runs").glob(f"{pre}_20*")):
        S = summary(run.name)
        if S.get("target", "alzette_2021") != tg:
            continue
        pts.append((lead_days(datetime.fromisoformat(S["t0"]), tg), S["history"][0]["window_corr"], S["history"][-1]["window_corr"]))
    pts.sort()
    L, c0, c1 = map(np.array, zip(*pts))
    ax.plot(L, c1, "o-", color=col, label=tg.replace("_", " ").capitalize())
    ax.plot(L, c0, "o--", color=col, alpha=0.4, ms=3)
ax.set_xlim(7.2, 0); ax.set_ylim(-0.2, 1); ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lead time (days)"); ax.set_ylabel("storm pattern correlation with observed rain")
ax.set_title("Optimised (solid) vs as run (dashed): five flood targets, three storms", fontsize=10)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig4_generality.png", dpi=150); plt.close(fig)

print(A.round(1).to_string(index=False)); print("sensitivity:", sens); print("figures ->", OUT)
