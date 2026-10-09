"""Standard verification statistics for the paper (complements 07_main_figures.py).

 1. Error reduction (as in Vonich & Hakim 2024): % reduction of the RMSE of 6-h rain over
    the regional window (vs radar), optimised vs control, per lead.
 2. Fractions Skill Score (Roberts & Lean 2008) of 6-h rain over the regional window,
    threshold 10 and 20 mm/6 h, neighbourhood 3 x 3 cells (0.75 deg); control, optimised and
    the 11 ERA5-EDA GraphCast members.
 3. Ensembles: CRPS of basin rain (mm) and probability of exceeding 50 % of the observed
    basin rain, for GraphCast-EDA (11) and ECMWF ENS (51), per lead.
 4. Flood (LISFLOOD-FP, as in the multi-model paper): NSE, modified KGE, peak discharge
    error and peak timing error at the four discharge gauges, vs the AGE observations.
Outputs: results/figures/metrics_*.csv
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/hpc"; OUT = ROOT / "results/figures"
CFG = json.load(open(ROOT / "configs/targets.json"))
PAPER = Path("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/analysis/data/pgfplots")
TRUTH = {"alzette_2021": ROOT / "data/radar_target/radar_6h_025deg.nc",
         "ahr_2021": ROOT / "data/radar_target/radklim_6h_025deg_wide.nc",
         "vesdre_2021": ROOT / "data/radar_target/radclim_6h_025deg_wide.nc",
         "boris_2024": ROOT / "data/truth_2024/imerg_boris_2024.nc",
         "valencia_2024": ROOT / "data/truth_2024/imerg_valencia_2024.nc"}
PREFIX = {"alzette_2021": "lead", "ahr_2021": "ahr", "vesdre_2021": "vesdre", "boris_2024": "boris", "valencia_2024": "valencia"}


def lead_days(t0, tg):
    return ((datetime.fromisoformat(CFG[tg]["valid"][0]) - timedelta(hours=6)) - t0).total_seconds() / 86400


def fss(f, o, thr, n=3):
    m = np.isfinite(o)
    fb = uniform_filter(np.where(m, (f >= thr).astype(float), 0), size=(1, n, n), mode="constant")
    ob = uniform_filter(np.where(m, (o >= thr).astype(float), 0), size=(1, n, n), mode="constant")
    num = np.mean((fb - ob)[np.broadcast_to(m, fb.shape)] ** 2)
    den = np.mean(fb[np.broadcast_to(m, fb.shape)] ** 2) + np.mean(ob[np.broadcast_to(m, ob.shape)] ** 2)
    return np.nan if den == 0 else 1 - num / den


def on_truth_grid(field, truth):
    """Model rain (time, lat, lon; lon in -180..180) sampled at the truth cells of the region."""
    return field.sel(lat=truth.lat, lon=truth.lon, method="nearest").values


def region_truth(tg):
    reg = CFG[tg]["region"]; v = [np.datetime64(x) for x in CFG[tg]["valid"]]
    t = xr.open_dataset(TRUTH[tg]).precip_6h.sel(time=v)
    return t.sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))


def crps_ens(x, y):
    x = np.asarray(x, float)
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))


# ── 1 + 2: error reduction and FSS (all five targets; members for 2021) ─────
rows = []
for tg, pre in PREFIX.items():
    obs = region_truth(tg); O = obs.values
    for run in sorted((R / "runs").glob(f"{pre}_20*")):
        S = json.load(open(run / "summary.json"))
        if S.get("target", "alzette_2021") != tg:
            continue
        d = xr.open_dataset(run / "rain_control_vs_optimised.nc").sel(time=obs.time)
        L = lead_days(datetime.fromisoformat(S["t0"]), tg)
        rec = dict(target=tg, lead=round(L, 2))
        for k in ("control", "optimised"):
            F = on_truth_grid(d[f"rain_{k}"], obs); m = np.isfinite(O)
            rec[f"rmse_{k}"] = float(np.sqrt(np.mean((F[m] - O[m]) ** 2)))
            for thr in (10, 20):
                rec[f"fss{thr}_{k}"] = fss(F, O, thr)
        rec["error_reduction_pct"] = 100 * (1 - rec["rmse_optimised"] / rec["rmse_control"])
        mem = R / "runs" / f"members_{run.name.split('_', 1)[1]}" / "members_rain.nc"
        if tg.endswith("_2021") and mem.exists():
            M = xr.open_dataset(mem).rain.sel(time=obs.time)
            for thr in (10, 20):
                rec[f"fss{thr}_members_median"] = float(np.nanmedian([fss(on_truth_grid(M.isel(member=i), obs), O, thr)
                                                                       for i in range(M.sizes["member"])]))
        rows.append(rec)
sp = pd.DataFrame(rows).sort_values(["target", "lead"])
sp.to_csv(OUT / "metrics_rain_spatial.csv", index=False, float_format="%.3f")

# ── 3: ensembles (2021 basins) ──────────────────────────────────────────────
op = pd.read_csv(R / "operational_basin_rain.csv")
erows = []
for tg in ("alzette_2021", "ahr_2021", "vesdre_2021"):
    cells = CFG[tg]["cells"]; v = [np.datetime64(x) for x in CFG[tg]["valid"]]
    y = float(xr.open_dataset(TRUTH[tg]).precip_6h.sel(time=v).sel(lat=cells["lat"], lon=cells["lon"]).mean(("lat", "lon")).sum())
    for mem in sorted((R / "runs").glob("members_2021*")):
        ds = xr.open_dataset(mem / "members_rain.nc")
        if not all(t in ds.time.values for t in v):
            continue
        x = ds.rain.sel(time=v).sel(lat=cells["lat"], lon=cells["lon"]).mean(("lat", "lon")).sum("time").values
        erows.append(dict(target=tg, system="GraphCast-EDA (11)", lead=round(lead_days(datetime.fromisoformat(ds.attrs["t0"]), tg), 2),
                          crps_mm=crps_ens(x, y), p_ge50pct=float(np.mean(x >= 0.5 * y)), median_pct=100 * np.median(x) / y))
    e = op[(op.basin == tg) & op.system.str.startswith("ENS")]
    for L, g in e.groupby("lead_days"):
        x = g.basin_mm.values
        erows.append(dict(target=tg, system="ECMWF ENS (51)", lead=L, crps_mm=crps_ens(x, y),
                          p_ge50pct=float(np.mean(x >= 0.5 * y)), median_pct=100 * np.median(x) / y))
ens = pd.DataFrame(erows).sort_values(["target", "system", "lead"])
ens.to_csv(OUT / "metrics_ensembles.csv", index=False, float_format="%.3f")

# ── 4: flood hydrograph skill (paper metrics) ───────────────────────────────
OBS = {g: pd.read_csv(PAPER / f"{g.lower()}_merged.csv")["Observed"].values for g in ("Steinsel", "Pfaffenthal", "Livange", "Hesperange")}
PEAK = {"Steinsel": 131.5, "Pfaffenthal": 134.5, "Livange": 98.75, "Hesperange": 122.6}


def kge_mod(s, o):
    r = np.corrcoef(s, o)[0, 1]; b = s.mean() / o.mean(); g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)


frows = []
for fd in sorted((R / "flood").glob("*")):
    f = fd / "station_Q.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    run, which = fd.name.rsplit("_", 1)
    S = json.load(open(R / "runs" / run / "summary.json"))
    rec = dict(run=run, forcing=which, lead=round(lead_days(datetime.fromisoformat(S["t0"]), "alzette_2021"), 2),
               bound=S.get("eda_k", 1.0))
    nse, kge, pde, pte = [], [], [], []
    for g, o in OBS.items():
        s = d[f"{g}_Q"].values[: len(o)]
        nse.append(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)); kge.append(kge_mod(s, o))
        pde.append(100 * (s.max() - PEAK[g]) / PEAK[g]); pte.append(6.0 * (np.argmax(s) - np.argmax(o)))
    rec.update(nse=np.mean(nse), kge=np.mean(kge), pde_pct=np.mean(pde), pte_h=np.mean(pte))
    frows.append(rec)
fl = pd.DataFrame(frows).sort_values(["forcing", "lead"])
fl.to_csv(OUT / "metrics_flood.csv", index=False, float_format="%.3f")

pd.set_option("display.width", 200)
print("== rain, regional window ==\n", sp.round(2).to_string(index=False))
print("\n== ensembles ==\n", ens.round(2).to_string(index=False))
print("\n== flood (gauge mean) ==\n", fl.round(2).to_string(index=False))
