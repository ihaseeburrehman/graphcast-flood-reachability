"""Per-gauge flood skill with bootstrap confidence intervals (v5 runs by default; argv[1] = version prefix).

For each forcing (radar at 0.25 deg; GraphCast control and v3-optimised per lead) and each of the
four gauges of the multi-model paper: NSE, modified KGE, peak discharge error (PDE, %) and peak
timing error (PTE, h) against the AGE observations. 95 % BCa intervals for NSE and KGE from a
moving-block bootstrap of the 6-h series (block 4 steps = 24 h, 5000 replicates); PDE and PTE are
single-peak quantities and are reported without intervals.
Outputs: results/figures/metrics_flood_gauges_<ver>.csv
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/hpc"; OUT = ROOT / "results/figures"
CFG = json.load(open(ROOT / "configs/targets.json"))
PAPER = Path("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/analysis/data/pgfplots")
GAUGES = ("Steinsel", "Pfaffenthal", "Livange", "Hesperange")
OBS = {g: pd.read_csv(PAPER / f"{g.lower()}_merged.csv")["Observed"].values for g in GAUGES}
PEAK = {"Steinsel": 131.5, "Pfaffenthal": 134.5, "Livange": 98.75, "Hesperange": 122.6}
rng = np.random.default_rng(1)


def nse(s, o):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(s, o):
    r = np.corrcoef(s, o)[0, 1]; b = s.mean() / o.mean(); g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)


def block_ci(f, s, o, block=4, n=5000, alpha=0.05):
    """BCa interval from a moving-block bootstrap (block = 4 six-hourly steps = 24 h). Bias correction z0 from the
    bootstrap distribution; acceleration from a delete-one-block jackknife (non-overlapping blocks)."""
    from scipy.stats import norm
    T = len(o); nb = int(np.ceil(T / block)); theta = f(s, o); vals = []
    for _ in range(n):
        idx = np.concatenate([np.arange(st, st + block) for st in rng.integers(0, T - block + 1, nb)])[:T]
        with np.errstate(all="ignore"):
            vals.append(f(s[idx], o[idx]))
    vals = np.array(vals); vals = vals[np.isfinite(vals)]
    z0 = norm.ppf(np.clip(np.mean(vals < theta), 1e-6, 1 - 1e-6))
    jack = []
    for k in range(0, T, block):
        keep = np.r_[0:k, min(k + block, T):T]
        with np.errstate(all="ignore"):
            jack.append(f(s[keep], o[keep]))
    jack = np.array(jack); d = jack.mean() - jack
    a = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5) if np.sum(d ** 2) > 0 else 0.0
    q = []
    for z in norm.ppf([alpha / 2, 1 - alpha / 2]):
        q.append(100 * norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    return np.percentile(vals, q)


def lead_days(t0):
    return round(((datetime.fromisoformat(CFG["alzette_2021"]["valid"][0]) - timedelta(hours=6)) - t0).total_seconds() / 86400, 2)


cases = [("radar (0.25°)", np.nan, R / "flood/radar025_20210713T12_control")]
VER = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "v5"
for fd in sorted((R / "flood").glob(f"{VER}lead_*_optimised")):
    d = fd.name.split("_")[1]
    L = lead_days(datetime.strptime(d, "%Y%m%dT%H"))
    cases.append((f"{VER} optimised", L, fd))
    for ctl in (R / "flood" / f"{VER}lead_{d}_control", R / "flood" / f"lead_{d}_control"):
        if ctl.exists():
            cases.append(("control", L, ctl)); break
rows = []
for forcing, L, fd in cases:
    f = fd / "station_Q.csv" if (fd / "station_Q.csv").exists() else fd / "results/station_Q.csv"
    if not f.exists():
        continue
    q = pd.read_csv(f)
    for g in GAUGES:
        o = OBS[g]; s = q[f"{g}_Q"].values[: len(o)]
        n_lo, n_hi = block_ci(nse, s, o); k_lo, k_hi = block_ci(kge, s, o)
        rows.append(dict(forcing=forcing, lead=L, gauge=g, nse=nse(s, o), nse_lo=n_lo, nse_hi=n_hi,
                         kge=kge(s, o), kge_lo=k_lo, kge_hi=k_hi, peak_m3s=s.max(),
                         pde_pct=100 * (s.max() - PEAK[g]) / PEAK[g], pte_h=6.0 * (np.argmax(s) - np.argmax(o))))
df = pd.DataFrame(rows).sort_values(["forcing", "lead", "gauge"])
df.to_csv(OUT / f"metrics_flood_gauges_{VER}.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 220)
print(df.round(2).to_string(index=False))
print("\ngauge means:")
print(df.groupby(["forcing", "lead"])[["nse", "kge", "pde_pct", "pte_h"]].mean().round(2).to_string())
