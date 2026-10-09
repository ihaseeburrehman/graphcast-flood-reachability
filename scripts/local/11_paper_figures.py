"""Figures of Paper A (2021 diagnosis). One function per figure; run all or one: 11_paper_figures.py [fig1 ...]

Lead convention: days from initialisation to 14 Jul 06 UTC (start of the Alzette/Ahr target window); Vesdre's
window starts 12 h earlier but is plotted at the same initialisation lead (as in 10_ensic_stats.py).
Basin rain = sum over the target's 6-h steps of the basin-cell mean, as % of the radar truth.
Outputs: manuscript/figures/figN_*.pdf (+ .png)
"""
import json, sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/hpc"; RUNS = R / "runs"; FIG = ROOT / "manuscript/figures"; FIG.mkdir(exist_ok=True)
CFG = json.load(open(ROOT / "configs/targets.json"))
REF = datetime(2021, 7, 14, 6)
BASINS = {"alzette_2021": ("Alzette", "v5lead"), "ahr_2021": ("Ahr", "v5ahr"), "vesdre_2021": ("Vesdre", "v5vesdre")}
C = dict(radar="k", ctl="#7f7f7f", opt="#d62728", hres="#1f77b4", ens="#1f77b4", wrf="#2ca02c")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "font.family": "sans-serif", "axes.titlesize": 9, "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    print("wrote", name)


def gc_runs(target, prefix):
    """GraphCast control and optimised basin rain (% of truth) per initialisation, from v5 runs."""
    c = CFG[target]; valid = [np.datetime64(v) for v in c["valid"]]; rows = []
    for d in sorted(RUNS.glob(f"{prefix}_2021*")):
        s = json.load(open(d / "summary.json")); t0 = datetime.fromisoformat(s["t0"])
        r = xr.open_dataset(d / "rain_control_vs_optimised.nc")
        tr = sum(s["truth_basin_mm"])
        ctl = float(r.basin_control.sel(time=valid).sum()); opt = float(r.basin_optimised.sel(time=valid).sum())
        cv = s["convergence"]
        rows.append(dict(t0=t0, lead=(REF - t0).total_seconds() / 86400, ctl=100 * ctl / tr, opt=100 * opt / tr,
                         opt_routed=s["final_forecast"]["basin_pct_routed"], chi2=cv["chi2_per_obs"],
                         grad_ratio=cv["grad_ratio"], Jb=cv["Jb_final"], r_win=s["final_forecast"]["window_corr_routed"]))
    return pd.DataFrame(rows).sort_values("lead")


def ifs(target):
    op = pd.read_csv(R / "operational_basin_rain.csv"); op = op[op.basin == target].copy()
    op["lead"] = (REF - pd.to_datetime(op.init)).dt.total_seconds() / 86400
    op["pct"] = 100 * op.basin_mm / op.truth_mm
    return op


def fig1():
    """Lead-time panels (a-c) + rain maps at 1.75 d (d-f)."""
    fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)
    for i, (tg, (name, pre)) in enumerate(BASINS.items()):
        ax = fig.add_subplot(gs[0, i]); g = gc_runs(tg, pre); o = ifs(tg)
        e = o[o.system == "ENS"].groupby("lead").pct; q = e.quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(q.index, q[0.1], q[0.9], color=C["ens"], alpha=0.15, lw=0, label="ECMWF ensemble 10–90 %")
        ax.plot(q.index, q[0.5], color=C["ens"], lw=1, ls="--", label="ECMWF ensemble median")
        h = o[o.system == "HRES"].sort_values("lead"); ax.plot(h.lead, h.pct, color=C["hres"], lw=1.2, label="ECMWF high-res.")
        ec = o[o.system == "ENS-control"].sort_values("lead"); ax.plot(ec.lead, ec.pct, color=C["hres"], lw=0.8, ls=":", label="ECMWF ens. control")
        ax.plot(g.lead, g.ctl, "o-", color=C["ctl"], ms=3.5, lw=1.2, label="GraphCast")
        conv = g.lead <= 3.0
        ax.plot(g.lead[conv], g.opt[conv], "o-", color=C["opt"], ms=4, lw=1.2, label="GraphCast, optimised start")
        ax.plot(g.lead[~conv], g.opt[~conv], "o", mfc="white", color=C["opt"], ms=4, label="  (beyond 3 days)")
        ax.axhline(100, color=C["radar"], lw=0.8, label="radar")
        ax.set_xlim(7.4, 0.4); ax.set_ylim(0, 160); ax.set_title(f"{'abc'[i]}  {name}", loc="left")
        ax.set_xlabel("lead time (days)")
        if i == 0:
            ax.set_ylabel("basin rain (% of radar)")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=6.5, handlelength=1.6,
                       bbox_to_anchor=(0.5, 1.03))
    # maps at 1.75 d (t0 12 Jul 12 UTC), Alzette window sum
    c = CFG["alzette_2021"]; valid = [np.datetime64(v) for v in c["valid"]]
    r = xr.open_dataset(RUNS / "v5lead_20210712T12/rain_control_vs_optimised.nc").sel(time=valid).sum("time")
    rad = xr.open_dataset(ROOT / c["truth"]).precip_6h.sel(time=valid).sum("time")
    reg = c["region"]; r = r.sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
    rad = rad.sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
    lev = np.arange(0, 105, 10)
    for j, (lab, fld) in enumerate([("radar", rad), ("GraphCast, 1.75 d", r.rain_control), ("optimised start, 1.75 d", r.rain_optimised)]):
        ax = fig.add_subplot(gs[1, j])
        cf = ax.contourf(fld.lon, fld.lat, fld, levels=lev, cmap="Blues", extend="max")
        bl, bo = c["cells"]["lat"], c["cells"]["lon"]
        ax.add_patch(plt.Rectangle((min(bo) - .125, min(bl) - .125), len(bo) * .25, len(bl) * .25, fill=False, ec="r", lw=1))
        ax.set_aspect(1 / np.cos(np.deg2rad(50))); ax.set_title(f"{'def'[j]}  {lab}", loc="left")
        ax.set_xlim(reg[2] - .125, reg[3] + .125); ax.set_ylim(reg[0] - .125, reg[1] + .125)
        if j == 0:
            ax.set_ylabel("latitude (°N)")
        ax.set_xlabel("longitude (°E)")
    cb = fig.colorbar(cf, ax=fig.axes[3:], shrink=0.8, pad=0.02); cb.set_label("rain 14 Jul 06 – 15 Jul 00 UTC (mm)")
    save(fig, "fig1_leadtime_maps")


GAUGES = ("Steinsel", "Pfaffenthal", "Hesperange", "Livange")
PGF = Path("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/analysis/data/pgfplots")


def station_q(name):
    d = R / "flood" / name; f = d / "station_Q.csv" if (d / "station_Q.csv").exists() else d / "results/station_Q.csv"
    q = pd.read_csv(f, parse_dates=["Time"]); return q


def fig2():
    """Floods: hydrographs at four gauges (1.75 d) + gauge-mean NSE and peak error vs lead."""
    fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.5, wspace=0.45)
    rad = station_q("radar025_20210713T12_control"); t = rad.Time
    ctl, opt = station_q("v5lead_20210712T12_control"), station_q("v5lead_20210712T12_optimised")
    for i, g in enumerate(GAUGES):
        ax = fig.add_subplot(gs[0, i]); o = pd.read_csv(PGF / f"{g.lower()}_merged.csv").Observed.values
        n = len(o); x = (t[:n] - t[0]).dt.total_seconds() / 3600
        ax.plot(x, o, "k-", lw=1.4, label="observed")
        ax.plot(x, rad[f"{g}_Q"][:n], color="k", ls=":", lw=1.1, label="radar-forced")
        ax.plot(x, ctl[f"{g}_Q"][:n], color=C["ctl"], lw=1.2, label="GraphCast")
        ax.plot(x, opt[f"{g}_Q"][:n], color=C["opt"], lw=1.2, label="optimised start")
        ax.set_title(f"{'abcd'[i]}  {g}", loc="left"); ax.set_xlabel("hours from 13 Jul 18 UTC" if i == 0 else "hours")
        if i == 0:
            ax.set_ylabel("discharge (m$^3$ s$^{-1}$)")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    m = pd.read_csv(ROOT / "results/figures/metrics_flood_gauges_v5.csv")
    r = m[m.forcing.str.startswith("radar")]
    for j, (col, lab) in enumerate([("nse", "Nash–Sutcliffe efficiency"), ("pde_pct", "peak error (%)")]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]) if j == 0 else fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        if j == 1:
            ax.set_position(ax.get_position().translated(0.03, 0))
        for f, c, l in [("control", C["ctl"], "GraphCast"), ("v5 optimised", C["opt"], "optimised start")]:
            d = m[(m.forcing == f) & (m.lead <= 3.0)]; mm = d.groupby("lead")[col].mean()
            for g, dg in d.groupby("gauge"):
                ax.plot(dg.lead, dg[col], "o", color=c, ms=2.5, alpha=0.35)
            ax.plot(mm.index, mm.values, "o-", color=c, ms=4, lw=1.3, label=l + " (gauge mean)")
        ax.axhline(r[col].mean(), color="k", ls=":", lw=1.1, label="radar-forced (gauge mean)")
        ax.set_xlim(3.0, 0.5); ax.set_xticks([2.75, 1.75, 0.75]); ax.set_xlabel("lead time (days)"); ax.set_ylabel(lab)
        ax.set_title(f"{'ef'[j]}", loc="left")
        if col == "nse":
            ax.set_ylim(-0.6, 1); ax.legend(fontsize=6, loc="lower left")
        else:
            ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-90, 10)
    save(fig, "fig2_floods")



def fig3():
    """The optimised change: a noise, b fields by lead, c specificity, d GraphCast from real ECMWF perturbations."""
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25, 1, 1], wspace=0.6))
    LEADS = [("20210713T12", "0.75"), ("20210712T12", "1.75"), ("20210711T12", "2.75")]
    ax = axs[0]
    for x, (d, L) in enumerate(LEADS):
        s = json.load(open(RUNS / f"v5robust_{d}/robust_summary.json")); v = s["variants"]
        pn = [m["basin_pct"] for m in s["optimised_plus_noise"]]; na = [m["basin_pct"] for m in s["noise_alone"]]
        ax.plot([x - 0.15] * len(na), na, "o", color=C["ctl"], ms=2.5, alpha=0.6)
        ax.plot([x + 0.15] * len(pn), pn, "o", color=C["opt"], ms=2.5, alpha=0.5)
        ax.plot(x - 0.15, v["control"]["basin_pct"], "_", color="k", ms=10, mew=1.5)
        ax.plot(x + 0.15, v["optimised"]["basin_pct"], "_", color="k", ms=10, mew=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([L for _, L in LEADS]); ax.set_xlabel("lead time (days)")
    ax.set_ylabel("Alzette rain (% of radar)"); ax.set_ylim(0, 100); ax.set_title("a  noise", loc="left")
    ax = axs[1]
    order = [("control", "none"), ("humidity_only", "humidity\nonly"), ("dynamics_only_zTuv", "dynamics\nonly"), ("no_humidity", "all but\nhumidity"), ("optimised", "all")]
    cols = ["#d9d9d9", "#969696", "#252525"]
    for k, (d, L) in enumerate(LEADS):
        v = json.load(open(RUNS / f"v5robust_{d}/robust_summary.json"))["variants"]
        ax.bar(np.arange(len(order)) + (k - 1) * 0.27, [v[o]["basin_pct"] for o, _ in order], 0.27, color=cols[k], label=f"{L} d")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=5.5)
    ax.set_ylabel("Alzette rain (% of radar)"); ax.set_ylim(0, 100); ax.legend(fontsize=6, loc="upper left", ncol=1)
    ax.set_title("b  fields changed", loc="left")
    ax = axs[2]
    n = pd.read_csv(R / "null_20210711T12.csv")
    ax.plot(n[~n.real].chi2_start, n[~n.real].chi2_final, "o", color=C["ctl"], ms=3.5, label="19 other storms")
    ax.plot(n[n.real].chi2_start, n[n.real].chi2_final, "*", color=C["opt"], ms=9, label="observed storm")
    b = np.polyfit(n[~n.real].chi2_start, n[~n.real].chi2_final, 1); xx = np.linspace(n.chi2_start.min(), n.chi2_start.max(), 10)
    ax.plot(xx, np.polyval(b, xx), "k:", lw=0.8)
    ax.set_xlabel("χ²/N before"); ax.set_ylabel("χ²/N after"); ax.set_title("c  other storms", loc="left")
    ax.legend(fontsize=5.5, loc="upper left")
    ax = axs[3]
    for x, (d, L) in enumerate(LEADS):
        s = json.load(open(RUNS / f"ensicv5_alz_{d}/ensic_summary.json")); tr = sum(s["truth_basin_mm"])
        g = 100 * np.array(s["graphcast_basin_sum"]) / tr
        ax.plot(x + np.random.default_rng(x).uniform(-0.15, 0.15, g.size), g, "o", color=C["ens"], ms=2, alpha=0.5)
        o = gc_runs("alzette_2021", "v5lead"); o = o.set_index(o.lead.round(2))
        ax.plot(x, o.loc[float(L), "ctl"], "_", color="k", ms=10, mew=1.5)
        ax.plot(x, o.loc[float(L), "opt"], "*", color=C["opt"], ms=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([L for _, L in LEADS]); ax.set_xlabel("lead time (days)")
    ax.set_ylim(0, 100); ax.set_title("d  ECMWF pert.", loc="left")
    save(fig, "fig3_robustness")


def fig4():
    """WRF transfer at 0.75 d: per-basin rain and pooled pattern r for control, -delta, matched nulls, +delta; maps."""
    W = R / "wrf_cases"; B3 = ("alzette_2021", "ahr_2021", "vesdre_2021")
    def val(n, m, b=None):
        s = json.load(open(W / n / "rain_summary.json")); return float(s[b][m]) if b else float(np.mean([s[x][m] for x in B3]))
    groups = [("control", ["P075_ctl"], [f"P075p_ctlm{k}" for k in range(1, 6)], C["ctl"]),
              ("−δ", ["P075p_mdelta"], [], "#4393c3"),
              ("random", [f"P075m_null{k:02d}" for k in range(1, 20)], [], "#bababa"),
              ("+δ", ["P075p_delta"], [f"P075p_deltam{k}" for k in range(1, 6)], C["opt"])]
    fig = plt.figure(figsize=(7.2, 4.8)); gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.1], hspace=0.5, wspace=2.2)
    rng = np.random.default_rng(0)
    panels = [(b, "pct", n) for b, n in zip(B3, ("Alzette", "Ahr", "Vesdre"))] + [(None, "window_r", "pattern r")]
    for j, (b, m, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        for x, (name, main, micro, col) in enumerate(groups):
            vm = [val(n, m, b) for n in main]; vu = [val(n, m, b) for n in micro]
            ax.plot(x + rng.uniform(-0.15, 0.15, len(vm)) * (len(vm) > 1), vm, "o", color=col, ms=2.5 if len(vm) > 1 else 5,
                    mec="k" if len(vm) == 1 else col, mew=0.5)
            if vu:
                ax.plot([x + 0.3] * len(vu), vu, "o", color=col, ms=2, alpha=0.6)
        if m == "pct":
            ax.axhline(100, color="k", lw=0.8, clip_on=True); ax.set_ylim(0, 140)
        else:
            ax.set_ylim(-0.1, 0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=6, rotation=45, ha="right")
        ax.set_title(f"{'abcd'[j]}  {lab}", loc="left")
        if j == 0:
            ax.set_ylabel("basin rain (% of radar)")
        if j == 3:
            ax.yaxis.tick_right(); ax.yaxis.set_label_position("right"); ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(True); ax.set_ylabel("pattern correlation (3 basins)")
    c = CFG["alzette_2021"]; valid = [np.datetime64(v) for v in c["valid"]]; reg = c["region"]
    rad = xr.open_dataset(ROOT / c["truth"]).precip_6h.sel(time=valid).sum("time").sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
    lev = np.arange(0, 105, 10)
    for j, (lab, case) in enumerate([("radar", None), ("WRF control", "P075_ctl"), ("WRF +δ", "P075p_delta")]):
        ax = fig.add_subplot(gs[1, 4 * j:4 * j + 4])
        f = rad if case is None else xr.open_dataset(W / case / "rain_025.nc").precip_6h.sel(time=valid).sum("time").sel(
            lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
        cf = ax.contourf(f.lon, f.lat, f, levels=lev, cmap="Blues", extend="max")
        bl, bo = c["cells"]["lat"], c["cells"]["lon"]
        ax.add_patch(plt.Rectangle((min(bo) - .125, min(bl) - .125), len(bo) * .25, len(bl) * .25, fill=False, ec="r", lw=1))
        ax.set_aspect(1 / np.cos(np.deg2rad(50))); ax.set_title(f"{'efg'[j]}  {lab}", loc="left")
        ax.set_xlim(reg[2] - .125, reg[3] + .125); ax.set_ylim(reg[0] - .125, reg[1] + .125); ax.set_xlabel("longitude (°E)")
        if j == 0:
            ax.set_ylabel("latitude (°N)")
    cb = fig.colorbar(cf, ax=fig.axes[-3:], shrink=0.8, pad=0.02); cb.set_label("rain 14 Jul 06 – 15 Jul 00 UTC (mm)")
    save(fig, "fig4_wrf")


if __name__ == "__main__":
    todo = sys.argv[1:] or [k for k in dir() if k.startswith("fig")]
    for k in todo:
        globals()[k]()
