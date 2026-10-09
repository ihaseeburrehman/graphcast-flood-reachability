"""Figures of the paper (Nature Portfolio figure specifications). Run all or one: 11_paper_figures.py [fig1 ...]

Style: Arial, text 5-7 pt, panel labels 8-pt bold lowercase, Okabe-Ito colour-blind-safe colours, no coloured text,
units on all axes, 183 mm (double-column) width, vector PDF with embedded TrueType fonts (pdf.fonttype 42).
Lead convention: days from initialisation to 14 Jul 06 UTC (start of the Alzette/Ahr target window); Vesdre's window
starts 12 h earlier but is plotted at the same initialisation lead. Basin rain = sum over the target's 6-h steps of the
basin-cell mean, as % of the radar total.
Outputs: manuscript/figures/figN_*.pdf (+ .png previews)
"""
import json, sys
from datetime import datetime
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
MM = 1 / 25.4
W2 = 183 * MM                                                     # double-column width
# Okabe-Ito
C = dict(obs="#000000", ctl="#7a7a7a", opt="#D55E00", ecmwf="#0072B2", ens_fill="#9ecae1", minus="#56B4E9",
         null="#bdbdbd")
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 450})


def panel(ax, letter, title="", x=-0.02):
    """8-pt bold lowercase panel label + 7-pt title, above the axes on the left."""
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
    if title:
        ax.text(x + 0.03, 1.06, title, transform=ax.transAxes, fontsize=7, va="bottom", ha="left")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    print("wrote", name)


def gc_runs(target, prefix):
    """GraphCast control and optimised basin rain (% of radar) per initialisation, from the final runs."""
    c = CFG[target]; valid = [np.datetime64(v) for v in c["valid"]]; rows = []
    for d in sorted(RUNS.glob(f"{prefix}_2021*")):
        s = json.load(open(d / "summary.json")); t0 = datetime.fromisoformat(s["t0"])
        r = xr.open_dataset(d / "rain_control_vs_optimised.nc")
        tr = sum(s["truth_basin_mm"])
        ctl = float(r.basin_control.sel(time=valid).sum()); opt = float(r.basin_optimised.sel(time=valid).sum())
        cv = s["convergence"]
        rows.append(dict(t0=t0, lead=(REF - t0).total_seconds() / 86400, ctl=100 * ctl / tr, opt=100 * opt / tr,
                         opt_routed=s["final_forecast"]["basin_pct_routed"], chi2=cv["chi2_per_obs"],
                         grad_ratio=cv["grad_ratio"], Jb=cv["Jb_final"], r_win=s["final_forecast"]["window_corr_routed"],
                         iters=cv["iterations"], stop=cv["stop"]))
    return pd.DataFrame(rows).sort_values("lead")


def ifs(target):
    op = pd.read_csv(R / "operational_basin_rain.csv"); op = op[op.basin == target].copy()
    op["lead"] = (REF - pd.to_datetime(op.init)).dt.total_seconds() / 86400
    op["pct"] = 100 * op.basin_mm / op.truth_mm
    return op


BASIN_SHP = ROOT / "data/shapes/alzette_upstream_luxembourg.geojson"   # Alzette upstream of Luxembourg City, 467 km2


def rain_map(ax, fld, c, lev, reg):
    """Rain map with the Alzette basin outline (solid) and the 0.25-deg cells averaged as basin rain (dashed)."""
    import geopandas as gpd
    cf = ax.contourf(fld.lon, fld.lat, fld, levels=lev, cmap="Blues", extend="max")
    gpd.read_file(BASIN_SHP).boundary.plot(ax=ax, color=C["obs"], lw=0.8)
    bl, bo = c["cells"]["lat"], c["cells"]["lon"]
    ax.add_patch(plt.Rectangle((min(bo) - .125, min(bl) - .125), len(bo) * .25, len(bl) * .25, fill=False,
                               ec=C["obs"], lw=0.7, ls="--"))
    ax.set_aspect(1 / np.cos(np.deg2rad(50)))
    ax.set_xlim(reg[2] - .125, reg[3] + .125); ax.set_ylim(reg[0] - .125, reg[1] + .125)
    ax.set_xlabel("Longitude (°E)")
    return cf


def fig1():
    """a-c basin rain against lead time; d-f Alzette-window rain 1.75 days ahead."""
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.3, top=0.86)
    for i, (tg, (name, pre)) in enumerate(BASINS.items()):
        ax = fig.add_subplot(gs[0, i]); g = gc_runs(tg, pre); o = ifs(tg)
        e = o[o.system == "ENS"].groupby("lead").pct; q = e.quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(q.index, q[0.1], q[0.9], color=C["ens_fill"], alpha=0.5, lw=0, label="ECMWF ensemble, 10–90%")
        ax.plot(q.index, q[0.5], color=C["ecmwf"], lw=0.9, ls="--", label="ECMWF ensemble median")
        ec = o[o.system == "ENS-control"].sort_values("lead")
        ax.plot(ec.lead, ec.pct, color=C["ecmwf"], lw=0.7, ls=":", label="ECMWF ensemble control")
        h = o[o.system == "HRES"].sort_values("lead")
        ax.plot(h.lead, h.pct, color=C["ecmwf"], lw=1.0, label="ECMWF high resolution")
        ax.plot(g.lead, g.ctl, "o-", color=C["ctl"], ms=2.8, lw=1.0, label="GraphCast")
        conv = g.lead <= 3.0
        ax.plot(g.lead[conv], g.opt[conv], "o-", color=C["opt"], ms=3.2, lw=1.1, label="GraphCast, optimised initial state")
        ax.plot(g.lead[~conv], g.opt[~conv], "o", mfc="white", mec=C["opt"], ms=3.2, mew=0.8,
                label="GraphCast, optimised (beyond 3 days, not interpreted)")
        ax.axhline(100, color=C["obs"], lw=0.7, label="Radar")
        ax.set_xlim(7.4, 0.4); ax.set_ylim(0, 160); ax.set_xlabel("Lead time (days)")
        panel(ax, "abc"[i], name)
        if i == 0:
            ax.set_ylabel("Basin rain (% of radar)")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
                       handlelength=2.2, columnspacing=1.2)
    c = CFG["alzette_2021"]; valid = [np.datetime64(v) for v in c["valid"]]; reg = c["region"]
    r = xr.open_dataset(RUNS / "v5lead_20210712T12/rain_control_vs_optimised.nc").sel(time=valid).sum("time")
    r = r.sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
    rad = xr.open_dataset(ROOT / c["truth"]).precip_6h.sel(time=valid).sum("time").sel(lat=slice(reg[0], reg[1]),
                                                                                      lon=slice(reg[2], reg[3]))
    lev = np.arange(0, 105, 10)
    for j, (lab, fld) in enumerate([("Radar", rad), ("GraphCast, 1.75 days", r.rain_control),
                                    ("Optimised initial state, 1.75 days", r.rain_optimised)]):
        ax = fig.add_subplot(gs[1, j]); cf = rain_map(ax, fld, c, lev, reg); panel(ax, "def"[j], lab)
        if j == 0:
            ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(cf, ax=fig.axes[3:], shrink=0.85, pad=0.02); cb.set_label("Rain, 14 Jul 06 UTC – 15 Jul 00 UTC (mm)")
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig1_leadtime_maps")


GAUGES = ("Steinsel", "Pfaffenthal", "Hesperange", "Livange")
PGF = Path("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/analysis/data/pgfplots")


def station_q(name):
    d = R / "flood" / name; f = d / "station_Q.csv" if (d / "station_Q.csv").exists() else d / "results/station_Q.csv"
    return pd.read_csv(f, parse_dates=["Time"])


def fig2():
    """a-d discharge at four gauges 1.75 days ahead; e NSE and f peak error against lead time."""
    fig = plt.figure(figsize=(W2, 105 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.6, wspace=0.42, top=0.87)
    rad = station_q("radar025_20210713T12_control"); t = rad.Time
    ctl, opt = station_q("v5lead_20210712T12_control"), station_q("v5lead_20210712T12_optimised")
    for i, g in enumerate(GAUGES):
        ax = fig.add_subplot(gs[0, i]); o = pd.read_csv(PGF / f"{g.lower()}_merged.csv").Observed.values
        n = len(o); x = (t[:n] - t[0]).dt.total_seconds() / 3600
        ax.plot(x, o, color=C["obs"], lw=1.2, label="Observed")
        ax.plot(x, rad[f"{g}_Q"][:n], color=C["obs"], ls=":", lw=1.0, label="Simulated, radar rain")
        ax.plot(x, ctl[f"{g}_Q"][:n], color=C["ctl"], lw=1.0, label="Simulated, GraphCast rain")
        ax.plot(x, opt[f"{g}_Q"][:n], color=C["opt"], lw=1.0, label="Simulated, GraphCast rain from optimised initial state")
        ax.set_xlabel("Time from 13 Jul 18 UTC (h)"); ax.set_xticks([0, 24, 48, 72, 96])
        panel(ax, "abcd"[i], g)
        if i == 0:
            ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    m = pd.read_csv(ROOT / "results/figures/metrics_flood_gauges_v5.csv")
    rr = m[m.forcing.str.startswith("radar")]
    for j, (col, lab) in enumerate([("nse", "Nash–Sutcliffe efficiency"), ("pde_pct", "Peak discharge error (%)")]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        for f, c, l in [("control", C["ctl"], "GraphCast rain"), ("v5 optimised", C["opt"], "Optimised initial state")]:
            d = m[(m.forcing == f) & (m.lead <= 3.0)]; mm = d.groupby("lead")[col].mean()
            ax.plot(d.lead, d[col], "o", color=c, ms=2.2, alpha=0.45, mew=0)
            ax.plot(mm.index, mm.values, "o-", color=c, ms=3.2, lw=1.1, label=l + " (mean of 4 gauges)")
        ax.axhline(rr[col].mean(), color=C["obs"], ls=":", lw=1.0, label="Radar rain (mean of 4 gauges)")
        ax.set_xlim(3.0, 0.5); ax.set_xticks([2.75, 1.75, 0.75]); ax.set_xlabel("Lead time (days)"); ax.set_ylabel(lab)
        panel(ax, "ef"[j])
        if col == "nse":
            ax.set_ylim(-0.6, 1); ax.legend(loc="lower left")
        else:
            ax.axhline(0, color=C["obs"], lw=0.5); ax.set_ylim(-90, 10)
    save(fig, "fig2_floods")


def fig3():
    """a noise, b fields changed, c other storms, d GraphCast from the 50 ECMWF initial perturbations."""
    fig, axs = plt.subplots(1, 4, figsize=(W2, 52 * MM), gridspec_kw=dict(width_ratios=[1, 1.7, 1, 1], wspace=0.5))
    LEADS = [("20210713T12", "0.75"), ("20210712T12", "1.75"), ("20210711T12", "2.75")]
    ax = axs[0]
    for x, (d, L) in enumerate(LEADS):
        s = json.load(open(RUNS / f"v5robust_{d}/robust_summary.json")); v = s["variants"]
        pn = [m["basin_pct"] for m in s["optimised_plus_noise"]]; na = [m["basin_pct"] for m in s["noise_alone"]]
        ax.plot([x - 0.15] * len(na), na, "o", color=C["ctl"], ms=2.2, alpha=0.6, mew=0,
                label="GraphCast + noise" if x == 0 else None)
        ax.plot([x + 0.15] * len(pn), pn, "o", color=C["opt"], ms=2.2, alpha=0.5, mew=0,
                label="Optimised + noise" if x == 0 else None)
        ax.plot([x - 0.15, x + 0.15], [v["control"]["basin_pct"], v["optimised"]["basin_pct"]], "_", color=C["obs"],
                ms=8, mew=1.2, label="Without noise" if x == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([L for _, L in LEADS]); ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Alzette rain (% of radar)"); ax.set_ylim(0, 100); ax.legend(loc="lower left", handletextpad=0.2)
    panel(ax, "a", "Noise")
    ax = axs[1]
    order = [("control", "None"), ("humidity_only", "q\nonly"), ("dynamics_only_zTuv", "z, T, u, v\nonly"),
             ("no_humidity", "All\nbut q"), ("optimised", "All")]
    greys = ["#d9d9d9", "#969696", "#404040"]
    for k, (d, L) in enumerate(LEADS):
        v = json.load(open(RUNS / f"v5robust_{d}/robust_summary.json"))["variants"]
        ax.bar(np.arange(len(order)) + (k - 1) * 0.27, [v[o]["basin_pct"] for o, _ in order], 0.27, color=greys[k],
               label=f"{L} days")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=5.5)
    ax.set_ylabel("Alzette rain (% of radar)"); ax.set_ylim(0, 100); ax.legend(loc="upper left")
    panel(ax, "b", "Fields changed")
    ax = axs[2]
    n = pd.read_csv(R / "null_20210711T12.csv")
    ax.plot(n[~n.real].chi2_start, n[~n.real].chi2_final, "o", color=C["ctl"], ms=3, mew=0, label="19 other storms")
    ax.plot(n[n.real].chi2_start, n[n.real].chi2_final, "*", color=C["opt"], ms=7, mew=0, label="Observed storm")
    b = np.polyfit(n[~n.real].chi2_start, n[~n.real].chi2_final, 1); xx = np.linspace(n.chi2_start.min(), n.chi2_start.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=C["obs"], ls=":", lw=0.8)
    ax.set_xlabel("Misfit before, χ²/N"); ax.set_ylabel("Misfit after, χ²/N"); ax.set_ylim(0.9, 4.8); ax.legend(loc="upper left", handletextpad=0.3)
    panel(ax, "c", "Other storms, 2.75 days")
    ax = axs[3]
    o = gc_runs("alzette_2021", "v5lead"); o = o.set_index(o.lead.round(2))
    for x, (d, L) in enumerate(LEADS):
        s = json.load(open(RUNS / f"ensicv5_alz_{d}/ensic_summary.json")); tr = sum(s["truth_basin_mm"])
        g = 100 * np.array(s["graphcast_basin_sum"]) / tr
        ax.plot(x + np.random.default_rng(x).uniform(-0.15, 0.15, g.size), g, "o", color=C["ecmwf"], ms=1.8, alpha=0.5,
                mew=0, label="ERA5 + ECMWF pert." if x == 0 else None)
        ax.plot(x, o.loc[float(L), "ctl"], "_", color=C["obs"], ms=8, mew=1.2, label="GraphCast" if x == 0 else None)
        ax.plot(x, o.loc[float(L), "opt"], "*", color=C["opt"], ms=6, mew=0, label="Optimised" if x == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([L for _, L in LEADS]); ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Alzette rain (% of radar)"); ax.set_ylim(0, 130); ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="upper center", handletextpad=0.2, borderaxespad=0.1)
    panel(ax, "d", "ECMWF perturbations")
    save(fig, "fig3_robustness")


def fig4():
    """WRF, 0.75 days ahead: a-c basin rain, d pattern correlation; e-g Alzette-window rain."""
    W = R / "wrf_cases"; B3 = ("alzette_2021", "ahr_2021", "vesdre_2021")

    def val(n, m, b=None):
        s = json.load(open(W / n / "rain_summary.json")); return float(s[b][m]) if b else float(np.mean([s[x][m] for x in B3]))
    groups = [("Control", ["P075_ctl"], [f"P075p_ctlm{k}" for k in range(1, 6)], C["ctl"]),
              ("−δ", ["P075p_mdelta"], [], C["minus"]),
              ("Random", [f"P075m_null{k:02d}" for k in range(1, 20)], [], C["null"]),
              ("+δ", ["P075p_delta"], [f"P075p_deltam{k}" for k in range(1, 6)], C["opt"])]
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.55, wspace=0.45)
    rng = np.random.default_rng(0)
    panels = [(b, "pct", n) for b, n in zip(B3, ("Alzette", "Ahr", "Vesdre"))] + [(None, "window_r", "Three basins")]
    for j, (b, m, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        for x, (name, main, micro, col) in enumerate(groups):
            vm = [val(n, m, b) for n in main]; vu = [val(n, m, b) for n in micro]
            ax.plot(x + rng.uniform(-0.15, 0.15, len(vm)) * (len(vm) > 1), vm, "o", color=col,
                    ms=2.2 if len(vm) > 1 else 4, mec=C["obs"] if len(vm) == 1 else col, mew=0.5 if len(vm) == 1 else 0)
            if vu:
                ax.plot([x + 0.3] * len(vu), vu, "o", color=col, ms=1.8, alpha=0.7, mew=0)
        if m == "pct":
            ax.axhline(100, color=C["obs"], lw=0.6); ax.set_ylim(0, 140)
            if j == 0:
                ax.set_ylabel("Basin rain (% of radar)")
        else:
            ax.set_ylim(-0.1, 0.8); ax.set_ylabel("Pattern correlation with radar")
        ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups])
        panel(ax, "abcd"[j], lab)
    c = CFG["alzette_2021"]; valid = [np.datetime64(v) for v in c["valid"]]; reg = c["region"]
    rad = xr.open_dataset(ROOT / c["truth"]).precip_6h.sel(time=valid).sum("time").sel(lat=slice(reg[0], reg[1]),
                                                                                      lon=slice(reg[2], reg[3]))
    lev = np.arange(0, 105, 10)
    gs2 = gs[1, :].subgridspec(1, 3, wspace=0.3)
    for j, (lab, case) in enumerate([("Radar", None), ("WRF control", "P075_ctl"), ("WRF +δ", "P075p_delta")]):
        ax = fig.add_subplot(gs2[0, j])
        f = rad if case is None else xr.open_dataset(W / case / "rain_025.nc").precip_6h.sel(time=valid).sum("time").sel(
            lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
        cf = rain_map(ax, f, c, lev, reg); panel(ax, "efg"[j], lab)
        if j == 0:
            ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(cf, ax=fig.axes[-3:], shrink=0.85, pad=0.02); cb.set_label("Rain, 14 Jul 06 UTC – 15 Jul 00 UTC (mm)")
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig4_wrf")


if __name__ == "__main__":
    for k in sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4"]:
        globals()[k]()
