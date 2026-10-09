"""Supplementary material of Paper A: Fig. S1 (optimised change) and Tables S1-S3 as LaTeX.

Fig. S1: change at t0 in 850-hPa humidity and temperature and 500-hPa geopotential height, Alzette runs at
0.75, 1.75 and 2.75 days. Table S1: per basin and lead, GraphCast and optimised rain, fit, convergence, size,
6-h ceiling. Table S2: GraphCast-ECMWF ensemble agreement, 2021 storm. Table S3: per-gauge flood metrics.
Outputs: manuscript/figures/figS1_increment.pdf, manuscript/si_tables.tex
"""
import importlib.util, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("f", Path(__file__).parent / "11_paper_figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
RUNS, R, FIG = F.RUNS, F.R, F.FIG

# ── Fig. S1 ──
leads = [("20210713T12", "0.75 d"), ("20210712T12", "1.75 d"), ("20210711T12", "2.75 d")]
fields = [("specific_humidity", 850, 1e3, "Δq 850 hPa (g kg$^{-1}$)", 1.5), ("temperature", 850, 1, "ΔT 850 hPa (K)", 0.4),
          ("geopotential", 500, 1 / 9.80665, "ΔZ 500 hPa (m)", 0.4)]
fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.4), constrained_layout=True)
for i, (d, lab) in enumerate(leads):
    inc = xr.open_dataset(RUNS / f"v5lead_{d}/increment.nc").isel(time=-1)
    lon = inc.lon.values; inc = inc.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
    for j, (v, lev, sc, name, vmax) in enumerate(fields):
        ax = axs[i, j]; x = inc[v].sel(level=lev) * sc
        pc = ax.pcolormesh(x.lon, x.lat, x, cmap="RdBu_r" if v != "specific_humidity" else "BrBG", vmin=-vmax, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.add_patch(plt.Rectangle((4.5, 48.5), 3.0, 2.5, fill=False, ec="k", lw=1))   # Alzette storm region
        ax.set_aspect(1 / np.cos(np.deg2rad(50))); ax.set_xlim(-45, 35); ax.set_ylim(25, 72)
        if i == 0:
            ax.set_title(name, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{lab}\nlatitude (°N)")
        if i == 2:
            ax.set_xlabel("longitude (°E)"); fig.colorbar(pc, ax=axs[:, j], orientation="horizontal", shrink=0.8, pad=0.02)
F.save(fig, "figS1_increment")

# ── Table S1 ──
era = pd.read_csv(R / "era5_6h_test.csv").groupby("target")[["graphcast_6h", "era5", "truth"]].sum()
rows = []
for tg, (name, pre) in F.BASINS.items():
    g = F.gc_runs(tg, pre); g = g[g.lead <= 3.0]
    ceil = 100 * era.loc[tg, "graphcast_6h"] / era.loc[tg, "truth"]
    for r in g.itertuples():
        rows.append(f"{name} & {r.lead:.2f} & {r.ctl:.0f} & {r.opt:.0f} & {r.r_win:.2f} & {r.chi2:.2f} & {r.grad_ratio:.2f} & "
                    f"{np.sqrt(2 * r.Jb):.1f} & {ceil:.0f} \\\\")
t1 = ("\\begin{table}[h]\\centering\\small\n\\caption{\\textbf{Optimisation results, 0.75--2.75 days.} Basin rain (\\% of radar) "
      "from ERA5 and from the optimised state; pattern correlation over the storm region; $\\chi^2/N = 2J_o/N$; final over initial "
      "gradient norm; $\\lVert z\\rVert$; GraphCast 6-h forecasts from ERA5 summed over the window (\\% of radar), the "
      "``ceiling''. At 1.75 days the Alzette value is 84, 81 and 73\\% for $r$ = 2, 4 and 8~mm.}\n"
      "\\begin{tabular}{lrrrrrrrr}\\toprule\nBasin & Lead (d) & GraphCast & Optimised & $r$ & $\\chi^2/N$ & $|g|/|g_0|$ & "
      "$\\lVert z\\rVert$ & 6-h ceiling \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\label{tab:s1}\\end{table}\n")

# ── Table S2 ──
e = pd.read_csv(ROOT / "results/figures/ensicv5_event_stats.csv")
fmt = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for test, lab in [("paired GC~IFS", "GraphCast vs ECMWF rain"), ("proj->IFS", "projection on $\\delta$ vs ECMWF rain")]:
    for L in ("PRIMARY 1.75+2.75", "0.75 d", "1.75 d", "2.75 d"):
        r = e[(e.test == test) & (e.leads == L)].iloc[0]
        cells = " & ".join(f"{r[f'r_{u}']:.2f} ({fmt(r[f'p_{u}'])})" for u in ("2021", "Boris", "Valencia"))
        rows.append(f"{lab} & {L.replace('PRIMARY ', '').replace(' d', '')} & {cells} \\\\")
t2 = ("\\begin{table}[h]\\centering\\small\n\\caption{\\textbf{GraphCast and the ECMWF ensemble (exploratory).} Mean Spearman "
      "correlation between GraphCast started from ERA5 plus each of the 50 ECMWF initial perturbations and the same ECMWF "
      "member, and between each perturbation's projection on the optimised change and that member's ECMWF rain; one-sided "
      "permutation $p$ in brackets (10\\,000 joint member shuffles). Storm Boris and the Valencia floods (2024) use the same "
      "method with IMERG rainfall.}\n\\begin{tabular}{llccc}\\toprule\nTest & Lead (d) & July 2021 & Boris & Valencia \\\\\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\label{tab:s2}\\end{table}\n")

# ── Table S3 ──
m = pd.read_csv(ROOT / "results/figures/metrics_flood_gauges_v5.csv")
m["forcing"] = m.forcing.map({"control": "GraphCast", "v5 optimised": "Optimised"}).fillna("Radar")
rows = []
for r in m[~(m.lead > 3.0)].sort_values(["forcing", "lead", "gauge"]).itertuples():
    L = "--" if np.isnan(r.lead) else f"{r.lead:.2f}"
    rows.append(f"{r.forcing} & {L} & {r.gauge} & {r.nse:.2f} [{r.nse_lo:.2f}, {r.nse_hi:.2f}] & {r.kge:.2f} & "
                f"{r.peak_m3s:.0f} & {r.pde_pct:.0f} \\\\")
t3 = ("{\\small\n\\begin{longtable}{llllrrr}\n\\caption{\\textbf{Flood metrics per gauge.} NSE with 95\\% block-bootstrap "
      "interval, modified KGE, simulated peak and peak error. Observed peaks: Steinsel 131.5, Pfaffenthal 134.5, Livange 98.8, "
      "Hesperange 122.6~m$^3$~s$^{-1}$.}\\label{tab:s3}\\\\\\toprule\nForcing & Lead (d) & Gauge & NSE & KGE & Peak & Error (\\%) "
      "\\\\\\midrule\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}\n")
# ── Table S4 ──
rows = []
for d, L in [("20210713T12", "0.75"), ("20210712T12", "1.75"), ("20210711T12", "2.75")]:
    b = json.load(open(RUNS / f"v5balance_{d}/balance_summary.json")); v = b["variants"]
    for k, lab in [("control", "GraphCast"), ("optimised", "optimised"), ("balanced_geopotential", "balanced $\\Delta\\phi$"),
                   ("no_geopotential", "no $\\Delta\\phi$")]:
        x = v[k]; rows.append(f"{L if k == 'control' else ''} & {lab} & {x['basin_pct']:.1f} & {x['window_corr']:.2f} & "
                              f"{x['rmse']['z500']:.1f} & {x['rmse']['t850']:.2f} & {x['rmse']['msl']:.2f} \\\\")
t4 = ("\\begin{table}[h]\\centering\\small\n\\caption{\\textbf{Hydrostatic balance and large-scale error} (Alzette optimisations). "
      "Balanced $\\Delta\\phi$: geopotential change recomputed from the temperature, humidity and sea-level-pressure changes; the "
      "optimised geopotential change is 5--8\\% of the balanced one. RMSE against ERA5 at the target times over 40--60$^\\circ$N, "
      "10$^\\circ$W--20$^\\circ$E (not used in the optimisation). At 2.75 days the changes fitted to the 19 other storms give Z500 "
      "13.2--18.6~m (median 14.2; 18 below the control), T850 0.96--1.14~K and MSLP 0.73--1.07~hPa.}\n\\begin{tabular}{llrrrrr}\\toprule\nLead (d) & Initial state & "
      "Rain (\\%) & $r$ & Z500 (m) & T850 (K) & MSLP (hPa) \\\\\\midrule\n" + "\n".join(rows)
      + "\n\\bottomrule\\end{tabular}\\label{tab:s4}\\end{table}\n")
(ROOT / "manuscript/si_tables.tex").write_text(t1 + "\n" + t2 + "\n" + t4 + "\n" + t3)
print("wrote si_tables.tex")
