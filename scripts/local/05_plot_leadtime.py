"""Basin rain reachable vs lead time: GraphCast control vs optimised (within 1 x EDA spread).

usage: 05_plot_leadtime.py [run_prefix=lead] [results_subdir=leadtime]
"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
prefix = sys.argv[1] if len(sys.argv) > 1 else "lead"
sub = sys.argv[2] if len(sys.argv) > 2 else "leadtime"
runs = sorted((ROOT / "results" / sub).glob(f"{prefix}_*"))
rows = []
for r in runs:
    S = json.load(open(r / "summary.json"))
    h0, h1 = S["history"][0], S["history"][-1]
    t0 = datetime.fromisoformat(S["t0"])
    truth = S.get("truth_basin_mm", S.get("radar_alzette_mm"))
    first_target = datetime(2021, 7, 14, 6)   # start of the first 6-h target interval (2021 basins)
    lead_d = (first_target - t0).total_seconds() / 86400
    get = lambda h, k: h.get(f"basin_{k}", h.get(f"alzette_{k}"))
    rows.append((lead_d, get(h0, "sum"), get(h1, "sum"), sum(truth), h0["window_corr"], h1["window_corr"],
                 float(np.mean(list(S["mean_abs_tanh"].values()))), S.get("target", "alzette_2021")))
rows.sort()
L = np.array([r[0] for r in rows]); ctl = np.array([r[1] for r in rows]); opt = np.array([r[2] for r in rows])
obs = rows[0][3]; rc = np.array([r[4] for r in rows]); ro = np.array([r[5] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(L, 100 * ctl / obs, "o-", color="0.45", label="GraphCast as run (ERA5 start)")
ax[0].plot(L, 100 * opt / obs, "o-", color="tab:blue", label="optimised start (≤ 1 × EDA spread)")
ax[0].fill_between(L, 100 * ctl / obs, 100 * opt / obs, color="tab:blue", alpha=0.12)
ax[0].axhline(100, color="k", ls="--", lw=1); ax[0].text(L.max(), 102, "radar", ha="right", fontsize=9)
ax[0].set_xlabel("lead time to the flood-producing rain (days)"); ax[0].set_ylabel("basin rain in target window (% of radar)")
ax[0].set_title(f"{rows[0][7]}: flood rain reachable from realistic starts", fontsize=10)
ax[0].set_ylim(0, 115); ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
for l, c, o in zip(L, ctl, opt):
    ax[0].annotate(f"{o:.0f} mm", (l, 100 * o / obs), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax[1].plot(L, rc, "o-", color="0.45", label="as run"); ax[1].plot(L, ro, "o-", color="tab:blue", label="optimised")
ax[1].set_xlabel("lead time (days)"); ax[1].set_ylabel("pattern correlation with radar (regional window)")
ax[1].set_title("Is the whole storm right, not just the basin?", fontsize=10)
ax[1].invert_xaxis(); ax[1].set_ylim(-0.2, 1); ax[1].axhline(0, color="k", lw=0.5); ax[1].legend(fontsize=8)
fig.tight_layout()
out = ROOT / "results" / sub / f"leadtime_{rows[0][7]}.png"
fig.savefig(out, dpi=130); print(out)
for r in rows:
    print(f"lead {r[0]:.1f} d: control {r[1]:5.1f} mm  optimised {r[2]:5.1f} mm  ({100*r[2]/r[3]:.0f}% of truth)  "
          f"r {r[4]:.2f} -> {r[5]:.2f}  mean|δ|/bound {r[6]:.2f}")
