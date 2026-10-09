"""Statistics of the powered 0.75 d WRF transfer test (cases P075p_*, control P075_ctl).

Pre-specified metrics (fixed before the results were seen): per basin and pooled over the three basins,
(1) basin rain as % of radar and (2) window pattern correlation r. Tests:
  rank of +delta among {+delta, 19 nulls}: one-sided p = rank / 20 (pooled: mean over basins of % / of r);
  +delta vs control as the difference of the micro-ensemble means against their pooled spread (Welch t);
  sign: +delta - (-delta).
usage: powered_stats.py <cases_dir> [null_prefix, default P075p_null]
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

NP = sys.argv[2] if len(sys.argv) > 2 else "P075p_null"
C = Path(sys.argv[1]); B = ("alzette_2021", "ahr_2021", "vesdre_2021")


def load(n):
    f = C / n / "rain_summary.json"
    return json.load(open(f)) if f.exists() else None


def val(s, m):                                   # per-basin values + pooled mean
    v = [s[b][m] for b in B]; return v + [float(np.mean(v))]


cols = [b.split("_")[0] for b in B] + ["pooled"]
for m in ("pct", "window_r"):
    print(f"\n=== {m} ===  " + "  ".join(f"{c:>8s}" for c in cols))
    d, md, ct = load("P075p_delta"), load("P075p_mdelta"), load("P075_ctl")
    nulls = [s for s in (load(f"{NP}{k:02d}") for k in range(1, 20)) if s]
    cm = [s for s in (load(f"P075p_ctlm{k}") for k in range(1, 6)) if s]
    dm = [s for s in (load(f"P075p_deltam{k}") for k in range(1, 6)) if s]
    for lab, s in (("control", ct), ("+delta", d), ("-delta", md)):
        if s: print(f"{lab:22s}" + "  ".join(f"{x:8.2f}" for x in val(s, m)))
    N = np.array([val(s, m) for s in nulls])
    if len(N):
        print(f"{'nulls median (n=%d)' % len(N):22s}" + "  ".join(f"{x:8.2f}" for x in np.median(N, 0)))
        print(f"{'nulls max':22s}" + "  ".join(f"{x:8.2f}" for x in N.max(0)))
    if d is not None and len(N):
        dv = np.array(val(d, m)); rank = 1 + (N >= dv).sum(0)
        print(f"{'+delta rank / p':22s}" + "  ".join(f"{r:2d}/{len(N)+1} {r/(len(N)+1):.2f}" for r in rank))
    if cm and dm:
        CM = np.array([val(s, m) for s in cm]); DM = np.array([val(s, m) for s in dm])
        print(f"{'ctl micro mean+-sd':22s}" + "  ".join(f"{a:5.1f}+-{b:4.1f}"[:8] if m == 'pct' else f"{a:4.2f}+-{b:.2f}" for a, b in zip(CM.mean(0), CM.std(0, ddof=1))))
        print(f"{'+delta micro mean+-sd':22s}" + "  ".join(f"{a:5.1f}+-{b:4.1f}"[:8] if m == 'pct' else f"{a:4.2f}+-{b:.2f}" for a, b in zip(DM.mean(0), DM.std(0, ddof=1))))
        p = [ttest_ind(DM[:, j], CM[:, j], equal_var=False, alternative="greater").pvalue for j in range(4)]
        print(f"{'Welch p (+d > ctl)':22s}" + "  ".join(f"{x:8.3f}" for x in p))
