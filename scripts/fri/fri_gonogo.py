"""FRI pilot go/no-go (criteria fixed in notes/reachability_design.md before any gradient was computed).

(a) repeat-gradient CV of |g| < 10 %                              (targets with 2 bf16 repeats)
(b) Spearman(|g|, prior-ensemble sd of R) >= 0.7, median |g|/sd in 0.5-2   (targets of inits with a prior ensemble)
(c) design-point forward reaches 0.7-1.4 x (R* - R0)              (median over targets with --design)
(d) redundancy: R^2 of log|g| on log R0 + lead + box lat + box lon + month < 0.5
usage: fri_gonogo.py <fri_dir>
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

fri = Path(sys.argv[1])
rows = []
for f in sorted(glob.glob(str(fri / "fri_grad_*.json"))):
    s = json.load(open(f)); tag = Path(f).stem.split("_")[-1]
    for r in s["targets"]:
        r = dict(r); r.pop("gnorm_var", None); r["tag"] = tag; rows.append(r)
df = pd.DataFrame(rows)
key = ["tag", "lead", "lat0", "lon0"]
g0 = df[df.rep == 0].set_index(key)
print(f"targets: {len(g0)}, inits: {g0.index.get_level_values(0).nunique()}")

# (a)
rep = df.pivot_table(index=key, columns="rep", values="gnorm")
if rep.shape[1] >= 2:
    rr = rep.dropna(); cv = (rr.std(1, ddof=1) / rr.mean(1))
    a_val = float(cv.median()); print(f"(a) repeat CV |g|: median {a_val:.3f}, max {cv.max():.3f} (n={len(rr)})  -> {'PASS' if a_val < 0.10 else 'FAIL'}")
else:
    a_val = None; print("(a) no repeats")

# (b)
sd = []
for f in sorted(glob.glob(str(fri / "fri_prior_*.npz"))):
    z = np.load(f); tag = Path(f).stem.split("_")[-1]; leads = list(z["leads"]); bl, bo = z["blat"], z["blon"]
    S = z["Rm"].std(0, ddof=1)
    for (t, L, la, lo), r in g0.iterrows():
        if t == tag:
            sd.append(dict(tag=t, lead=L, lat0=la, lon0=lo, gnorm=r.gnorm, sdR=float(S[leads.index(L), la - bl[0], lo - bo[0]])))
if sd:
    sd = pd.DataFrame(sd); rho = spearmanr(sd.gnorm, sd.sdR).correlation; ratio = float((sd.gnorm / sd.sdR).median())
    ok = rho >= 0.7 and 0.5 <= ratio <= 2
    print(f"(b) Spearman(|g|, sd R) {rho:.2f}, median ratio {ratio:.2f} (n={len(sd)})  -> {'PASS' if ok else 'FAIL'}")
# (c)
if "design_ratio" in g0:
    dr = g0.design_ratio.dropna(); m = float(dr.median())
    print(f"(c) design ratio median {m:.2f}, IQR {dr.quantile(.25):.2f}-{dr.quantile(.75):.2f} (n={len(dr)})  -> {'PASS' if 0.7 <= m <= 1.4 else 'FAIL'}")
# (d)
d = g0.reset_index()
d = d[(d.gnorm > 0) & (d.R0 > 0)]
X = np.column_stack([np.ones(len(d)), np.log(d.R0), pd.get_dummies(d.lead, drop_first=True).values.astype(float), d.lat0, d.lon0,
                     pd.to_datetime(d.tag, format="%Y%m%dT%H").dt.month])
y = np.log(d.gnorm.values); beta, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"(d) R^2 log|g| ~ log R0 + lead + lat + lon + month = {r2:.2f} (n={len(d)})  -> {'PASS' if r2 < 0.5 else 'FAIL'}")
print(f"    Spearman(|g|, R0) = {spearmanr(d.gnorm, d.R0).correlation:.2f}")
g0.reset_index().to_csv(fri / "fri_targets_table.csv", index=False)
