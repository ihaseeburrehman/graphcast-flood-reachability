"""IFS ENS physics check: merge GraphCast-from-ENS-perturbation runs (ensic_summary.json) with the
IFS ENS member rain of the same forecast and basin window, and test:
  paired : Spearman(GraphCast member rain, IFS member rain) - same initial perturbations, two models
  proj→IFS: Spearman(projection of each IFS perturbation on the GraphCast increment, IFS member rain),
            with its rank among 200 random B^1/2 z increments (one-sided null)
  proj→GC : the same with GraphCast member rain
usage: ensic_analysis.py <runs_dir> <operational_2021.csv> <operational_2024.csv> <out_csv>
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

runs, op21, op24, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
op = pd.concat([pd.read_csv(op21), pd.read_csv(op24)])
rows = []
for f in sorted(glob.glob(str(runs / "ensic_*/ensic_summary.json"))):
    s = json.load(open(f))
    init = str(pd.Timestamp(s["t0"]))
    e = op[(op.basin == s["target"]) & (op.init == init) & (op.system == "ENS")].set_index("member").basin_mm
    if e.empty:
        print("no IFS rain for", f); continue
    ifs = e.reindex(s["members"]).values
    gc, pj, pr = np.array(s["graphcast_basin_sum"]), np.array(s["proj"]), np.array(s["proj_random"])
    obs = sum(s["truth_basin_mm"])
    r_pair = spearmanr(gc, ifs).correlation
    r_pi, r_pg = spearmanr(pj, ifs).correlation, spearmanr(pj, gc).correlation
    null_i = np.array([spearmanr(p, ifs).correlation for p in pr])
    null_g = np.array([spearmanr(p, gc).correlation for p in pr])
    lead = op[(op.basin == s["target"]) & (op.init == init)].lead_days.iloc[0]
    rows.append(dict(target=s["target"], t0=init, lead=lead, obs_mm=round(obs, 1),
                     ifs_median_pct=100 * np.median(ifs) / obs, gc_median_pct=100 * np.median(gc) / obs,
                     r_paired=r_pair, p_paired=spearmanr(gc, ifs).pvalue,
                     r_proj_ifs=r_pi, p_null_ifs=(1 + np.sum(null_i >= r_pi)) / (1 + null_i.size),
                     r_proj_gc=r_pg, p_null_gc=(1 + np.sum(null_g >= r_pg)) / (1 + null_g.size),
                     mean_cos=float(np.mean(s["cos"]))))
df = pd.DataFrame(rows).sort_values(["target", "lead"])
df.to_csv(out, index=False, float_format="%.3f")
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
