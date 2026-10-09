"""Null test with start-matching (review of 2026-10-05): is the real storm better fit only because the
control already sits closer to it? For the real-radar run and each IFS-member alternative truth at a
lead: starting and final 2J_o/N (final = repeat mean) and the fractional J_o reduction; rank of the real
storm in each metric (1 = best), and the same restricted to nulls that start at least as close as the real one.
usage: null_analysis.py <runs_dir> <t0 YYYYMMDDTHH> <real_run_prefix e.g. v5lead> <out_csv>
"""
import glob, json, os, sys
import pandas as pd

runs, t0, real, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = []
for f in [f"{runs}/{real}_{t0}/summary.json"] + sorted(glob.glob(f"{runs}/v5null_m*_{t0}/summary.json")):
    if not os.path.exists(f):
        continue
    s = json.load(open(f)); c = s["convergence"]; n = c["n_obs"]; h = s["history"]
    jo0, jof = h[0]["Jo"], c["Jo_final"]
    rows.append(dict(run=os.path.basename(os.path.dirname(f)), real=f.startswith(f"{runs}/{real}_"),
                     chi2_start=2 * jo0 / n, chi2_final=2 * jof / n, frac_reduction=(jo0 - jof) / jo0,
                     grad_ratio=c["grad_ratio"], stop=c["stop"]))
d = pd.DataFrame(rows)
r = d[d.real].iloc[0]; nl = d[~d.real]
print(f"t0 {t0}: {len(nl)} nulls")
print(d.round(3).to_string(index=False))
for k, better in (("chi2_final", "low"), ("frac_reduction", "high"), ("chi2_start", "low")):
    rank = 1 + int((nl[k] < r[k]).sum() if better == "low" else (nl[k] > r[k]).sum())
    print(f"{k:15s} real {r[k]:.3f}   null median {nl[k].median():.3f}   rank {rank} of {len(d)}  (p = {rank/len(d):.3f})")
m = nl[nl.chi2_start <= r.chi2_start]
print(f"start-matched nulls (starting at least as close as the real storm): {len(m)}; "
      f"of these with a better final fit than the real storm: {int((m.chi2_final < r.chi2_final).sum())}")
d.to_csv(out, index=False, float_format="%.4f")
