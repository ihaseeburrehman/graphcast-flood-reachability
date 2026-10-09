"""Event-level statistics for the IFS ENS physics check (audit fix M3).

Unit of independence = weather event: the 2021 storm (Alzette, Ahr and Vesdre share the same storm
and the same 50 IFS perturbations), Storm Boris 2024, Valencia 2024. For each unit and lead set, the
statistic is the mean Spearman r over its basins (and leads); its one-sided p comes from a
permutation test that shuffles member labels jointly across the unit's basins (10 000 permutations),
which keeps the dependence between basins. Unit p-values are combined with Fisher's method.
Primary test (fixed by the external review before any ENS result was seen): leads 1.75 and 2.75 d,
paired GraphCast-IFS member rain. All other leads and tests are exploratory, with Benjamini-Hochberg
FDR over the per-lead tests.
Lead alignment: Vesdre's target window starts 12 h earlier; its leads are mapped to the Alzette
lead of the same initialisation (0.25 -> 0.75 d etc.).
Inputs: results/hpc/runs/ensic_*/ensic_summary.json, results/hpc/operational_basin_rain*.csv
Output: results/figures/{PREFIX}_event_stats.csv
"""
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, rankdata

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/hpc"
PREFIX = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "ensic"
op = pd.concat([pd.read_csv(R / "operational_basin_rain.csv"), pd.read_csv(R / "operational_basin_rain_2024.csv")])
UNIT = {"alzette_2021": "2021 storm", "ahr_2021": "2021 storm", "vesdre_2021": "2021 storm",
        "boris_2024": "Boris 2024", "valencia_2024": "Valencia 2024"}
rng = np.random.default_rng(0); NPERM = 10000

cases = []
for f in sorted(glob.glob(str(R / f"runs/{PREFIX}_*/ensic_summary.json"))):
    s = json.load(open(f)); init = str(pd.Timestamp(s["t0"]))
    e = op[(op.basin == s["target"]) & (op.init == init) & (op.system == "ENS")].set_index("member").basin_mm
    lead = float(op[(op.basin == s["target"]) & (op.init == init)].lead_days.iloc[0])
    lead = lead + 0.5 if s["target"] == "vesdre_2021" else lead
    cases.append(dict(unit=UNIT[s["target"]], target=s["target"], lead=lead,
                      gc=rankdata(s["graphcast_basin_sum"]), ifs=rankdata(e.reindex(s["members"]).values),
                      proj=rankdata(s["proj"])))


def rho(a, b):                                    # Spearman on ranks = Pearson of ranks
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def unit_test(cs, x, y):
    obs = np.mean([rho(c[x], c[y]) for c in cs])
    n = len(cs[0][x]); null = np.empty(NPERM)
    for k in range(NPERM):
        p = rng.permutation(n)                    # same member shuffle for every basin and lead of the unit
        null[k] = np.mean([rho(c[x][p], c[y]) for c in cs])
    return obs, (1 + np.sum(null >= obs)) / (1 + NPERM)


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


rows = []
for test, (x, y) in {"paired GC~IFS": ("gc", "ifs"), "proj->IFS": ("proj", "ifs"), "proj->GC": ("proj", "gc")}.items():
    for name, leads in [("PRIMARY 1.75+2.75", (1.75, 2.75))] + [(f"{L:.2f} d", (L,)) for L in (0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75)]:
        per_unit = []
        for u in ("2021 storm", "Boris 2024", "Valencia 2024"):
            cs = [c for c in cases if c["unit"] == u and c["lead"] in leads]
            if cs:
                r, p = unit_test(cs, x, y); per_unit.append((u, r, p, len(cs)))
        if not per_unit:
            continue
        rows.append(dict(test=test, leads=name, n_units=len(per_unit),
                         **{f"r_{u.split()[0]}": r for u, r, p, n in per_unit},
                         **{f"p_{u.split()[0]}": p for u, r, p, n in per_unit},
                         fisher_p=combine_pvalues([p for _, _, p, _ in per_unit]).pvalue))
df = pd.DataFrame(rows)
for t in df.test.unique():
    m = (df.test == t) & ~df.leads.str.startswith("PRIMARY")
    df.loc[m, "fdr_q"] = bh(df.loc[m, "fisher_p"].values)
df.to_csv(ROOT / f"results/figures/{PREFIX}_event_stats.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
