"""FRI pilot: model-climate thresholds and near-miss target selection (chosen on the FORECAST, never on obs).

Inputs: fri_forward_<tag>.npz (GraphCast control box rain R0[lead, lat0, lon0] per init), E-OBS box daily rain and
q98 thresholds (eobs_events.py). Threshold in GraphCast's own climate: R*_box,L = thr_obs_box x k_L, where k_L =
(pooled q98 of GraphCast R0 at lead L) / (pooled q98 of E-OBS on the same days and boxes) - a one-number quantile
map per lead (GraphCast under-forecasts heavy rain). Near-miss: 0.3 R* <= R0 < R*. Up to N targets per init,
drawn at random (seed 0) among near-miss (lead, box) pairs of valid E-OBS boxes.
Output: targets.json {tag: "L:lat0:lon0:Rstar,..."}, thresholds.json (k_L, counts)
usage: fri_select.py <fri_dir> <eobs_box_dir> [N=8]
"""
import glob, json, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

fri, eb = Path(sys.argv[1]), Path(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
thr = pd.read_csv(eb / "eobs_thresholds_q98.csv").dropna()
thr = {(int(r.blat - 0.5), int(r.blon - 0.5)): float(r.thr_q98) for r in thr.itertuples()}
obs = xr.open_dataset(eb / "eobs_box_daily.nc").rr
F = {}
for f in sorted(glob.glob(str(fri / "fri_forward_*.npz"))):
    z = np.load(f); F[Path(f).stem.split("_")[-1]] = z
leads = [int(x) for x in next(iter(F.values()))["leads"]]
BLAT, BLON = F[next(iter(F))]["blat"], F[next(iter(F))]["blon"]
k, cnt = {}, {}
for li, L in enumerate(leads):
    m, o = [], []
    for tag, z in F.items():
        day = datetime.strptime(tag, "%Y%m%dT%H") + timedelta(days=L)          # E-OBS day of the window
        for (la, lo) in thr:
            m.append(float(z["R0"][li, la - BLAT[0], lo - BLON[0]]))
            o.append(float(obs.sel(time=np.datetime64(day.date()), blat=la + 0.5, blon=lo + 0.5)))
    m, o = np.array(m), np.array(o)
    k[L] = float(np.nanquantile(m, 0.98) / np.nanquantile(o, 0.98)); cnt[L] = dict(n=len(m), q98_model=float(np.nanquantile(m, 0.98)),
                                                                                 q98_obs=float(np.nanquantile(o, 0.98)))
rng = np.random.default_rng(0); out, n_near = {}, 0
for tag, z in F.items():
    cand = []
    for li, L in enumerate(leads):
        for (la, lo), t in thr.items():
            rs = t * k[L]; r0 = float(z["R0"][li, la - BLAT[0], lo - BLON[0]])
            if 0.3 * rs <= r0 < rs:
                cand.append(f"{L}:{la}:{lo}:{rs:.2f}")
    n_near += len(cand)
    pick = list(rng.choice(cand, size=min(N, len(cand)), replace=False)) if cand else []
    out[tag] = ",".join(sorted(pick))
json.dump(out, open(fri / "targets.json", "w"), indent=1)
json.dump(dict(k=k, counts=cnt, n_near_miss_total=n_near, n_targets=sum(len(v.split(",")) for v in out.values() if v)),
          open(fri / "thresholds.json", "w"), indent=1)
print(json.dumps(dict(k=k, counts=cnt, n_near_miss_total=n_near), indent=1))
print({t: len(v.split(",")) if v else 0 for t, v in out.items()})
