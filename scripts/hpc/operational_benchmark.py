"""What forecasters actually had: ECMWF operational HRES and ENS (51 members), July 2021.

For each forecast issued 7-13 July (00 and 12 UTC) and each 2021 basin, the basin-mean rain
in the same target window as the GraphCast experiment (from configs/targets.json), averaged
over the same footprint (the 0.25 deg GraphCast cells covering the basin, +-0.125 deg).
tp is accumulated from the start of the forecast, so window rain = tp(end) - tp(start).

usage: operational_benchmark.py <operational_dir> <out_csv>          (2021: HRES + ENS)
       operational_benchmark.py <ens_ic_dir> <out_csv> 2024             (Boris, Valencia: ENS only,
                                                                         files <event>_{cf,pf}_tp.grib)
"""
import json, sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OP, OUT = Path(sys.argv[1]), sys.argv[2]
cfg = json.load(open(Path(__file__).resolve().parents[2] / "configs/targets.json"))
YEAR = sys.argv[3] if len(sys.argv) > 3 else "2021"
basins = {k: v for k, v in cfg.items() if k.endswith("_" + YEAR)}
# truth in the same windows (basin cells), from the GraphCast runs' summaries (radar 2021, IMERG 2024)
TRUTH = {"alzette_2021": 74.5, "ahr_2021": 75.4, "vesdre_2021": 107.6, "boris_2024": 95.7, "valencia_2024": 74.5}


def open_tp(f):
    d = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
    return d["tp"].assign_coords(longitude=((d.longitude + 180) % 360) - 180)


def footprint_mean(da, cells):
    la0, la1 = min(cells["lat"]) - 0.125, max(cells["lat"]) + 0.125
    lo0, lo1 = min(cells["lon"]) - 0.125, max(cells["lon"]) + 0.125
    sub = da.where((da.latitude >= la0) & (da.latitude <= la1) & (da.longitude >= lo0) & (da.longitude <= lo1), drop=True)
    return sub.mean(("latitude", "longitude"))


rows = []
FILES = ([("HRES", "hres_tp.grib", None), ("ENS-control", "ens_cf_tp.grib", None), ("ENS", "ens_pf_tp.grib", None)]
         if YEAR == "2021" else
         [(sy, f"{k.split('_')[0]}_{t}_tp.grib", k) for k in basins for sy, t in (("ENS-control", "cf"), ("ENS", "pf"))])
for system, f, only in FILES:
    tp = open_tp(OP / f)
    for init in np.atleast_1d(tp.time.values):
        a = tp.sel(time=init)
        for name, b in basins.items():
            if only and name != only:
                continue
            t_start = pd.Timestamp(b["valid"][0]) - timedelta(hours=6)
            t_end = pd.Timestamp(b["valid"][-1])
            s0 = (t_start - pd.Timestamp(init)) / pd.Timedelta(hours=1)
            s1 = (t_end - pd.Timestamp(init)) / pd.Timedelta(hours=1)
            if s0 < 0:
                continue                                   # forecast starts inside the window
            steps = pd.to_timedelta(a.step.values) / pd.Timedelta(hours=1)
            if s1 not in set(steps) or s0 not in set(steps):
                continue
            win = (a.sel(step=pd.Timedelta(hours=s1)) - a.sel(step=pd.Timedelta(hours=s0))) * 1000.0
            m = footprint_mean(win, b["cells"])
            vals = np.atleast_1d(m.values)
            members = np.atleast_1d(m["number"].values) if "number" in m.dims else [0]
            lead_d = (t_start - pd.Timestamp(init)) / pd.Timedelta(days=1)
            for num, v in zip(members, vals):
                rows.append(dict(system=system, basin=name, init=str(pd.Timestamp(init)), lead_days=round(lead_d, 2),
                                 member=int(num), basin_mm=round(float(v), 2),
                                 truth_mm=TRUTH[name]))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
df["pct_truth"] = (100 * df.basin_mm / df.truth_mm).round(1)
ens = df[df.system.str.startswith("ENS")]
summ = ens.groupby(["basin", "lead_days"]).agg(median_pct=("pct_truth", "median"), max_pct=("pct_truth", "max"),
                                               share_ge50pct=("pct_truth", lambda s: round((s >= 50).mean(), 2)))
if YEAR == "2021":
    print("HRES, % of radar:")
    print(df[df.system == "HRES"].pivot_table(index="lead_days", columns="basin", values="pct_truth"))
print("ENS (51 members), % of radar:")
print(summ.to_string())
print("wrote", OUT)
