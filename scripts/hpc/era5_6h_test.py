"""Training-target check (audit M8): can GraphCast reproduce the flood rain at a 6-h lead from ERA5?

For each 2021 basin and each 6-h target step, compares
  GraphCast : the first 6-h step of a GraphCast forecast started from ERA5 at t0 = valid - 6 h
              (runs v5era5fc_<t0>, --mode forward --fp32; forward_summary.json)
  ERA5      : ERA5's own 6-h rain ending at the valid time (the training target)
  persist.  : ERA5's 6-h rain ending at t0 (a GraphCast input; a 'copy the input' baseline)
  truth     : radar
Totals over each basin's target window, mm and % of truth.
usage: era5_6h_test.py <runs_dir> <out_csv>
"""
import json, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gc_optimise_ic as G

runs, out = Path(sys.argv[1]), sys.argv[2]
C = json.load(open(G.TARGETS))
ac = xr.open_dataset(G.era5_sources("2021")["accum"])
ac = ac.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in ac.dims else ac
tpv = [k for k in ac.data_vars if "precip" in k or k == "tp"][0]
rows = []
for tg in ("alzette_2021", "ahr_2021", "vesdre_2021"):
    c = C[tg]; lon = [x % 360 for x in c["cells"]["lon"]]
    truth = xr.open_dataset(G.CODE_ROOT / c["truth"]).precip_6h
    for v in c["valid"]:
        tv = datetime.fromisoformat(v); t0 = tv - timedelta(hours=6)
        f = runs / f"v5era5fc_{t0:%Y%m%dT%H}" / "forward_summary.json"
        if not f.exists():
            print("missing", f); continue
        s = json.load(open(f))
        gc = s["basin_rain_6h"][tg][0]                      # first step = rain ending at t0 + 6 h
        e = lambda t: float(ac[tpv].sel(time=np.datetime64(t)).sel(lat=c["cells"]["lat"], lon=lon, method="nearest").mean())
        sc = 1000.0 if float(ac[tpv].max()) < 1 else 1.0
        tr = float(truth.sel(time=np.datetime64(tv)).sel(lat=c["cells"]["lat"], lon=c["cells"]["lon"], method="nearest").mean())
        rows.append(dict(target=tg, valid=v, graphcast_6h=gc, era5=e(tv) * sc, persistence=e(t0) * sc, truth=tr))
df = pd.DataFrame(rows)
df.to_csv(out, index=False, float_format="%.2f")
tot = df.groupby("target")[["graphcast_6h", "era5", "persistence", "truth"]].sum()
for k in ("graphcast_6h", "era5", "persistence"):
    tot[k + "_pct"] = 100 * tot[k] / tot["truth"]
pd.set_option("display.width", 200)
print(df.round(1).to_string(index=False)); print(); print(tot.round(1).to_string())
