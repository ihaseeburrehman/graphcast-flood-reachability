"""Alternative 'truths' for the null test: realistic storms of the same weather regime.

From the operational IFS ENS forecast issued 11 Jul 2021 12 UTC (lead ~3 d), take the 6-h rain of
each member at the Alzette target steps (14 Jul 12, 18 UTC, 15 Jul 00 UTC) over the regional window,
regridded to the 0.25 deg cells (linear interpolation of 0.2 deg fields). The 3-day ENS mostly
under-forecast the storm (only one member within +-25 % of the radar window rain), so each member's
field is rescaled to the radar's window-mean 3-step rain: realistic patterns and placements of the
observed amount. Members with < 40 % of the radar amount are excluded (no extreme scaling); of the
rest, 19 are drawn at random (seed 0; 19 nulls let a one-sided rank test reach p = 0.05; picking the
least-correlated members would bias towards far-away storms). Each is written in the radar file's
format, for gc_optimise_ic.py --truth-file.
usage: make_ens_truths.py <ens_pf_tp.grib> <radar_6h_025deg.nc> <out_dir>
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

grib, radar_nc, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
INIT = np.datetime64("2021-07-11T12")
VALID = [np.datetime64(x) for x in ("2021-07-14T12", "2021-07-14T18", "2021-07-15T00")]
rad = xr.open_dataset(radar_nc).precip_6h
R = rad.sel(time=VALID)
tp = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""})["tp"].sel(time=INIT)
tp = tp.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
steps = {v: pd.Timedelta(v - INIT) for v in VALID}
fields = []
for v in VALID:
    a = (tp.sel(step=steps[v]) - tp.sel(step=steps[v] - pd.Timedelta(hours=6))) * 1000.0   # mm per 6 h
    fields.append(a.interp(lat=rad.lat, lon=rad.lon).clip(min=0))
F = xr.concat(fields, "time").assign_coords(time=VALID).transpose("number", "time", "lat", "lon")
r3, f3 = R.sum("time").values.ravel(), F.sum("time").values.reshape(F.sizes["number"], -1)
size = f3.mean(1) / r3.mean()
corr = np.array([np.corrcoef(x, r3)[0, 1] for x in f3])
ok = np.where(size >= 0.4)[0]
pick = np.sort(np.random.default_rng(0).choice(ok, size=min(19, ok.size), replace=False))
meta = []
for i in pick:
    m = int(F.number.values[i])
    xr.Dataset({"precip_6h": (F.isel(number=i) / size[i]).drop_vars(["number", "step", "valid_time", "surface"], errors="ignore")
                .astype(np.float32).assign_attrs(units="mm")},
               attrs={"source": f"IFS ENS member {m}, init 2021-07-11 12 UTC, 6-h rain regridded to 0.25 deg",
                      "use": "null-test alternative truth"}).to_netcdf(out / f"ens_truth_m{m:02d}.nc")
    meta.append(dict(member=m, size_ratio_before_scaling=round(float(size[i]), 3), scale_factor=round(float(1 / size[i]), 3),
                     corr_with_radar=round(float(corr[i]), 3)))
json.dump(dict(init=str(INIT), valid=[str(v) for v in VALID], n_eligible=int(ok.size), picked=meta),
          open(out / "ens_truths.json", "w"), indent=1)
print(f"{ok.size} members with >= 40 % of the radar window rain; picked (rescaled):", meta)
