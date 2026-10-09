"""FRI pilot step 0: heavy-rain events in E-OBS on ~1 deg boxes (no GPU).

E-OBS v33.0e daily rain (0.1 deg, ens mean; day = 06-06 UTC) averaged onto 1 deg boxes over 45-55N, 0-15E
(boxes with >= 70 % valid land cells). Thresholds = 95th / 98th percentile of MJJAS daily box rain 1991-2020
(wet and dry days). Counts box-days above threshold in MJJAS 2019-2023 and "independent" events (box-days
touching in space or on consecutive days, 3x3x3 neighbourhood, merged).
Output: <out>/eobs_box_daily.nc (box rain MJJAS 1991-2023), <out>/eobs_thresholds.csv, printed counts.
usage: eobs_events.py <eobs_rr.nc> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label

f, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
d = xr.open_dataset(f).rr.sel(latitude=slice(45, 55), longitude=slice(0, 15), time=slice("1991-01-01", "2023-12-31"))
d = d.sel(time=d.time.dt.month.isin([5, 6, 7, 8, 9]))
lat_b = np.floor(d.latitude.values).astype(int); lon_b = np.floor(d.longitude.values).astype(int)
d = d.assign_coords(blat=("latitude", lat_b + 0.5), blon=("longitude", lon_b + 0.5))
valid = d.isel(time=slice(0, 200)).notnull().mean("time") > 0.5                 # land cells
frac = valid.groupby("blat").mean().groupby("blon").mean()
box = d.where(valid).groupby("blat").mean().groupby("blon").mean().load()      # (time, blat, blon)
box = box.where(frac >= 0.7)
box.to_netcdf(out / "eobs_box_daily.nc")
clim = box.sel(time=slice("1991", "2020"))
rows = []
for q in (0.95, 0.98):
    thr = clim.quantile(q, "time")
    ev = (box.sel(time=slice("2019", "2023")) > thr)
    n_boxday = int(ev.sum())
    lab, n_ind = label(ev.values, structure=np.ones((3, 3, 3)))                 # merge neighbours in space/time
    nbox = int(thr.notnull().sum())
    rows.append(dict(q=q, boxes=nbox, thr_median_mm=float(thr.median()), box_days=n_boxday,
                     events_3x3x3=n_ind,
                     per_season=round(n_ind / 5, 1)))
    thr.to_dataframe(name=f"thr_q{int(q*100)}").to_csv(out / f"eobs_thresholds_q{int(q*100)}.csv")
print(pd.DataFrame(rows).to_string(index=False))
