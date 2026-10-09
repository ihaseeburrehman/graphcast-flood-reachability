"""Write one optimisation run's rain as GraphCast-style forecast files for the paper's forcing builder.

The multi-model paper's builder (ai_to_lisflood_rain_96h.py) reads
forecast_<valid YYYYMMDDTHH>.nc holding total_precipitation_6hr in metres on a 0.25 deg
lat/lon grid. Writing our control and optimised rain in exactly that form lets the flood
runs reuse the builder unchanged (same interval-start convention, window and 10 m grid).

usage: split_for_lisflood.py <run_dir> <control|optimised> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

run, which, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
d = xr.open_dataset(run / "rain_control_vs_optimised.nc")
r = d[f"rain_{which}"].clip(min=0) / 1000.0                     # mm -> m
for t in d.time.values:
    stamp = pd.Timestamp(t).strftime("%Y%m%dT%H")
    f = r.sel(time=t).rename("total_precipitation_6hr").expand_dims(batch=1)
    f.to_dataset().to_netcdf(out / f"forecast_{stamp}.nc")
print(f"{which}: {d.time.size} files {pd.Timestamp(d.time.values[0]):%Y-%m-%d %H} .. "
      f"{pd.Timestamp(d.time.values[-1]):%Y-%m-%d %H} -> {out}")
