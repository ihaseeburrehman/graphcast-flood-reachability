"""ERA5 6-h precipitation as an alternative 'truth' (same format as the radar truth files).

Reviewer request: fit GraphCast to ERA5 rain (the rain it was trained on) as well as to radar, to separate the
radar-ERA5 difference from the initial-state effect. ERA5 total_precipitation_6hr at valid time T (sum over (T-6h, T],
the GraphCast convention, from the accumulation file of the 2021 runs), converted to mm, on the 0.25 deg grid over
47.5-52.5 N, 3-9.5 E, 13 Jul 06 UTC - 16 Jul 00 UTC.
usage: make_era5_truth.py <accum_v2.nc> <out.nc>
"""
import sys

import numpy as np
import xarray as xr

d = xr.open_dataset(sys.argv[1]).total_precipitation_6hr
d = d.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
t = np.arange(np.datetime64("2021-07-13T06"), np.datetime64("2021-07-16T06"), np.timedelta64(6, "h"))
o = (d.sel(time=t, lat=slice(47.5, 52.5), lon=slice(3.0, 9.5)) * 1000.0).astype(np.float32)
xr.Dataset({"precip_6h": o.assign_attrs(units="mm")}, attrs={"source": "ERA5 total_precipitation_6hr (accum_v2), mm per 6 h"}
           ).to_netcdf(sys.argv[2])
print(o.sizes, float(o.max()))
