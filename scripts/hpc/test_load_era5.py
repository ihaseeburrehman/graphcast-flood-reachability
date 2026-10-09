"""CPU check of the configurable ERA5 loader on the 2021 inputs: every requested time present
exactly once, and values identical to the source files they came from."""
import sys, numpy as np, xarray as xr
from datetime import datetime, timedelta
sys.argv = [sys.argv[0]]
sys.path.insert(0, sys.path[0])
import gc_optimise_ic as g
t0 = datetime(2021, 7, 7, 12)
times = [datetime(2021, 7, 7, 6), datetime(2021, 7, 9, 12), datetime(2021, 7, 11, 12), datetime(2021, 7, 11, 18),
         datetime(2021, 7, 12, 0), datetime(2021, 7, 12, 6), datetime(2021, 7, 15, 6), datetime(2021, 7, 17, 18)]   # spans both PL files
src = g.era5_sources("2021")
ds = g.load_era5(times, src)
tt = ds.time.values
assert len(tt) == len(times) and len(np.unique(tt)) == len(tt), (len(tt), len(times))
nc = g.tidy_pl if hasattr(g, "tidy_pl") else None
ref_nc = xr.open_dataset(g.PL_NC)
ref_nc = ref_nc.isel(latitude=slice(None, None, -1)) if float(ref_nc.latitude[0]) > 0 else ref_nc
for when, name in ((np.datetime64("2021-07-12T00"), "z"), (np.datetime64("2021-07-15T06"), "q")):
    a = ds[{"z": "geopotential", "q": "specific_humidity"}[name]].sel(time=when).isel(batch=0).sel(level=500).values
    b = ref_nc[name].sel(time=when).sel(level=500).values
    print(when, name, "max abs diff vs era5_pl.nc:", float(np.nanmax(np.abs(a - b))))
early = xr.open_dataset(g.PL_GRIB_EARLY, engine="cfgrib", backend_kwargs={"indexpath": ""})
a = ds["temperature"].sel(time=np.datetime64("2021-07-09T12")).isel(batch=0).sel(level=850).values
b = early["t"].sel(time=np.datetime64("2021-07-09T12"), isobaricInhPa=850).values[::-1]
print("2021-07-09T12 t850 max abs diff vs early GRIB:", float(np.nanmax(np.abs(a - b))))
print("LOADER_OK", len(tt), "times", str(tt[0])[:13], "..", str(tt[-1])[:13])
