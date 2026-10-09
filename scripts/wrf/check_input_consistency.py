"""Check 1: is WRF's ERA5 starting state (ungrib intermediate file) the same ERA5 that GraphCast started from?
Compares T, u, v, q at 850 and 500 hPa at t0 between the intermediate file and GraphCast's ERA5 input files
(gc_optimise_ic.era5_sources('2021')) over the GraphCast perturbation box.
usage: check_input_consistency.py <FILE:YYYY-MM-DD_HH>
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import gc_optimise_ic as G  # noqa: E402
import wps_io               # noqa: E402

f = sys.argv[1]; t0 = np.datetime64(datetime.strptime(Path(f).name[5:18], "%Y-%m-%d_%H"))
F, (la, lo) = wps_io.read(f)
lo180 = np.where(lo > 180, lo - 360, lo)
src = None
for p in G.era5_sources("2021")["pl"]:
    d = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}) if str(p).endswith("grib") else xr.open_dataset(p)
    d = d.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon", "isobaricInhPa": "level", "plev": "level"}.items() if k in d.dims or k in d.coords})
    if "time" in d.coords and t0 in d.time.values:
        src = d.sel(time=t0); break
assert src is not None, f"t0 {t0} not in GraphCast's ERA5 files"
lev = src.level.values; scale = 100.0 if float(lev.max()) > 2000 else 1.0
for gname, wname in (("t", "TT"), ("u", "UU"), ("v", "VV"), ("q", "SPECHUMD")):
    var = [k for k in src.data_vars if k in (gname, {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind", "q": "specific_humidity"}[gname])][0]
    for p in (850, 500):
        g = src[var].sel(level=p * scale).sortby("lat").sel(lat=slice(25, 72))
        g = g.assign_coords(lon=((g.lon + 180) % 360) - 180).sortby("lon").sel(lon=slice(-45, 35))
        w = F[(wname, p * 100)]
        ii = [int(np.argmin(np.abs(la - x))) for x in g.lat.values]; jj = [int(np.argmin(np.abs(lo180 - x))) for x in g.lon.values]
        a, b = g.values, w[np.ix_(ii, jj)]
        print(f"{wname:8s} {p} hPa: max |WRF input - GraphCast input| = {np.max(np.abs(a - b)):.3e}  "
              f"(field sd {a.std():.3e}; corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f})")
