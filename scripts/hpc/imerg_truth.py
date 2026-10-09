"""GPM IMERG Final (V07, half-hourly, 0.1 deg) -> 6-hour totals on the GraphCast 0.25 deg grid.

Downloads the half-hourly files for [start, end) from GES DISC (login file given by
--netrc, never printed), sums the twelve half-hours ending at each 6-h valid time T
(precipitation is mm/h -> x0.5 h), and averages the 0.1 deg cells whose centres fall in
each 0.25 deg GraphCast cell (6-7 cells each). Same output layout as the radar targets.

usage: imerg_truth.py --start 2024-10-28T00 --end 2024-10-31T00 \
          --bbox 37.5 41.5 -2.5 1.5 --out .../imerg_valencia_2024.nc --netrc FILE --cache DIR
"""
import argparse, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
ap.add_argument("--bbox", nargs=4, type=float, required=True, help="lat0 lat1 lon0 lon1")
ap.add_argument("--out", required=True); ap.add_argument("--netrc", required=True)
ap.add_argument("--cache", required=True)
a = ap.parse_args()
t_start = datetime.strptime(a.start, "%Y-%m-%dT%H"); t_end = datetime.strptime(a.end, "%Y-%m-%dT%H")
cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
cookies = cache / "cookies.txt"


def readable(f):
    try:
        with xr.open_dataset(f, group="Grid", engine="netcdf4", decode_times=False) as h:
            h["precipitation"].values
        return True
    except Exception:
        return False


def fetch(t):
    """Half-hour starting at t; returns the local file (downloads if missing)."""
    doy = t.timetuple().tm_yday
    mins = t.hour * 60 + t.minute
    e = t + timedelta(minutes=29, seconds=59)
    name = f"3B-HHR.MS.MRG.3IMERG.{t:%Y%m%d}-S{t:%H%M%S}-E{e:%H%M%S}.{mins:04d}.V07B.HDF5"
    f = cache / name
    if f.exists() and not readable(f):            # truncated earlier download: fetch again
        f.unlink()
    if not f.exists() or f.stat().st_size < 1e6:
        url = f"{BASE}/{t:%Y}/{doy:03d}/{name}"
        for attempt in range(8):                      # curl 7.61 on Atos: no --retry-all-errors
            r = subprocess.run(["curl", "-s", "-f", "-L", "--netrc-file", a.netrc, "-b", str(cookies),
                                "-c", str(cookies), "-o", str(f), url])
            if r.returncode == 0 and f.exists() and f.stat().st_size > 1e6 and readable(f):
                break
            f.unlink(missing_ok=True)
            time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"download failed after 8 attempts: {name}")
    return f


lat0, lat1, lon0, lon1 = a.bbox
LAT = np.arange(lat0, lat1 + 1e-6, 0.25); LON = np.arange(lon0, lon1 + 1e-6, 0.25)
grid_lat = np.round(np.arange(-89.95, 90, 0.1), 2); grid_lon = np.round(np.arange(-179.95, 180, 0.1), 2)
ila = np.where((grid_lat >= lat0 - 0.125) & (grid_lat < lat1 + 0.125))[0]
ilo = np.where((grid_lon >= lon0 - 0.125) & (grid_lon < lon1 + 0.125))[0]
clat = np.floor((grid_lat[ila] - (lat0 - 0.125)) / 0.25).astype(int)
clon = np.floor((grid_lon[ilo] - (lon0 - 0.125)) / 0.25).astype(int)
flat = (clat[None, :] * LON.size + clon[:, None])                   # (lon, lat) like IMERG

valid = []
T = t_start + timedelta(hours=6)
while T <= t_end:
    valid.append(T); T += timedelta(hours=6)

fields = []
for T in valid:
    acc = 0.0
    for k in range(12):
        t = T - timedelta(minutes=30 * (12 - k))
        with xr.open_dataset(fetch(t), group="Grid", engine="netcdf4", decode_times=False) as h:
            p = h["precipitation"].values[0][np.ix_(ilo, ila)].astype(np.float64)   # (lon, lat) mm/h
        p[p < 0] = np.nan
        acc = acc + 0.5 * p
    s = np.bincount(flat.ravel(), weights=np.nan_to_num(acc).ravel(), minlength=LAT.size * LON.size)
    n = np.bincount(flat.ravel(), weights=np.isfinite(acc).ravel().astype(float), minlength=LAT.size * LON.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        fields.append(np.where(n >= 4, s / n, np.nan).reshape(LAT.size, LON.size))
    print(f"{T:%Y-%m-%d %H}Z  max cell {np.nanmax(fields[-1]):6.1f} mm/6h", flush=True)

xr.Dataset({"precip_6h": (("time", "lat", "lon"), np.array(fields, np.float32), {"units": "mm"})},
           coords={"time": np.array(valid, dtype="datetime64[ns]"), "lat": LAT, "lon": LON},
           attrs={"source": "GPM IMERG Final V07B half-hourly, 0.1 deg",
                  "method": "sum of 12 half-hours ending at T (x0.5 h); mean of 0.1 deg cells per 0.25 deg cell"}
           ).to_netcdf(a.out)
print("wrote", a.out)
