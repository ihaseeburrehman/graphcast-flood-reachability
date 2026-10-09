"""WRF rain of one transfer-test case on the GraphCast/radar 0.25 deg cells.

Reads the case's wrfout files (6-hourly), forms 6-h rain from RAINC + RAINNC differences, averages the
WRF grid points falling in each 0.25 deg cell (+-0.125 deg) over 47.5-52.5N, 3-9E, and writes
<case>/rain_025.nc (precip_6h(time, lat, lon), time = END of the 6-h interval, as the radar files). Prints
the basin rain (sum over each target's valid steps, mm and % of its truth) and the window pattern
correlation with radar for the 2021 targets.
usage: wrf_rain.py <case_dir>
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import xarray as xr

case = Path(sys.argv[1])
CODE = Path("/ec/res4/hpcperm/lux0804/gc_flood_predictability/code")
C = json.load(open(CODE / "configs/targets.json"))
files = sorted(glob.glob(str(case / "wrfout_d01_*")))
times, acc = [], []                                           # no dask needed: read file by file
for f in files:
    with xr.open_dataset(f) as d:
        times += [np.datetime64(t.decode().replace("_", "T")) for t in d.Times.values]
        acc += list((d.RAINC + d.RAINNC).values)               # mm accumulated since t0
        lat, lon = d.XLAT.values[0], d.XLONG.values[0]
times, acc = np.array(times), np.array(acc)
LAT = np.arange(47.5, 52.501, 0.25); LON = np.arange(3.0, 9.001, 0.25)
il = np.floor((lat - (LAT[0] - 0.125)) / 0.25).astype(int); jl = np.floor((lon - (LON[0] - 0.125)) / 0.25).astype(int)
ok = (il >= 0) & (il < LAT.size) & (jl >= 0) & (jl < LON.size)
flat = (il * LON.size + jl)[ok]; n = np.bincount(flat, minlength=LAT.size * LON.size)
out = []
for k in range(1, len(times)):
    d6 = (acc[k] - acc[k - 1])[ok]
    s = np.bincount(flat, weights=d6, minlength=LAT.size * LON.size)
    out.append(np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(LAT.size, LON.size))
R = xr.Dataset({"precip_6h": (("time", "lat", "lon"), np.array(out, np.float32), {"units": "mm"})},
               coords={"time": times[1:], "lat": LAT, "lon": LON}, attrs={"case": case.name})
R.to_netcdf(case / "rain_025.nc")
res = {}
for tg in ("alzette_2021", "ahr_2021", "vesdre_2021"):
    c = C[tg]; v = [np.datetime64(x) for x in c["valid"]]
    if not all(t in R.time.values for t in v):
        continue
    w = float(R.precip_6h.sel(time=v).sel(lat=c["cells"]["lat"], lon=c["cells"]["lon"], method="nearest").mean(("lat", "lon")).sum())
    tr = xr.open_dataset(CODE / c["truth"]).precip_6h.sel(time=v)
    t = float(tr.sel(lat=c["cells"]["lat"], lon=c["cells"]["lon"], method="nearest").mean(("lat", "lon")).sum())
    reg = c["region"]; tw = tr.sel(lat=slice(reg[0], reg[1]), lon=slice(reg[2], reg[3]))
    fw = R.precip_6h.sel(time=v).sel(lat=tw.lat, lon=tw.lon, method="nearest").values
    m = np.isfinite(tw.values) & np.isfinite(fw)
    res[tg] = dict(basin_mm=w, truth_mm=t, pct=100 * w / t, window_r=float(np.corrcoef(fw[m], tw.values[m])[0, 1]))
    print(f"{case.name:22s} {tg:13s} basin {w:6.1f} mm ({100*w/t:5.1f} % of {t:.1f})  window r {res[tg]['window_r']:.2f}")
json.dump(res, open(case / "rain_summary.json", "w"), indent=1)
