"""6-hour radar truth for July 2021 on the GraphCast 0.25 deg grid, wide enough for the
Alzette, Vesdre and Ahr basins.

  radclim  RMI (Belgium) radar merged with gauges; 5-min files of rolling 1-h totals,
           int16 hundredths of mm, EPSG:3812, 1 km. Covers Belgium, Luxembourg and the
           Eifel. Same convention as 01_radar_to_graphcast_grid.py.
  radklim  DWD RADKLIM RW (Germany), gauge-adjusted hourly totals, 1 km, 2-D lat/lon;
           each hourly value ends at HH:50, so the six values ending T-5h50 ... T-0h10
           make the 6-h total ending at T (a 10-minute offset, noted in the attrs).

Both are averaged over the radar pixels whose centres fall in each 0.25 deg cell (cells
with fewer than 300 valid pixels are set to NaN). Output: data/radar_target/<name>_6h_025deg_wide.nc
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RADCLIM = Path("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Radar_and_Weather/Belgium_Radar_data_2021/2021/07")
RADKLIM = Path("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Radar_and_Weather/DWD_Radar/2021/RW_2017.002_202107.nc")
VALID = [datetime(2021, 7, 13, 6) + timedelta(hours=6 * k) for k in range(12)]   # 13 Jul 06 .. 16 Jul 00
LAT = np.arange(48.50, 51.501, 0.25)
LON = np.arange(4.00, 8.501, 0.25)
MIN_PIX = 300


def cell_index(lat, lon):
    ilat = np.floor((lat - (LAT[0] - 0.125)) / 0.25).astype(int)
    ilon = np.floor((lon - (LON[0] - 0.125)) / 0.25).astype(int)
    ok = (ilat >= 0) & (ilat < LAT.size) & (ilon >= 0) & (ilon < LON.size)
    return ilat * LON.size + ilon, ok


def cell_mean(flat, ok, field):
    good = ok & np.isfinite(field)
    n = np.bincount(flat[good], minlength=LAT.size * LON.size)
    s = np.bincount(flat[good], weights=field[good], minlength=LAT.size * LON.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n >= MIN_PIX, s / n, np.nan)
    return m.reshape(LAT.size, LON.size), n.reshape(LAT.size, LON.size)


def radclim():
    f0 = RADCLIM / "13/accum1h/tif/20210713060000.radclim.accum1h.tif"
    with rasterio.open(f0) as ds:
        ny, nx = ds.shape
        rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        x, y = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel(), offset="center")
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(np.asarray(x), np.asarray(y))
    flat, ok = cell_index(lat, lon)
    out = []
    for T in VALID:
        acc = 0
        for k in range(6):
            t = T - timedelta(hours=5 - k)
            with rasterio.open(RADCLIM / f"{t:%d}/accum1h/tif/{t:%Y%m%d%H}0000.radclim.accum1h.tif") as ds:
                acc = acc + np.clip(ds.read(1).astype(np.float64), 0, None) / 100.0
        m, n = cell_mean(flat, ok, acc.ravel())
        out.append(m)
    return np.array(out), n, "RMI radclim accum1h (HH:00 files), 1 km"


def radklim():
    d = xr.open_dataset(RADKLIM)
    lat, lon = d.lat.values.ravel(), d.lon.values.ravel()
    flat, ok = cell_index(lat, lon)
    out = []
    for T in VALID:
        ends = [np.datetime64(T - timedelta(hours=5 - k) - timedelta(minutes=10)) for k in range(6)]
        acc = d.RR.sel(time=ends).sum("time", min_count=6).values.ravel()
        m, n = cell_mean(flat, ok, acc)
        out.append(m)
    return np.array(out), n, "DWD RADKLIM RW 2017.002, hourly totals ending HH:50 (6-h sums end 10 min before T)"


for name, fn in (("radclim", radclim), ("radklim", radklim)):
    p, n, src = fn()
    ds = xr.Dataset({"precip_6h": (("time", "lat", "lon"), p.astype(np.float32), {"units": "mm"}),
                     "npix": (("lat", "lon"), n.astype(np.int32))},
                    coords={"time": np.array(VALID, dtype="datetime64[ns]"), "lat": LAT, "lon": LON},
                    attrs={"source": src, "method": f"area mean per 0.25 deg cell; cells with < {MIN_PIX} pixels = NaN"})
    out = ROOT / f"data/radar_target/{name}_6h_025deg_wide.nc"
    ds.to_netcdf(out)
    print("wrote", out, "valid cells:", int(np.isfinite(p[0]).sum()), "/", p[0].size)
