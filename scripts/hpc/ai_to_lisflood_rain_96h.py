#!/usr/bin/env python3
"""
ai_to_lisflood_rain.py
-----------------------
Convert AI-NWP 6-hourly precipitation forecasts (GraphCast / FuXi / AIFS) into a
LISFLOOD-FP `dynamicrainfile` NetCDF on the 10 m Alzette grid (EPSG:2169), matching
the schema of the WRF-forced rain_10m.nc *exactly* so it is a drop-in forcing.

Usage:
    python ai_to_lisflood_rain.py <model> <forecast_dir> <dem_asc> <out_nc>
        model        : graphcast | fuxi | aifs
        forecast_dir : dir of rapid-cycle forecast_<YYYYMMDDTHH>.nc files
        dem_asc      : target DEM .asc (header defines the 10 m EPSG:2169 grid)
        out_nc       : output rain NetCDF

Window: 2021-07-13 00:00 -> 2021-07-17 00:00 (96 h, 6-hourly => 17 layers:
16 accumulations at 0..90 h plus a closing zero at 96 h).
6 h precipitation accumulation valid at <stamp> (filename = true valid time).

GraphCast/FuXi: var `total_precipitation_6hr` on a regular 0.25 deg lat/lon grid.
AIFS:           var `tp` on a reduced-Gaussian point cloud (1-D `values`).
All precip assumed in metres -> x1000 = mm. A wide bbox around Luxembourg keeps
the small Alzette basin well inside the source convex hull (no edge zero-fill).
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata

# Option B window. The hydraulic start grid (13_july_18hr.wd) is the model state
# at 13 July 18:00 UTC, so the forcing must begin there: applying rain from
# 13 July 00:00 would replay the 18 h already contained in the start state.
START = pd.Timestamp("2021-07-13 18:00")
END = pd.Timestamp("2021-07-17 18:00")
STEP_H = 6
BBOX = dict(lat_min=48.5, lat_max=51.0, lon_min=4.5, lon_max=8.0)
MM_PER_M = 1000.0
PROJ4_2169 = ("+proj=tmerc +lat_0=49.8333333333333 +lon_0=6.16666666666667 "
              "+k=0.9996 +x_0=80000 +y_0=100000 +ellps=GRS80 +units=m +no_defs")


def read_asc_grid(path):
    """Read .asc header -> cell-centre coords. ``y`` is DESCENDING (north first)
    to match the DEM's .asc row order and the WRF rain_10m.nc. LISFLOOD-FP maps
    the rain grid by coordinate origin; an ascending y makes it index rows out of
    bounds, crashing the GPU solver with cudaErrorIllegalAddress."""
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            key, val = f.readline().split()
            hdr[key.lower()] = float(val)
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    xll, yll, cs = hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"]
    x = xll + (np.arange(ncols) + 0.5) * cs
    y = (yll + (np.arange(nrows) + 0.5) * cs)[::-1]   # DESCENDING (north first)
    return x, y


def load_points(model, fpath):
    """Return (lon, lat, precip_mm) 1-D arrays inside BBOX for one forecast file."""
    ds = xr.open_dataset(fpath, decode_times=False)
    if model in ("graphcast", "fuxi"):
        da = ds["total_precipitation_6hr"].squeeze(drop=True)   # -> (lat, lon)
        lat, lon = ds["lat"].values, ds["lon"].values
        LON, LAT = np.meshgrid(lon, lat)
        lon1, lat1, p1 = LON.ravel(), LAT.ravel(), np.asarray(da.values).ravel()
    elif model == "aifs":
        lat1 = ds["latitude"].values.ravel()
        lon1 = ds["longitude"].values.ravel()
        p1 = np.asarray(ds["tp"].values).ravel()
    else:
        raise ValueError(f"unknown model: {model}")
    ds.close()
    lon1 = np.where(lon1 < 0, lon1 + 360.0, lon1)            # normalise to 0..360
    m = ((lat1 >= BBOX["lat_min"]) & (lat1 <= BBOX["lat_max"]) &
         (lon1 >= BBOX["lon_min"]) & (lon1 <= BBOX["lon_max"]) & np.isfinite(p1))
    return lon1[m], lat1[m], p1[m] * MM_PER_M


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: ai_to_lisflood_rain.py <model> <forecast_dir> <dem_asc> <out_nc>")
    model, fdir, dem, out = sys.argv[1:5]

    x, y = read_asc_grid(dem)
    nx, ny = x.size, y.size
    Xt, Yt = np.meshgrid(x, y)
    target = np.column_stack([Xt.ravel(), Yt.ravel()])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2169", always_xy=True)

    valid_times = pd.date_range(START + pd.Timedelta(hours=STEP_H), END, freq=f"{STEP_H}h")
    layers = []
    for t in valid_times:
        stamp = t.strftime("%Y%m%dT%H")
        fpath = os.path.join(fdir, f"forecast_{stamp}.nc")
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        lon, lat, pmm = load_points(model, fpath)
        xs, ys = tr.transform(lon, lat)
        grid = griddata((xs, ys), pmm, target, method="linear", fill_value=0.0).reshape(ny, nx)
        grid = np.clip(grid, 0.0, None).astype(np.float32)
        layers.append(grid)
        print(f"  {model} {stamp}: src_pts={pmm.size:4d} src_max={pmm.max():6.1f}mm "
              f"grid_mean={grid.mean():6.2f}mm", flush=True)

    # TIMESTAMP CONVENTION (corrected 2026-09-09).
    # forecast_<stamp>.nc holds the accumulation VALID AT <stamp>, i.e. covering the
    # SIX HOURS ENDING at <stamp>. LISFLOOD-FP reads a dynamicrainfile value as a rate
    # held FORWARD from its timestamp to the next one, so writing the accumulation at
    # <stamp> applies it over <stamp> -> <stamp>+6 h, one interval too late. The
    # accumulation must therefore be stamped at the START of its own interval.
    # The earlier version of this script zero-padded at t=0 and stamped each
    # accumulation at its valid time; the runs analysed in the manuscript were built
    # that way and are one interval late (see the manuscript's limitations section).
    # Verified by closing dV/dt + Qout against the applied rainfall over the basin
    # mask: RMSE 8.69 mm per 6 h as previously written, 0.17 mm with this correction.
    rain = np.concatenate([np.stack(layers, axis=0),
                           np.zeros((1, ny, nx), np.float32)], axis=0)
    thours = (np.arange(rain.shape[0]) * STEP_H).astype(np.float64)  # [0,6,...,96]
    print(f"  EVENT total domain-mean = {rain.mean(axis=(1, 2)).sum():.1f} mm "
          f"(WRF After-DA ref ~87.6)")

    # NOTE: define time, x, y, rainfall_depth in THIS order (as plain variables,
    # not coords) so the NetCDF declares dimensions time, x, y -- identical to the
    # WRF rain_10m.nc. LISFLOOD-FP reads the rain grid dimensions positionally, so
    # a y-before-x order triggers "NetCDF: Start+count exceeds dimension bound".
    out_ds = xr.Dataset(
        {
            "time": ("time", thours, {"units": "hour", "long_name": "time", "axis": "T"}),
            "x": ("x", x, {"units": "m", "long_name": "x", "axis": "X", "spatial_ref": "EPSG:2169"}),
            "y": ("y", y, {"units": "m", "long_name": "y", "axis": "Y", "spatial_ref": "EPSG:2169"}),
            "rainfall_depth": (("time", "y", "x"), rain,
                               {"units": "mm", "long_name": "rainfall_depth"}),
        },
        attrs={"crs": "EPSG:2169", "spatial_ref": PROJ4_2169, "source_model": model,
               "time_stamp_convention": "interval_start",
               "window_start_utc": START.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "window_end_utc": END.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "time_stamp_note": ("each layer is the accumulation over "
                                   "[t, t+6h); the final layer is a closing zero")},
    )
    out_ds.to_netcdf(out, format="NETCDF4",
                     encoding={"time": {"_FillValue": None, "dtype": "float64"},
                               "x": {"_FillValue": None, "dtype": "float64"},
                               "y": {"_FillValue": None, "dtype": "float64"},
                               "rainfall_depth": {"_FillValue": None, "dtype": "float32",
                                                  "zlib": True, "complevel": 4}},
                     unlimited_dims=["time"])
    print(f"  WROTE {out}  shape={rain.shape}")


if __name__ == "__main__":
    main()
