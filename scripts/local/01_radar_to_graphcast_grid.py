"""Belgian RMI radclim radar -> 6-hour accumulations on the GraphCast 0.25 deg grid.

This is the "truth" the GraphCast optimisation is pushed towards.

Input : 5-minute radclim files, each holding the rolling 1-hour accumulation
        ending at its timestamp (int16, hundredths of a millimetre, EPSG:3812, 1 km).
        Only the full-hour files (HH:00:00) are used, so the six files ending at
        T-5h ... T tile the interval (T-6h, T] without overlap. This is the same
        convention as lib_radar.aggregate_radar_6h in the GMD companion pipeline.
Output: data/radar_target/radar_6h_025deg.nc
        precip_6h(time, lat, lon)  mm per 6 h, area mean over the radar pixels whose
                                   centres fall inside each 0.25 deg cell
        npix(lat, lon)             number of radar pixels in each cell
        Time is the END of the 6-hour interval, matching GraphCast's
        total_precipitation_6hr at valid time T.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer

RADAR_ROOT = Path("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Radar_and_Weather/"
                  "Belgium_Radar_data_2021/2021/07")
OUT = Path(__file__).resolve().parents[2] / "data/radar_target/radar_6h_025deg.nc"

# Valid times (end of each 6-hour interval) spanning the flood-producing rain.
VALID = [datetime(2021, 7, 13, 6) + timedelta(hours=6 * k) for k in range(12)]  # 13 Jul 06 .. 16 Jul 00

# GraphCast grid points (0.25 deg) in a box around Luxembourg; each point is the centre
# of a 0.25 deg cell. Latitudes ascending, as in the GraphCast pipeline.
LAT = np.arange(48.50, 51.001, 0.25)
LON = np.arange(4.50, 7.501, 0.25)


def hourly_file(t):
    return RADAR_ROOT / f"{t:%d}" / "accum1h" / "tif" / f"{t:%Y%m%d%H}0000.radclim.accum1h.tif"


def read_mm(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
    return np.clip(arr, 0, None) / 100.0


def cell_index_map(path):
    """For every radar pixel, the (lat, lon) index of the 0.25 deg cell containing it."""
    with rasterio.open(path) as ds:
        ny, nx = ds.shape
        cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
        x, y = rasterio.transform.xy(ds.transform, rows.ravel() - 0.5, cols.ravel() - 0.5, offset="center")
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(np.asarray(x), np.asarray(y))
    ilat = np.floor((lat - (LAT[0] - 0.125)) / 0.25).astype(int)
    ilon = np.floor((lon - (LON[0] - 0.125)) / 0.25).astype(int)
    ok = (ilat >= 0) & (ilat < LAT.size) & (ilon >= 0) & (ilon < LON.size)
    return ilat.reshape(ny, nx), ilon.reshape(ny, nx), ok.reshape(ny, nx)


def main():
    ilat, ilon, ok = cell_index_map(hourly_file(VALID[0]))
    flat = (ilat * LON.size + ilon)[ok]
    npix = np.bincount(flat, minlength=LAT.size * LON.size).reshape(LAT.size, LON.size)

    fields = []
    for T in VALID:
        acc = None
        for k in range(6):
            f = hourly_file(T - timedelta(hours=5 - k))
            if not f.exists():
                raise FileNotFoundError(f)
            a = read_mm(f)
            acc = a if acc is None else acc + a
        s = np.bincount(flat, weights=acc[ok], minlength=LAT.size * LON.size).reshape(LAT.size, LON.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            fields.append(np.where(npix > 0, s / npix, np.nan))
        print(f"{T:%Y-%m-%d %H}Z  domain max {np.nanmax(acc):6.1f} mm   cell max {np.nanmax(fields[-1]):6.1f} mm")

    ds = xr.Dataset(
        {"precip_6h": (("time", "lat", "lon"), np.array(fields, dtype=np.float32),
                       {"units": "mm", "long_name": "6-hour accumulation ending at time"}),
         "npix": (("lat", "lon"), npix.astype(np.int32), {"long_name": "radar pixels per cell"})},
        coords={"time": np.array(VALID, dtype="datetime64[ns]"), "lat": LAT, "lon": LON},
        attrs={"source": "RMI radclim accum1h (hourly files HH:00), 1 km, EPSG:3812",
               "method": "sum of six hourly files ending T-5h..T; area mean per 0.25 deg cell"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
