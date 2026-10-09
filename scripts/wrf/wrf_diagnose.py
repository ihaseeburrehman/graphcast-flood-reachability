"""Why does WRF rain more with the GraphCast change? Process diagnosis of WRF runs (control vs +delta vs a random change).

For each case and output time (6-hourly) over the storm area 49-51.5 N, 4.5-8 E:
  TCWV   column water vapour (mm)
  VIMFC  vertically integrated moisture-flux convergence, surface-500 hPa (mm per 6 h equivalent)
  U850/V850 and wind speed at 850 hPa (area mean, m/s)
  cut-off low: minimum sea-level pressure (hPa) and its position over 45-55 N, 0-15 E; minimum 500-hPa height (gpm) and position
  RAIN   6-h rain (area mean, mm)
Also writes, for the time of peak rain, maps of VIMFC, MSLP and 850-hPa wind for each case (npz) for a figure.
usage: wrf_diagnose.py <cases_dir> <out_dir> <case> [<case> ...]
"""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import xarray as xr

C, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CASES = sys.argv[3:]
G, RD = 9.80665, 287.04
AREA = dict(lat=(49.0, 51.5), lon=(4.5, 8.0))
LOWBOX = dict(lat=(45.0, 55.0), lon=(0.0, 15.0))


def clean(a):
    """WRF output carries ~1e38 garbage at a few MPI tile corners (output artefact): set to NaN."""
    a = np.asarray(a, dtype=np.float64); return np.where(np.isfinite(a) & (np.abs(a) < 1e6), a, np.nan)


def interp_p(field, p, plev):
    """Interpolate (k, j, i) field to pressure plev (Pa), linear in log p; p decreasing with k."""
    lp = np.log(p); t = np.log(plev)
    k = np.argmax(lp < t, axis=0)                       # first level above plev
    k = np.clip(k, 1, p.shape[0] - 1)
    j, i = np.indices(k.shape)
    l0, l1 = lp[k - 1, j, i], lp[k, j, i]; w = (t - l0) / (l1 - l0)
    return field[k - 1, j, i] * (1 - w) + field[k, j, i] * w


res, maps = {}, {}
for case in CASES:
    f = sorted(glob.glob(str(C / case / "wrfout_d01_*")))[0]
    d = xr.open_dataset(f)
    lat, lon = d.XLAT.values[0], d.XLONG.values[0]
    m = (lat >= AREA["lat"][0]) & (lat <= AREA["lat"][1]) & (lon >= AREA["lon"][0]) & (lon <= AREA["lon"][1])
    mb = (lat >= LOWBOX["lat"][0]) & (lat <= LOWBOX["lat"][1]) & (lon >= LOWBOX["lon"][0]) & (lon <= LOWBOX["lon"][1])
    dx = float(d.attrs["DX"]); rows = []
    rain_prev = None; best = (-1, None)
    for k in range(d.sizes["Time"]):
        p = clean((d.P[k] + d.PB[k]).values)                                   # (kz, j, i) Pa
        ph = clean((d.PH[k] + d.PHB[k]).values); z = 0.5 * (ph[1:] + ph[:-1]) / G
        q = clean(d.QVAPOR[k].values)
        U = clean(d.U[k].values); u = 0.5 * (U[:, :, 1:] + U[:, :, :-1])
        V = clean(d.V[k].values); v = 0.5 * (V[:, 1:, :] + V[:, :-1, :])
        mut = (d.MU[k] + d.MUB[k]).values                                 # dry column mass (Pa)
        c1, c2, dnw = d.C1H[k].values[:, None, None], d.C2H[k].values[:, None, None], d.DNW[k].values[:, None, None]
        dp = -(c1 * mut[None] + c2) * dnw                                  # dry-air pressure thickness of each layer (Pa)
        low = p >= 50000.0
        qu = np.nansum(np.where(low, q * u * dp, 0), axis=0) / G; qv = np.nansum(np.where(low, q * v * dp, 0), axis=0) / G
        div = (np.gradient(qu, dx, axis=1) + np.gradient(qv, dx, axis=0))   # kg m-2 s-1 (map factors ignored)
        vimfc = -div * 21600.0                                               # mm per 6 h
        tcwv = np.nansum(q * dp, axis=0) / G
        u850, v850 = interp_p(u, p, 85000.0), interp_p(v, p, 85000.0)
        z500 = interp_p(z, p, 50000.0); z500 = np.where(np.abs(z500 - np.nanmedian(z500[mb])) < 600, z500, np.nan)
        t2 = d.T2[k].values; psfc = d.PSFC[k].values; hgt = d.HGT[k].values
        mslp = psfc * np.exp(G * hgt / (RD * (t2 + 0.0065 * hgt / 2))) / 100.0
        acc = (d.RAINC[k] + d.RAINNC[k]).values
        r6 = None if rain_prev is None else float((acc - rain_prev)[m].mean()); rain_prev = acc
        jmin = np.unravel_index(np.nanargmin(np.where(mb, mslp, 1e9)), mslp.shape)
        zmin = np.unravel_index(np.nanargmin(np.where(mb & np.isfinite(z500), z500, 1e9)), z500.shape)
        t = str(d.Times.values[k].decode())
        rows.append(dict(time=t, tcwv=float(np.nanmean(tcwv[m])), vimfc=float(np.nanmean(vimfc[m])), u850=float(np.nanmean(u850[m])),
                         v850=float(np.nanmean(v850[m])), wspd850=float(np.nanmean(np.hypot(u850, v850)[m])), rain6h=r6,
                         mslp_min=float(mslp[jmin]), mslp_lat=float(lat[jmin]), mslp_lon=float(lon[jmin]),
                         z500_min=float(z500[zmin]), z500_lat=float(lat[zmin]), z500_lon=float(lon[zmin])))
        if r6 is not None and r6 > best[0]:
            best = (r6, k)
        if t.startswith("2021-07-14_18"):
            maps[case] = dict(vimfc=vimfc.astype(np.float32), mslp=mslp.astype(np.float32), u850=u850.astype(np.float32),
                              v850=v850.astype(np.float32), tcwv=tcwv.astype(np.float32), lat=lat, lon=lon)
    res[case] = rows
    print(f"== {case}")
    for r in rows:
        print(f"  {r['time']}  rain6h {r['rain6h'] if r['rain6h'] is None else round(r['rain6h'],1)}  TCWV {r['tcwv']:.1f}  "
              f"VIMFC {r['vimfc']:.1f}  850 wind {r['wspd850']:.1f} ({r['u850']:.1f},{r['v850']:.1f})  "
              f"MSLP min {r['mslp_min']:.1f} @ {r['mslp_lat']:.1f}N {r['mslp_lon']:.1f}E  "
              f"Z500 min {r['z500_min']:.0f} @ {r['z500_lat']:.1f}N {r['z500_lon']:.1f}E")
json.dump(res, open(OUT / "wrf_diagnosis.json", "w"), indent=1)
np.savez_compressed(OUT / "wrf_diagnosis_maps_20210714T18.npz",
                    **{f"{c}__{k}": v for c, dd in maps.items() for k, v in dd.items()})
