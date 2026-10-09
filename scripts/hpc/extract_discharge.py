"""Gauge discharge from LISFLOOD-FP flood runs, on the HPC (numpy only).

Same method as the multi-model paper (extract_discharge_line_integral.py): signed normal
flux integrated exactly across a transect of the surveyed width at each gauge. header,
grid, transect_cells and line_integral are copied verbatim from that script; the gauge
orientations come from data/gauges/stations.json, computed with the paper's own
centreline_dirs (validated: reproduces the published GraphCast discharge to 1e-4 m3/s).

usage: extract_discharge.py <flood_dir> [<flood_dir> ...]    -> <flood_dir>/station_Q.csv
"""
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

S = json.load(open(Path(__file__).resolve().parents[2] / "data/gauges/stations.json"))
START = datetime.strptime(S["start_utc"], "%Y-%m-%dT%H:%M:%SZ")
INTERVAL_H = S["interval_h"]


# ── verbatim from extract_discharge_line_integral.py ────────────────────────
def header(p):
    h = {}
    with open(p) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return h


def grid(p, h):
    return np.loadtxt(p, skiprows=6, dtype=np.float32).reshape(h["nrows"], h["ncols"])


def transect_cells(h, x, y, W, px, py):
    c = h["cellsize"]
    top = h["yllcorner"] + h["nrows"] * c
    t0, t1 = -W / 2.0, W / 2.0
    breaks = {t0, t1}
    if abs(px) > 1e-12:
        ka = int(np.floor(min(x + t0 * px, x + t1 * px) / c - h["xllcorner"] / c))
        kb = int(np.ceil(max(x + t0 * px, x + t1 * px) / c - h["xllcorner"] / c))
        for k in range(ka, kb + 1):
            t = (h["xllcorner"] + k * c - x) / px
            if t0 < t < t1:
                breaks.add(float(t))
    if abs(py) > 1e-12:
        ka = int(np.floor((top - max(y + t0 * py, y + t1 * py)) / c))
        kb = int(np.ceil((top - min(y + t0 * py, y + t1 * py)) / c))
        for k in range(ka, kb + 1):
            t = (top - k * c - y) / py
            if t0 < t < t1:
                breaks.add(float(t))
    ts = sorted(breaks)
    out = []
    for ta, tb in zip(ts[:-1], ts[1:]):
        L = tb - ta
        if L <= 1e-12:
            continue
        tm = 0.5 * (ta + tb)
        cc = int(np.floor((x + tm * px - h["xllcorner"]) / c))
        rr = int(np.floor((top - (y + tm * py)) / c))
        out.append((rr, cc, L))
    return out


def line_integral(QX, QY, h, x, y, W, ux, uy, ds=None):
    px, py = -uy, ux
    nodata = h.get("nodata_value", -9999.0)
    Q, covered = 0.0, 0.0
    for rr, cc, L in transect_cells(h, x, y, W, px, py):
        if not (0 <= rr < h["nrows"] and 0 <= cc < h["ncols"]):
            continue
        qx, qy = float(QX[rr, cc]), float(QY[rr, cc])
        if qx == nodata or qy == nodata:
            continue
        Q += (qx * ux + qy * uy) * L
        covered += L
    if covered < W - 1e-6:
        raise SystemExit(f"transect at ({x:.0f},{y:.0f}) covers {covered:.2f} m of "
                         f"{W:.2f} m; outside the grid or masked")
    return Q
# ─────────────────────────────────────────────────────────────────────────────


def check_epoch(fdir):
    """Refuse a run whose forcing declares a different window start (paper's safeguard)."""
    rain = fdir / "rain_96h.nc"
    w0 = xr.open_dataset(rain).attrs.get("window_start_utc")
    if w0 is None or pd.Timestamp(w0.replace("Z", "")) != pd.Timestamp(START):
        raise SystemExit(f"{fdir.name}: forcing window_start_utc {w0} != {START}; refusing")


for arg in sys.argv[1:]:
    fdir = Path(arg)
    check_epoch(fdir)
    rd = fdir / "results"
    qxs = sorted(rd.glob("*.Qx"), key=lambda p: int(re.search(r"-(\d+)", p.name).group(1)))
    if not qxs:
        print(f"{fdir.name}: no Qx grids"); continue
    h = header(qxs[0])
    top = h["yllcorner"] + h["nrows"] * h["cellsize"]
    rows = []
    for qx in qxs:
        idx = int(re.search(r"-(\d+)", qx.name).group(1))
        qy, wd = Path(str(qx).replace(".Qx", ".Qy")), Path(str(qx).replace(".Qx", ".wd"))
        if not (qy.exists() and wd.exists()):
            continue
        QX, QY, WD = grid(qx, h), grid(qy, h), grid(wd, h)
        rec = {"Time": START + timedelta(hours=idx * INTERVAL_H)}
        for n, s in S["stations"].items():
            rec[f"{n}_Q"] = line_integral(QX, QY, h, s["x"], s["y"], s["width_m"], s["ux"], s["uy"])
            cc = int(np.floor((s["x"] - h["xllcorner"]) / h["cellsize"]))
            rr = int(np.floor((top - s["y"]) / h["cellsize"]))
            rec[f"{n}_Depth_cell"] = max(float(WD[rr, cc]), 0.0)
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(fdir / "station_Q.csv", index=False, float_format="%.4f")
    print(f"{fdir.name}: {len(d)} steps  peak Q " +
          str({n: round(float(d[f'{n}_Q'].max()), 1) for n in S["stations"]}), flush=True)
