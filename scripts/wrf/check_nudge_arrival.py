"""Check 2: does the nudge reach WRF intact? For a perturbed case vs the control case at t0:
(a) intermediate-file difference = the nudge as inserted; (b) met_em difference on the WRF grid (after metgrid's
horizontal interpolation) -- correlation with (a) sampled at the WRF grid points; (c) wrfinput difference
(after real.exe's vertical interpolation and rebalancing): rms of T, QVAPOR, U, V differences and correlation of
the column-integrated QVAPOR difference with the met_em column q difference.
usage: check_nudge_arrival.py <case_dir> <control_case_dir> <t0 YYYY-MM-DD_HH>
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wps_io  # noqa: E402

case, ctl, t0 = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
Fc, (la, lo) = wps_io.read(case / "fg" / f"FILE:{t0}"); F0, _ = wps_io.read(ctl / "fg" / f"FILE:{t0}")
mc = xr.open_dataset(case / f"met_em.d01.{t0}:00:00.nc"); m0 = xr.open_dataset(ctl / f"met_em.d01.{t0}:00:00.nc")
XL, XO = mc.XLAT_M.values[0], mc.XLONG_M.values[0]
ii = np.clip(np.round((XL - la[0]) / (la[1] - la[0])).astype(int), 0, la.size - 1)
jj = np.clip(np.round((((XO - lo[0]) % 360)) / (lo[1] - lo[0])).astype(int), 0, lo.size - 1)
plev = mc.PRES.values[0][:, 0, 0] if "PRES" in mc else None
for wname, mname in (("TT", "TT"), ("UU", "UU"), ("VV", "VV"), ("SPECHUMD", "SPECHUMD"), ("RH", "RH")):
    for p in (850, 500):
        if (wname, p * 100) not in Fc or mname not in mc:
            continue
        d_int = (Fc[(wname, p * 100)] - F0[(wname, p * 100)])[ii, jj]
        k = int(np.argmin(np.abs(mc.PRES.values[0][:, 0, 0] - p * 100))) if "PRES" in mc else None
        dm = (mc[mname].values[0] - m0[mname].values[0])
        dm = dm[k] if dm.ndim == 3 else dm
        if mname in ("UU",):
            dm = 0.5 * (dm[:, 1:] + dm[:, :-1])
        if mname in ("VV",):
            dm = 0.5 * (dm[1:, :] + dm[:-1, :])
        n = min(dm.shape[0], d_int.shape[0]), min(dm.shape[1], d_int.shape[1])
        a, b = d_int[:n[0], :n[1]].ravel(), dm[:n[0], :n[1]].ravel()
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        print(f"{wname:8s} {p} hPa: rms nudge (intermediate) {a.std():.3e}  rms met_em diff {b.std():.3e}  corr {r:.3f}")
wi, w0 = xr.open_dataset(case / "wrfinput_d01"), xr.open_dataset(ctl / "wrfinput_d01")
for v in ("T", "QVAPOR", "U", "V"):
    d = wi[v].values - w0[v].values
    print(f"wrfinput {v:7s} rms diff {np.sqrt(np.mean(d ** 2)):.3e}  max |diff| {np.max(np.abs(d)):.3e}")
mu = (wi.MUB + wi.MU).values[0]; dq = (wi.QVAPOR.values[0] - w0.QVAPOR.values[0]).sum(0)
qm = (mc.SPECHUMD.values[0] - m0.SPECHUMD.values[0]).sum(0) if "SPECHUMD" in mc else None
if qm is not None:
    print(f"column q: corr(wrfinput dQVAPOR summed, met_em dq summed) = {np.corrcoef(dq.ravel(), qm.ravel())[0, 1]:.3f}")
