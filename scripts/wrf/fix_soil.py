"""Repair undefined soil values in wrfinput_d01 (applied identically to every case, after real.exe).

At a few coastal/island land points the 12 km land mask has land where ERA5 has none, and metgrid leaves the
soil fields undefined (NaN or +-1e35): such a point made wrf.exe blow up at the first step whatever the
physics or time step. Bad values (non-finite, |TSLB - 273| > 100 K, soil moisture outside [0, 1]) are
replaced, layer by layer, by the mean of valid land neighbours in a growing window (3x3, 5x5, ...);
at water/lake points (soil unused) TSLB = TSK and moisture = 1; SST undefined at WRF water points ->
neighbouring valid SST, and water TSK reset to SST where they differ by > 3 K (see below).
usage: fix_soil.py <wrfinput_d01>
"""
import sys
import numpy as np
from netCDF4 import Dataset

f = Dataset(sys.argv[1], "r+")
land = f["LANDMASK"][0] > 0.5
rep = {}
for v, lo, hi in (("TSLB", 173.0, 373.0), ("SMOIS", 0.0, 1.0), ("SH2O", 0.0, 1.0), ("SMCREL", 0.0, 1.0)):
    if v not in f.variables:
        continue
    a = np.array(f[v][0], dtype=np.float64)
    n = 0
    for k in range(a.shape[0]):
        x = a[k]; bad = land & (~np.isfinite(x) | (x < lo) | (x > hi))
        good = land & ~bad
        for j, i in zip(*np.where(bad)):
            for r in range(1, 30):
                s = (slice(max(j - r, 0), j + r + 1), slice(max(i - r, 0), i + r + 1))
                if good[s].any():
                    x[j, i] = x[s][good[s]].mean(); break
            n += 1
        a[k] = x
    # water (and lake) points: soil is unused by Noah but must be defined -> TSLB = skin temperature, moisture = 1
    w = ~land
    tsk = np.array(f["TSK"][0], dtype=np.float64)
    for k in range(a.shape[0]):
        x = a[k]; badw = w & (~np.isfinite(x) | (x < lo) | (x > hi))
        x[badw] = tsk[badw] if v == "TSLB" else 1.0
        n += int(badw.sum()); a[k] = x
    f[v][0] = a; rep[v] = n
# ERA5 volumetric soil moisture refers to ERA5's soil texture; WRF uses the 16-class STATSGO/FAO map. Values
# above the WRF class's porosity (MAXSMC) or below a small dry limit make Noah produce NaN in its first step
# (then SFCLAY segfaults): clip SMOIS (and SH2O <= SMOIS) to [0.02, MAXSMC(class)] at land points.
MAXSMC = np.array([0, 0.339, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406, 0.468, 0.468, 0.439,
                   1.0, 0.200, 0.421])                                   # SOILPARM.TBL, STAS, classes 1-16
if "ISLTYP" in f.variables:
    cls = np.clip(np.array(f["ISLTYP"][0]).astype(int), 0, 16); por = MAXSMC[cls]
    sm = np.array(f["SMOIS"][0], dtype=np.float64); sh = np.array(f["SH2O"][0], dtype=np.float64)
    n_clip = 0
    for k in range(sm.shape[0]):
        x = sm[k]; m = land & ((x > por) | (x < 0.02)); n_clip += int(m.sum())
        x[m] = np.clip(x[m], 0.02, por[m]); sm[k] = x; sh[k] = np.minimum(sh[k], x)
    f["SMOIS"][0] = sm; f["SH2O"][0] = sh; rep["SMOIS_clipped"] = n_clip
# Water points (WRF sea/lake) where ERA5 has land: SST is undefined (0 K) and TSK is a hot land skin value
# (up to ~325 K at 12 UTC). Over "sea" that makes the surface layer so unstable that SFCLAY's stability
# lookup runs off its table and segfaults. SST <- mean of valid SST in a growing window; TSK <- SST wherever
# the water skin temperature departs from SST by more than 3 K.
if "SST" in f.variables:
    sst = np.array(f["SST"][0], dtype=np.float64); tsk = np.array(f["TSK"][0], dtype=np.float64)
    water = ~land; good = water & np.isfinite(sst) & (sst > 250)
    bad = water & ~good
    for j, i in zip(*np.where(bad)):
        for r in range(1, 60):
            sl = (slice(max(j - r, 0), j + r + 1), slice(max(i - r, 0), i + r + 1))
            if good[sl].any():
                sst[j, i] = sst[sl][good[sl]].mean(); break
    off = water & (np.abs(tsk - sst) > 3.0)
    tsk[off] = sst[off]
    f["SST"][0] = sst; f["TSK"][0] = tsk; rep["SST"] = int(bad.sum()); rep["TSK_water"] = int(off.sum())
# real.exe leaves uninitialised values (~1e-42 ... 1e-20, different in every run) in SMOIS/SH2O/SMCREL at a few
# hundred points; inside [0, 1] they pass the checks above but make "identical" runs differ. Make the soil state
# deterministic: water points -> SMOIS = SH2O = SMCREL = 1; land points -> SH2O = SMOIS where the soil is unfrozen
# (TSLB > 273.15 K) or SH2O is junk (< 1e-6); SMCREL = SMOIS / porosity (clipped to [0, 1]).
sm = np.array(f["SMOIS"][0], dtype=np.float64); sh = np.array(f["SH2O"][0], dtype=np.float64)
tl = np.array(f["TSLB"][0], dtype=np.float64)
cls = np.clip(np.array(f["ISLTYP"][0]).astype(int), 0, 16); por = np.maximum(MAXSMC[cls], 0.05)
n_det = 0
for k in range(sm.shape[0]):
    w_ = ~land; sm[k][w_] = 1.0; sh[k][w_] = 1.0
    fix = land & ((tl[k] > 273.15) | (sh[k] < 1e-6)); n_det += int((fix & (sh[k] != sm[k])).sum()); sh[k][fix] = sm[k][fix]
f["SMOIS"][0] = sm; f["SH2O"][0] = sh
if "SMCREL" in f.variables:
    f["SMCREL"][0] = np.where(land[None], np.clip(sm / por[None], 0, 1), 1.0)
rep["soil_deterministic"] = n_det
f.close()
print("fix_soil:", rep)
