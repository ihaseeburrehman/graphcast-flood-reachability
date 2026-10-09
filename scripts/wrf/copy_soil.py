"""Give every case of one initial time the identical soil state.

real.exe leaves a few soil points at its MPI tile edges uninitialised (different in every run; after fix_soil.py
they become 0.02 in one run and a real value in another), which makes two identical controls differ by ~20 % in
basin rain. The GraphCast increment never touches the soil, so all cases of a t0 take SMOIS, SH2O, SMCREL and TSLB
from one reference wrfinput (the first control, after fix_soil.py).
usage: copy_soil.py <wrfinput_d01> <soil_ref.nc>     (ref missing -> it is written from this wrfinput)
"""
import os, shutil, sys

import numpy as np
from netCDF4 import Dataset

VARS = ("SMOIS", "SH2O", "SMCREL", "TSLB")
wi, ref = sys.argv[1], sys.argv[2]
if not os.path.exists(ref):
    tmp = f"{ref}.{os.getpid()}"; shutil.copy(wi, tmp); os.replace(tmp, ref)   # atomic: concurrent controls are safe
    print(f"copy_soil: wrote reference {ref}"); sys.exit(0)
with Dataset(ref) as r, Dataset(wi, "r+") as f:
    assert np.array_equal(r["LANDMASK"][:], f["LANDMASK"][:]), "reference has another land mask"
    n = {v: int(np.sum(np.asarray(r[v][:]) != np.asarray(f[v][:]))) for v in VARS if v in f.variables}
    for v in n:
        f[v][:] = r[v][:]
print(f"copy_soil: {wi} <- {ref}; values replaced {n}")
