"""Radar rain at 0.25 deg written as a pseudo optimisation run, so flood_route.sbatch routes it
exactly like GraphCast rain: isolates the flood model's own error at the forcing resolution."""
import json, numpy as np, xarray as xr
from pathlib import Path
CODE = Path("/ec/res4/hpcperm/lux0804/gc_flood_predictability/code")
out = Path("/ec/res4/scratch/lux0804/gc_flood_predictability/runs/radar025_20210713T12"); out.mkdir(parents=True, exist_ok=True)
r = xr.open_dataset(CODE / "data/radar_target/radar_6h_025deg.nc").precip_6h
print("radar times", str(r.time.values[0])[:13], "..", str(r.time.values[-1])[:13], r.sizes)
t = np.array([np.datetime64("2021-07-13T18") + np.timedelta64(6 * k, "h") for k in range(17)])
r = r.reindex(time=t)          # radar ends 16 Jul 00Z; basin rain was already 0.0 there -> zero after
print("steps beyond the radar record (set to 0):", int(np.isnan(r.values).all((1, 2)).sum()))
print("NaN cells per step", np.isnan(r.values).sum((1, 2)))
r = r.fillna(0.0)
xr.Dataset({"rain_control": r, "rain_optimised": r}, attrs={"note": "RMI radclim 6-h rain at 0.25 deg"}).to_netcdf(out / "rain_control_vs_optimised.nc")
json.dump(dict(t0="2021-07-13 12:00:00", target="alzette_2021", truth="RMI radclim (forcing)"), open(out / "summary.json", "w"))
print("wrote", out)
