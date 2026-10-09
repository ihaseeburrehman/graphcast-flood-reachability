"""Add a GraphCast increment (or a null perturbation) to a WPS intermediate file (ungrib output, format 5).

The increment of an optimisation run is delta = (delta/sigma) x sigma_EDA at t0 (increment_tanh_a.npz holds
delta/sigma on the perturbation box, GraphCast's 37 levels; sigma = ERA5 EDA spread at t0, as in the
optimiser). It is added to the ERA5 fields of the intermediate file on the same 0.25 deg grid, in float32
(ERA5 GRIB 16-bit packing would quantise delta-T of ~1e-3 K):
  TT <- temperature, UU/VV <- u/v wind, SPECHUMD <- q (clipped >= 0), HGT <- geopotential / g (ungrib converts GEOPT to HGT),
  RH  recomputed so that it carries the q and T increments: RH' = RH (q'/q) es(T)/es(T')  (fixed p),
  surface (200100): TT <- 2 m temperature, UU/VV <- 10 m wind, RH at fixed dew point; PMSL <- msl.
w is not used by WRF; PSFC, skin, soil and SST are left unchanged.
Variants: --scale k (x k, negative = sign flip), --only q (humidity only), --null SEED: envelope-matched
random field (Gaussian-smoothed white noise, ~250 km, times the smoothed |delta| amplitude map, rescaled
to the norm of delta per variable) - same place and size as delta, random structure.
--null-match: the null's horizontal scale is 250 km in true distance in both directions (as the optimiser's B) and
its vertical smoothing is chosen so that its adjacent-level correlation matches delta's.
--micro SEED: also add white-noise temperature perturbations (sd 0.05 K, every level, whole grid) - a micro-
perturbation that samples WRF's own chaotic spread; with opt_run_dir = none only this noise is added.
usage: perturb_intermediate.py <FILE:in> <FILE:out> <opt_run_dir|none> [--scale k] [--only q] [--null SEED] [--micro SEED]
"""
import argparse, struct, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc"))
import gc_optimise_ic as G                                     # noqa: E402  (load_eda_spread, EDA_MAP)

ap = argparse.ArgumentParser()
ap.add_argument("fin"); ap.add_argument("fout"); ap.add_argument("run")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--only", choices=["q"], default=None)
ap.add_argument("--null", type=int, default=-1)
ap.add_argument("--micro", type=int, default=-1)
ap.add_argument("--null-match", action="store_true")   # nulls: 250 km in true km both ways + delta-matched vertical coherence
MICRO_SD = 0.05                                                # K
args = ap.parse_args()


# ── WPS intermediate format 5: big-endian Fortran sequential records ─────────────────────────────
def read_records(path):
    b = open(path, "rb").read(); i = 0; out = []
    while i < len(b):
        n = struct.unpack(">i", b[i:i + 4])[0]; out.append(b[i + 4:i + 4 + n]); i += 8 + n
    return out


def write_records(path, recs):
    with open(path, "wb") as f:
        for r in recs:
            f.write(struct.pack(">i", len(r))); f.write(r); f.write(struct.pack(">i", len(r)))


recs = read_records(args.fin)
fields = []                                                    # (index of header record, field, level, nx, ny, slab idx)
k = 0
while k < len(recs):
    h = recs[k + 1]
    field = h[24 + 4 + 32:24 + 4 + 32 + 9].decode().strip()
    xlvl, nx, ny, iproj = struct.unpack(">f3i", h[24 + 4 + 32 + 9 + 25 + 46:24 + 4 + 32 + 9 + 25 + 46 + 16])
    assert iproj == 0, f"expected lat-lon grid, got iproj {iproj}"
    st, la0, lo0, dla, dlo, _ = struct.unpack(">8s5f", recs[k + 2][:28])
    fields.append(dict(field=field, lvl=xlvl, nx=nx, ny=ny, slab=k + 4, la0=la0, lo0=lo0, dla=dla, dlo=dlo))
    k += 5
grid = fields[0]
lats = grid["la0"] + grid["dla"] * np.arange(grid["ny"]); lons = grid["lo0"] + grid["dlo"] * np.arange(grid["nx"])
lons180 = np.where(lons > 180, lons - 360, lons)
hdate = recs[1][:24].decode().strip()
t0 = datetime.strptime(hdate[:13], "%Y-%m-%d_%H")


def slab(f):
    return np.frombuffer(recs[f["slab"]], ">f4").reshape(f["ny"], f["nx"]).astype(np.float64)


def set_slab(f, a):
    recs[f["slab"]] = a.astype(">f4").tobytes()


def get(name, lvl):
    m = [f for f in fields if f["field"] == name and abs(f["lvl"] - lvl) < 0.5]
    return m[0] if m else None


# ── the increment delta on the GraphCast box, regridded by coordinate matching ───────────────────
run = Path(args.run)
NONE = args.run == "none"
D = np.load(run / "increment_tanh_a.npz") if not NONE else None
plat, plon = (D["plat"], np.where(D["plon"] > 180, D["plon"] - 360, D["plon"])) if not NONE else (lats[:1], lons180[:1])
eda = G.load_eda_spread([t0 - timedelta(hours=6), t0], G.EDA_DIR) if not NONE else None
LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
          650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
ii = np.array([int(np.argmin(np.abs(lats - x))) for x in plat]); jj = np.array([int(np.argmin(np.abs(lons180 - x))) for x in plon])
assert np.allclose(lats[ii], plat, atol=1e-3) and np.allclose(lons180[jj], plon, atol=1e-3), "box not on the file grid"


def delta(var):
    """delta (level?, lat, lon) on the box for a GraphCast variable, physical units."""
    kind, name = G.EDA_MAP[var]
    sp = G.ascending(eda[kind][name].sel(time=np.datetime64(t0)))
    sp = sp.sel(lat=plat, lon=D["plon"], method="nearest")
    if "level" in sp.dims:
        sp = sp.sel(level=LEVELS)
    return D[var].astype(np.float32)[0, 1] * sp.values


VARS = {"temperature": "TT", "u_component_of_wind": "UU", "v_component_of_wind": "VV", "specific_humidity": "SPECHUMD",
        "geopotential": "HGT", "2m_temperature": ("TT", 200100.0), "10m_u_component_of_wind": ("UU", 200100.0),
        "10m_v_component_of_wind": ("VV", 200100.0), "mean_sea_level_pressure": ("PMSL", 200100.0)}
if args.only == "q":
    VARS = {"specific_humidity": "SPECHUMD"}
if NONE:
    VARS = {}
dl = {v: delta(v) for v in VARS}
def adj_corr(x):                                               # mean correlation between adjacent levels
    return float(np.mean([np.corrcoef(x[k].ravel(), x[k + 1].ravel())[0, 1] for k in range(x.shape[0] - 1)
                          if x[k].std() > 0 and x[k + 1].std() > 0]))


if args.null >= 0:                                             # envelope-matched null, same norm per variable
    rng = np.random.default_rng(args.null); s = 250.0 / 27.8   # ~250 km in 0.25 deg cells (latitude)
    sx = s / np.cos(np.deg2rad(float(np.mean(plat)))) if args.null_match else s
    for v, d in dl.items():
        sv = 0.0
        if args.null_match and d.ndim == 3:                    # vertical smoothing matching delta's level coherence
            r_d = adj_corr(d); trial = np.random.default_rng(999).standard_normal(d.shape)
            env = gaussian_filter(np.abs(d), (0, s, sx))       # match after the envelope is applied
            sv = min(tuple(np.round(np.arange(0, 6.01, 0.1), 2)),
                     key=lambda x: abs(adj_corr(gaussian_filter(trial, (x, s, sx)) * env) - r_d))
        ax = (sv, s, sx) if d.ndim == 3 else (s, sx)
        n = gaussian_filter(rng.standard_normal(d.shape), ax); n /= n.std()
        nul = n * gaussian_filter(np.abs(d), (0, s, sx) if d.ndim == 3 else (s, sx))
        dl[v] = nul * np.sqrt(np.sum(d ** 2) / max(np.sum(nul ** 2), 1e-30))
        if args.null_match:
            print(f"  null {v}: vertical sigma {sv} levels, adj-level corr delta {adj_corr(d) if d.ndim == 3 else 0:.2f}"
                  f" null {adj_corr(dl[v]) if d.ndim == 3 else 0:.2f}, sigma lat/lon {s:.1f}/{sx:.1f} cells")
for v in dl:
    dl[v] = dl[v] * args.scale


def es(T):                                                     # Bolton (1980), hPa
    return 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


box = np.ix_(ii, jj)
done = []
for v, tgt in VARS.items():
    name, lv = (tgt, None) if isinstance(tgt, str) else tgt
    d = dl[v]
    for li, L in enumerate(LEVELS if d.ndim == 3 else [None]):
        lvl = L * 100.0 if L is not None else lv
        f = get(name, lvl)
        if f is None:
            continue
        a = slab(f); add = (d[li] if d.ndim == 3 else d) / (9.80665 if name == "HGT" else 1.0)
        if name in ("TT", "SPECHUMD"):                         # keep RH consistent at this level
            fr = get("RH", lvl)
            if fr is not None:
                rh = slab(fr); T = slab(get("TT", lvl)); rat = np.ones_like(rh)
                if name == "SPECHUMD":
                    q = a[box]; rat[box] = np.maximum(q + add, 0) / np.maximum(q, 1e-12)
                else:
                    rat[box] = es(T[box]) / es(T[box] + add)
                set_slab(fr, np.clip(rh * rat, 0, 100))
        a[box] = a[box] + add
        if name == "SPECHUMD":
            a = np.maximum(a, 0)
        set_slab(f, a); done.append((name, lvl))
if args.micro >= 0:                                            # micro-perturbation: white T noise everywhere
    rng_m = np.random.default_rng(10_000 + args.micro); nm = 0
    for f in fields:
        if f["field"] == "TT":
            a = slab(f); dT = MICRO_SD * rng_m.standard_normal(a.shape)
            fr = get("RH", f["lvl"])
            if fr is not None:                                 # keep q fixed: RH at the new temperature
                set_slab(fr, np.clip(slab(fr) * es(a) / es(a + dT), 0, 100))
            set_slab(f, a + dT); nm += 1
    done.append(("micro", nm))
write_records(args.fout, recs)
print(f"{args.fin} -> {args.fout}: t0 {t0}, run {run.name}, scale {args.scale}, only {args.only}, null {args.null}, micro {args.micro}; "
      f"{len(done)} slabs perturbed; fields in file: {sorted({f['field'] for f in fields})}")
