"""Flood-reachability index (FRI) pilot: GraphCast box rain, its adjoint gradient, and a prior ensemble.

Same model, increment and background-error model as the 2021 study (gc_optimise_ic.py, --cost 4dvar):
delta = sigma_EDA * B^1/2 z * taper on the perturbation box (25-72N, 45W-35E), B^1/2 = 250 km Gaussian with
unit-L2 rows, one z for both input times. For a target (box b, lead L) with 24-h rain R (E-OBS day, 06-06 UTC):
  FRI1 = (R* - R0) / |dR/dz|      (mean-value FORM / Hasofer-Lind index; P(R >= R*) ~ Phi(-FRI1) if R is linear)
t0 = 06 UTC; lead L = 1, 2, 3 is the E-OBS day starting at t0 + 24 L h (GraphCast steps 4L+1 ... 4L+4).
Boxes: 1 x 1 deg, SW corners 45..54 N, 0..14 E; GraphCast cells with centres at corner + 0, .25, .5, .75 deg.

modes
  forward  fp32 control forecast; box rain R0 (lead, box) for all boxes -> fri_forward_<t0>.npz
  grad     targets "L:lat0:lon0:Rstar,...": bf16 forward + one backward per target (|g| = |dR/dz|), optional
           --repeats (CV of |g|) and --design (fp32 forward at z = FRI1 g/|g|) -> fri_grad_<t0>.json
  prior    --members N fp32 forecasts from random z ~ N(0, I) -> fri_prior_<t0>.npz (box rain per member)
usage: fri_gradient.py --mode forward|grad|prior --t0 2021-07-12T06 --era5-dir DIR --out DIR [...]
"""
import argparse, json, sys, time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc"))
import gc_optimise_ic as G                                         # noqa: E402  (env, loaders, remat)

import numpy as np, xarray as xr, haiku as hk, jax, jax.numpy as jnp  # noqa: E402
from graphcast import graphcast, normalization, data_utils, checkpoint, casting, xarray_jax  # noqa: E402
import dataclasses  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["forward", "grad", "prior"], required=True)
ap.add_argument("--t0", required=True)
ap.add_argument("--era5-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--targets", default="", help="grad: L:lat0:lon0:Rstar[,...]")
ap.add_argument("--targets-file", default="", help="grad: JSON {t0 tag: 'L:lat0:lon0:Rstar,...'} (overrides --targets)")
ap.add_argument("--repeats", type=int, default=1, help="grad: independent bf16 forward+backward passes")
ap.add_argument("--design", action="store_true", help="grad: fp32 forward at the design point of each target")
ap.add_argument("--members", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pert-box", nargs=4, type=float, default=[25.0, 72.0, -45.0, 35.0])
ap.add_argument("--smooth-km", type=float, default=250.0)
ap.add_argument("--taper-deg", type=float, default=5.0)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
G.enable_block_remat()
log = G.log

t0 = datetime.strptime(args.t0.replace("T", " "), "%Y-%m-%d %H")
assert t0.hour == 6, "t0 must be 06 UTC (E-OBS days run 06-06 UTC)"
LEADS = (1, 2, 3)
NSTEPS = 4 * max(LEADS) + 4                                        # 16 steps = 96 h
times = [t0 - timedelta(hours=6), t0] + [t0 + timedelta(hours=6 * k) for k in range(1, NSTEPS + 1)]
win_steps = {L: list(range(4 * L, 4 * L + 4)) for L in LEADS}      # 0-based step index k (valid t0 + 6(k+1) h)
BLAT = np.arange(45, 55); BLON = np.arange(0, 15)
tag = t0.strftime("%Y%m%dT%H")

# ── data: real inputs at t0-6h, t0; forcings (TISR) from ERA5 at every step; targets = placeholders ──
d = Path(args.era5_dir)
src = dict(sl=d / "era5_sl_instant.nc", accum=d / "era5_accum_v2.nc", pl=[d / "era5_pl.grib"], eda=d)
ds_in = G.load_era5(times[:2], src)
ac = xr.open_dataset(src["accum"])
ac = G.ascending(ac.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in ac.dims else ac)
tv = ds_in.data_vars
ext = []
for t in times[2:]:
    e = ds_in.isel(time=[1]).assign_coords(time=[np.datetime64(t)], datetime=(("batch", "time"), [[np.datetime64(t)]]))
    for v in ("toa_incident_solar_radiation",):
        if v in e and v in ac:
            e[v] = ac[v].sel(time=[np.datetime64(t)]).expand_dims("batch").transpose(*e[v].dims).assign_coords(time=e.time)
    ext.append(e)
ds = xr.concat([ds_in] + ext, dim="time", data_vars="minimal", coords="minimal", compat="override")
ds = ds.assign_coords(datetime=(("batch", "time"), np.array([np.datetime64(t) for t in times]).reshape(1, -1)))

with open(G.WEIGHTS, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params, mcfg, tcfg = ckpt.params, ckpt.model_config, ckpt.task_config
stats = {k: xr.load_dataset(G.BASE / f"weights/{k}.nc").compute()
         for k in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}
inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    ds, target_lead_times=slice("6h", f"{6*NSTEPS}h"), **dataclasses.asdict(tcfg))
assert "toa_incident_solar_radiation" not in forcings or np.all(np.isfinite(forcings["toa_incident_solar_radiation"].values))
opt_vars = [v for v in G.OPT_VARS if v in inputs]
const_keys = [v for v in inputs.data_vars if v not in targets and v not in forcings]
constants = inputs[const_keys]
t1_coord = targets.coords["time"][:1]
tmpl1_spec = {v: (targets[v].dims, targets[v].isel(time=[0]).shape, targets[v].dtype) for v in targets.data_vars}
tmpl1_coords = targets.isel(time=[0]).coords
step_forcings = [forcings.isel(time=[k]).assign_coords(time=t1_coord) for k in range(NSTEPS)]
del ds, targets, ext


def predictor(bf16):
    p = graphcast.GraphCast(mcfg, tcfg)
    if bf16:
        p = casting.Bfloat16Cast(p)
    return normalization.InputsAndResiduals(p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                                            mean_by_level=stats["mean_by_level"], stddev_by_level=stats["stddev_by_level"])


def core(state, cst, frc, bf16):
    tmpl = xarray_jax.Dataset({v: xarray_jax.Variable(dd, jnp.full(sh, jnp.nan, dt)) for v, (dd, sh, dt) in tmpl1_spec.items()},
                              coords=tmpl1_coords)
    pred = predictor(bf16)(xr.merge([cst, state]), tmpl, forcings=frc)
    nxt_frame = xr.merge([pred, frc])
    nxt = (xr.concat([state, nxt_frame[list(state.keys())]], dim="time")
           .tail(time=state.sizes["time"]).assign_coords(time=state.coords["time"]))
    rain = xarray_jax.unwrap_data(pred["total_precipitation_6hr"].transpose("batch", "time", "lat", "lon"))
    return nxt, rain[0, 0] * 1000.0


s16 = hk.transform_with_state(lambda st, c, f: core(st, c, f, True))
s32 = hk.transform_with_state(lambda st, c, f: core(st, c, f, False))
rng = jax.random.PRNGKey(0)
step_fn16 = lambda st, c, f: s16.apply(params, {}, rng, st, c, f)[0]   # noqa: E731
step16 = jax.jit(step_fn16)
step32 = jax.jit(lambda st, c, f: s32.apply(params, {}, rng, st, c, f)[0])


@jax.jit
def step_bwd(state, cst, frc, g_next, g_rain):
    _, vjp = jax.vjp(lambda x: step_fn16(x, cst, frc), state)
    return vjp((g_next, g_rain))[0]


# ── grid, boxes, increment (identical construction to gc_optimise_ic.py --cost 4dvar) ──
lat = inputs.lat.values; lon = inputs.lon.values; lon180 = np.where(lon > 180, lon - 360, lon)


def lon_idx(lo0, lo1):
    i = np.where((lon180 >= lo0 - 1e-6) & (lon180 <= lo1 + 1e-6))[0]
    return i[np.argsort(lon180[i])]


BOX = {(la, lo): (np.where((lat >= la - 1e-6) & (lat <= la + 0.75 + 1e-6))[0], lon_idx(lo, lo + 0.75))
       for la in BLAT for lo in BLON}


def box_rain(rains):                                               # rains (NSTEPS, lat, lon) -> (lead, nlat, nlon)
    o = np.zeros((len(LEADS), BLAT.size, BLON.size), np.float32)
    for li, L in enumerate(LEADS):
        r24 = rains[win_steps[L]].sum(0)
        for (la, lo), (ii, jj) in BOX.items():
            o[li, la - BLAT[0], lo - BLON[0]] = r24[np.ix_(ii, jj)].mean()
    return o


PL, PO = args.pert_box[:2], args.pert_box[2:]
plat = np.where((lat >= PL[0]) & (lat <= PL[1]))[0]; plon = lon_idx(*PO)
eda = G.load_eda_spread(times[:2], d)
bounds, shapes = {}, {}
for v in opt_vars:
    da = inputs[v]; kind, name = G.EDA_MAP[v]; sp = eda[kind][name]
    if "level" in da.dims:
        sp = sp.sel(level=da.level)
    sp = sp.isel(lat=plat, lon=plon).assign_coords(time=da.time).expand_dims(batch=da.sizes["batch"])
    b = sp.transpose(*da.dims).values
    assert np.all(np.isfinite(b)), v
    bounds[v] = jnp.asarray(b, jnp.float32); shapes[v] = b.shape
blat_deg, blon_deg = lat[plat], lon180[plon]; R = 6371.0
dy = (blat_deg[:, None] - blat_deg[None, :]) * np.pi / 180 * R
S_lat = np.exp(-0.5 * (dy / args.smooth_km) ** 2); S_lat /= S_lat.sum(1, keepdims=True)
dl = (blon_deg[:, None] - blon_deg[None, :]) * np.pi / 180
S_lon = np.exp(-0.5 * (dl[None] * R * np.cos(np.deg2rad(blat_deg))[:, None, None] / args.smooth_km) ** 2)
S_lon /= S_lon.sum(2, keepdims=True)
S_lat = S_lat * (S_lat.sum(1, keepdims=True) / np.sqrt((S_lat ** 2).sum(1, keepdims=True)))
S_lon = S_lon * (S_lon.sum(2, keepdims=True) / np.sqrt((S_lon ** 2).sum(2, keepdims=True)))
S_lat, S_lon = jnp.asarray(S_lat, jnp.float32), jnp.asarray(S_lon, jnp.float32)


def taper1(x, w):
    t = np.clip(np.minimum(x - x.min(), x.max() - x) / w, 0, 1); return 0.5 * (1 - np.cos(np.pi * t))


taper = jnp.asarray(taper1(blat_deg, args.taper_deg)[:, None] * taper1(blon_deg, args.taper_deg)[None, :], jnp.float32)
smooth = lambda x: jnp.einsum("ab,...bj->...aj", S_lat, jnp.einsum("...ik,ijk->...ij", x, S_lon))  # noqa: E731
ctrl_shapes = {v: shapes[v][:1] + (1,) + shapes[v][2:] for v in opt_vars}
state0 = inputs.drop_vars(const_keys)


def perturbed_state(a, st):
    new = st.copy()
    for v in opt_vars:
        da = st[v]; full = xarray_jax.unwrap_data(da)
        dd = bounds[v] * smooth(a[v]) * taper
        idx = [slice(None)] * full.ndim
        idx[da.dims.index("lat")] = slice(plat[0], plat[-1] + 1); idx[da.dims.index("lon")] = jnp.asarray(plon)
        upd = full.at[tuple(idx)].add(dd)
        if v == "specific_humidity":
            upd = jnp.maximum(upd, 0.0)
        new[v] = xarray_jax.DataArray(upd, dims=da.dims, coords=da.coords)
    return new


init_fwd = jax.jit(perturbed_state)


@jax.jit
def init_bwd(a, st, g_state):
    _, vjp = jax.vjp(lambda x: perturbed_state(x, st), a)
    return vjp(g_state)[0]


def forecast(a, bf16=False, keep=False):
    st = init_fwd(a, state0); saved, rains = [], []
    for k in range(NSTEPS):
        if keep:
            saved.append(jax.device_get(st))
        st, r = (step16 if bf16 else step32)(st, constants, step_forcings[k]); rains.append(np.asarray(r))
    return np.stack(rains), saved, st


def gradient(a, saved, st_end, L, la, lo, shape):
    """d (box mean 24-h rain of target) / d z: backward through the saved bf16 states."""
    ii, jj = BOX[(la, lo)]; g_st = jax.tree_util.tree_map(jnp.zeros_like, st_end)
    for k in reversed(range(max(win_steps[L]) + 1)):
        g_r = np.zeros(shape, np.float32)
        if k in win_steps[L]:
            g_r[np.ix_(ii, jj)] = 1.0 / (ii.size * jj.size)
        g_st = step_bwd(jax.device_put(saved[k]), constants, step_forcings[k], g_st, jnp.asarray(g_r))
    g = init_bwd(a, state0, g_st)
    return {v: np.asarray(g[v]) for v in opt_vars}


zero = {v: jnp.zeros(ctrl_shapes[v], jnp.float32) for v in opt_vars}
log(f"FRI {args.mode}: t0 {t0}, {NSTEPS} steps, {len(BOX)} boxes, pert box {args.pert_box}, devices {jax.devices()}")

if args.mode == "forward":
    t1 = time.time(); rains, _, _ = forecast(zero)
    R0 = box_rain(rains)
    np.savez_compressed(out / f"fri_forward_{tag}.npz", R0=R0, leads=LEADS, blat=BLAT, blon=BLON, t0=str(t0),
                        rain6h_eu=rains[:, np.where((lat >= 40) & (lat <= 60))[0]][:, :, lon_idx(-10, 25)].astype(np.float16))
    log(f"forward done {time.time()-t1:.0f}s; box 24h rain max per lead {R0.reshape(3, -1).max(1).round(1)}")

elif args.mode == "prior":
    rng_np = np.random.default_rng(args.seed); Rm = []
    R0 = box_rain(forecast(zero)[0])
    for m in range(args.members):
        t1 = time.time()
        a = {v: jnp.asarray(rng_np.standard_normal(ctrl_shapes[v]), jnp.float32) for v in opt_vars}
        Rm.append(box_rain(forecast(a)[0]))
        log(f"prior member {m}: {time.time()-t1:.0f}s")
    np.savez_compressed(out / f"fri_prior_{tag}.npz", R0=R0, Rm=np.stack(Rm), leads=LEADS, blat=BLAT, blon=BLON, seed=args.seed)

elif args.mode == "grad":
    spec = json.load(open(args.targets_file)).get(tag, "") if args.targets_file else args.targets
    tg = [tuple(float(x) for x in s.split(":")) for s in spec.split(",") if s]
    if not tg:
        log(f"no targets for {tag}"); sys.exit(0)
    R0 = box_rain(forecast(zero)[0])
    res = []
    for rep in range(args.repeats):
        t1 = time.time(); rains16, saved, st_end = forecast(zero, bf16=True, keep=True)
        R0_16 = box_rain(rains16)
        for (L, la, lo, rstar) in tg:
            L, la, lo = int(L), int(la), int(lo); li = LEADS.index(L)
            t2 = time.time(); g = gradient(zero, saved, st_end, L, la, lo, rains16.shape[1:])
            gn = float(np.sqrt(sum(np.sum(x.astype(np.float64) ** 2) for x in g.values())))
            r0 = float(R0[li, la - BLAT[0], lo - BLON[0]])
            row = dict(rep=rep, lead=L, lat0=la, lon0=lo, Rstar=rstar, R0=r0, R0_bf16=float(R0_16[li, la - BLAT[0], lo - BLON[0]]),
                       gnorm=gn, gnorm_var={v: float(np.sqrt(np.sum(x.astype(np.float64) ** 2))) for v, x in g.items()},
                       FRI1=(rstar - r0) / gn if gn > 0 else float("inf"), sec=round(time.time() - t2, 1))
            if args.design and rep == 0 and gn > 0:
                beta = max(row["FRI1"], 0.0)
                a = {v: jnp.asarray(g[v] * (beta / gn), jnp.float32) for v in opt_vars}
                Rd = box_rain(forecast(a)[0]); row["R_design"] = float(Rd[li, la - BLAT[0], lo - BLON[0]])
                row["design_ratio"] = (row["R_design"] - r0) / (rstar - r0) if rstar != r0 else float("nan")
            if rep == 0:                                           # sensitivity map: q 850 hPa on the pert box
                q = g["specific_humidity"]; lev = list(inputs.level.values).index(850)
                np.save(out / f"gq850_{tag}_L{L}_{la}_{lo}.npy", q[0, 0, lev].astype(np.float16))
            res.append(row)
            log(f"rep {rep} target L{L} ({la},{lo}): R0 {r0:.1f} R* {rstar:.1f} |g| {gn:.3e} FRI1 {row['FRI1']:.2f}"
                f"{'  design ' + format(row.get('design_ratio', float('nan')), '.2f') if 'design_ratio' in row else ''}  {row['sec']}s")
        del saved
        log(f"repeat {rep} done {time.time()-t1:.0f}s")
    json.dump(dict(t0=str(t0), pert_box=args.pert_box, targets=res), open(out / f"fri_grad_{tag}.json", "w"), indent=1)
