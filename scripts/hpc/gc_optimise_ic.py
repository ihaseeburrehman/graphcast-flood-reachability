"""Optimise GraphCast initial conditions so that its rain over the Alzette matches radar.

Pilot for the flood-predictability study. For one initialisation time t0 it:
  1. runs the control forecast (ERA5 initial state, unchanged);
  2. adds a bounded increment to the atmospheric input fields and adjusts it by
     gradient descent (Adam) so that GraphCast's 6-hour rain over a target box
     matches the radar in the flood-producing window;
  3. writes the control and optimised rain, the loss history and the increment.

The increment is  delta = bound * tanh(a),  so |delta| <= bound everywhere. For the
pilot, bound = BOUND_FRAC * climatological std per level (a placeholder); the study
will use the ERA5 ensemble (EDA) spread instead. Precipitation, solar radiation and
the static fields are never perturbed, and the increment is confined to a box
upstream of Luxembourg.

usage: gc_optimise_ic.py --t0 2021-07-13T12 --iters 20 --out <dir>
"""
import argparse, dataclasses, json, os, sys, time, warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# GPU scatter-adds (GraphCast's message passing) are non-deterministic: the same z gives slightly
# different J. --xla_gpu_deterministic_ops makes one forecast take > 1.5 h, so instead the 4dvar
# optimiser measures this noise and works with it (see the L-BFGS block).
if os.environ.get("GC_DETERMINISTIC"):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
sys.path.insert(0, "/usr/local/apps/ai-models/0.49/env-jax/lib/python3.11/site-packages")

import numpy as np, xarray as xr, haiku as hk, jax, jax.numpy as jnp
from graphcast import graphcast, normalization, data_utils, checkpoint, casting, xarray_jax
from graphcast import typed_graph_net


def enable_block_remat():
    """Recompute every graph-network block (encoder, 16 processor layers, decoder)
    in the backward pass instead of storing its internals.

    Without this, one GraphCast step stores the concatenated edge inputs of the
    mesh-to-grid decoder (3.1 M edges x 1536 features, 9.6 GB) and similar tensors;
    backward through 3 steps then needs 55 GB. deep_typed_graph_net looks these
    constructors up on the typed_graph_net module at call time, so patching the
    module attributes is enough; the graphcast package itself is not modified.
    """
    for name in ("InteractionNetwork", "GraphMapFeatures"):
        orig = getattr(typed_graph_net, name)
        if getattr(orig, "_remat_wrapped", False):
            continue

        def make(orig=orig):
            def wrapped(*args, **kwargs):
                return hk.remat(orig(*args, **kwargs))
            wrapped._remat_wrapped = True
            return wrapped
        setattr(typed_graph_net, name, make())

# ── inputs (read-only; the pipeline of the multi-model paper) ───────────────
BASE = Path("/ec/res4/scratch/lux0804/ai_rerun_2026")
WEIGHTS = BASE / "weights/GraphCast_1979-2017_precip_in_out.npz"
SL_NC = BASE / "data/2021_event/era5_mars/era5_sl_instant.nc"          # 20 Jun - 20 Jul
ACCUM_NC = BASE / "v2_tisr_fix/data/2021_event_accum_v2.nc"             # corrected tp6h + hourly tisr
PL_NC = BASE / "data/2021_floodwindow/era5_mars/era5_pl.nc"             # 12 - 17 Jul
EDA_DIR = Path("/ec/res4/scratch/lux0804/gc_flood_predictability/era5")   # ERA5 EDA spread (MARS type=es)
PL_GRIB_EARLY = EDA_DIR / "era5_pl_0707_0711.grib"                       # ERA5 PL 7-11 Jul (MARS oper an)
EDA_MAP = {  # GraphCast name -> (file, grib short name)
    "2m_temperature": ("sfc", "t2m"), "mean_sea_level_pressure": ("sfc", "msl"),
    "10m_u_component_of_wind": ("sfc", "u10"), "10m_v_component_of_wind": ("sfc", "v10"),
    "temperature": ("pl", "t"), "geopotential": ("pl", "z"), "u_component_of_wind": ("pl", "u"),
    "v_component_of_wind": ("pl", "v"), "vertical_velocity": ("pl", "w"), "specific_humidity": ("pl", "q")}
SAVE_LEVELS = [250, 500, 700, 850]                                       # levels written to increment.nc

SL_RENAME = {"t2m": "2m_temperature", "msl": "mean_sea_level_pressure",
             "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
             "sst": "sea_surface_temperature", "siconc": "sea_ice_cover",
             "tcwv": "total_column_water_vapour", "z": "geopotential_at_surface", "lsm": "land_sea_mask"}
PL_RENAME = {"z": "geopotential", "q": "specific_humidity", "t": "temperature",
             "u": "u_component_of_wind", "v": "v_component_of_wind", "w": "vertical_velocity"}

# Fields the optimiser may change. Precipitation input, TISR and statics are frozen.
OPT_VARS = ("2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "temperature", "geopotential", "u_component_of_wind",
            "v_component_of_wind", "vertical_velocity", "specific_humidity")

CODE_ROOT = Path(__file__).resolve().parents[2]
TARGETS = CODE_ROOT / "configs/targets.json"     # basin, truth, target window, perturbation box


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def gpu_mem():
    try:
        s = jax.devices()[0].memory_stats()
        return f"GPU in use {s['bytes_in_use']/2**30:.1f} GB, peak {s['peak_bytes_in_use']/2**30:.1f} GB"
    except Exception:
        return "GPU memory n/a"


def ascending(d):
    return d.isel(lat=slice(None, None, -1)) if float(d.lat[0]) > float(d.lat[-1]) else d


def era5_sources(key):
    """ERA5 input files for an event: the 2021 inputs of the multi-model paper, or a
    directory written by prep_event.sbatch."""
    if key == "2021":
        return dict(sl=SL_NC, accum=ACCUM_NC, pl=[PL_GRIB_EARLY, PL_NC], eda=EDA_DIR)
    d = EDA_DIR / key
    return dict(sl=d / "era5_sl_instant.nc", accum=d / "era5_accum_v2.nc", pl=[d / "era5_pl.grib"], eda=d)


def load_era5(times, src):
    sl = xr.open_dataset(src["sl"]).rename({k: v for k, v in SL_RENAME.items()})
    sl = sl.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in sl.dims else sl
    for v in ("geopotential_at_surface", "land_sea_mask"):
        if "time" in sl[v].dims:
            sl[v] = sl[v].isel(time=0, drop=True)
    ac = xr.open_dataset(src["accum"])
    ac = ac.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in ac.dims else ac
    def tidy_pl(pl):
        pl = pl.drop_vars(["valid_time", "step", "number"], errors="ignore")
        pl = pl.rename({k: v for k, v in PL_RENAME.items() if k in pl})
        pl = pl.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in pl.dims else pl
        for old in ("plev", "isobaricInhPa"):
            if old in pl.dims:
                pl = pl.rename({old: "level"})
        if float(pl.level.max()) > 2000:
            pl = pl.assign_coords(level=pl.level / 100)
        pl = pl.assign_coords(level=pl.level.astype(np.int32)).sortby("level")
        return ascending(pl)

    t = np.array([np.datetime64(x) for x in times])
    parts = []
    for f in src["pl"]:                                                    # GRIB and/or NetCDF pieces
        d = (xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""}) if str(f).endswith(".grib")
             else xr.open_dataset(f))
        d = tidy_pl(d)
        keep = np.isin(d.time.values, t)
        if keep.any():
            parts.append(d.isel(time=np.where(keep)[0]))
    pl = xr.concat([p[sorted(p.data_vars)] for p in parts], dim="time").sortby("time")
    pl = pl.isel(time=np.unique(pl.time.values, return_index=True)[1])
    sl, ac = ascending(sl), ascending(ac)
    sl_i, ac_i, pl_i = sl.sel(time=t).compute(), ac.sel(time=t).compute(), pl.sel(time=t).compute()
    sl_i["sea_surface_temperature"] = sl_i["sea_surface_temperature"].fillna(0.0)
    sl_i["sea_ice_cover"] = sl_i["sea_ice_cover"].fillna(0.0)
    m = xr.merge([sl_i, ac_i, pl_i], compat="override").expand_dims("batch")
    return m.assign_coords(datetime=(("batch", "time"), t.reshape(1, -1)))


def load_eda_spread(input_times, eda_dir):
    """ERA5 EDA spread at the two input times, on the GraphCast grid (lat ascending)."""
    out = {}
    for kind in ("sfc", "pl"):
        d = xr.open_dataset(eda_dir / f"era5_eda_spread_{kind}.grib", engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
        d = d.drop_vars(["valid_time", "step", "number", "surface"], errors="ignore")
        d = d.rename({"latitude": "lat", "longitude": "lon"})
        if "isobaricInhPa" in d.dims:
            d = d.rename({"isobaricInhPa": "level"})
            d = d.assign_coords(level=d.level.astype(np.int32))
        d = ascending(d).sel(time=[np.datetime64(t) for t in input_times])
        missing = [str(t) for t in input_times if np.datetime64(t) not in d.time.values]
        if missing:
            raise ValueError(f"EDA spread missing for {missing}; download it first")
        out[kind] = d.load()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="alzette_2021", help="entry in configs/targets.json")
    ap.add_argument("--t0", required=True, help="initialisation, e.g. 2021-07-13T12")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bound-frac", type=float, default=0.3,
                    help="placeholder bound as a fraction of the climatological std per level")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bound", choices=["eda", "clim"], default="eda",
                    help="eda: k x ERA5 EDA spread (default); clim: bound-frac x climatological std")
    ap.add_argument("--eda-k", type=float, default=1.0, help="bound = eda_k x EDA spread")
    ap.add_argument("--reg-weight", type=float, default=1.0,
                    help="weight of the regional term (whole radar window) relative to the basin box")
    ap.add_argument("--lr-decay", choices=["cosine", "none"], default="cosine")
    ap.add_argument("--smooth-km", type=float, default=250.0,
                    help="horizontal correlation length of the increment (Gaussian, km); 0 = no smoothing")
    ap.add_argument("--cost", choices=["mse", "4dvar"], default="mse",
                    help="mse: bounded tanh increment, squared-error loss (v1); 4dvar: J = 1/2|z|^2 + "
                         "1/2 sum((F-O)/r)^2 with delta = sigma_EDA * B^1/2 z, one increment for both input times")
    ap.add_argument("--obs-err", nargs=2, type=float, default=[4.0, 0.0], metavar=("R0_MM", "R1_FRAC"),
                    help="4dvar: rain observation error r = R0 + R1 * observed (default: fixed 4 mm per 6 h)")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="4dvar: stop when the cost falls by less than this fraction over 4 iterations")
    ap.add_argument("--gtol", type=float, default=1e-2,
                    help="4dvar: stop when |grad J| < gtol * |grad J at the first iteration|")
    ap.add_argument("--init-seed", type=int, default=-1,
                    help="4dvar: >=0 starts from a random z (J_b = 30) with this seed instead of z = 0")
    ap.add_argument("--init-jb", type=float, default=30.0, help="4dvar: J_b of the random starting z (with --init-seed)")
    ap.add_argument("--max-restarts", type=int, default=5, help="4dvar: L-BFGS memory resets before stopping")
    ap.add_argument("--truth-file", default=None, help="override the target's truth file (precip_6h(time, lat, lon), mm)")
    ap.add_argument("--fp32", action="store_true", help="run GraphCast in float32 (no bfloat16 cast)")
    ap.add_argument("--bg-ens-member", type=int, default=0, help="background = ERA5 + IFS ENS member m perturbation")
    ap.add_argument("--save-truth", default=None, help="forward: write the regional-window rain as a truth file (twin)")
    ap.add_argument("--noise-reps", type=int, default=6, help="4dvar: forward repeats to estimate the J noise")
    ap.add_argument("--init-zrms", type=float, default=0.0,
                    help="4dvar: >0 starts from smooth random z with this rms (increment ~ init_zrms x sigma)")
    ap.add_argument("--pert-box", nargs=4, type=float, default=None, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="override the perturbation box of the target (degrees, lon in -180..180)")
    ap.add_argument("--extend-to", default="2021-07-17T18",
                    help="after optimising, forecast (no gradients) to this time for the flood model")
    ap.add_argument("--taper-deg", type=float, default=5.0,
                    help="cosine taper width at the edges of the perturbation box (degrees)")
    ap.add_argument("--no-block-remat", action="store_true",
                    help="store block internals (faster, but backward through >1 step needs >40 GB)")
    ap.add_argument("--ref-run", default=None, help="ensic: optimisation run whose increment the IFS perturbations are projected on")
    ap.add_argument("--n-random", type=int, default=200, help="ensic: random B^1/2 z increments for the null")
    ap.add_argument("--holdout", default="", help="comma list of targets whose box cells are removed from J_o and "
                    "evaluated afterwards (spatial hold-out), e.g. ahr_2021,vesdre_2021")
    ap.add_argument("--n-noise", type=int, default=20, help="robust: noisy copies of the increment")
    ap.add_argument("--noise-sigma", type=float, default=0.3, help="robust: noise amplitude (x sigma_EDA, B-smoothed)")
    ap.add_argument("--mode", choices=["optimise", "forward", "members", "ensic", "gradtest", "robust", "balance"], default="optimise",
                    help="forward: control forecast only; members: forecasts from ERA5 + (EDA member - EDA mean)")
    ap.add_argument("--truth-shift", nargs=2, type=float, default=[0.0, 0.0], metavar=("DLAT", "DLON"),
                    help="null test: move the truth field by (dlat, dlon) degrees")
    ap.add_argument("--truth-transform", choices=["flipns", "flipew", "rot180", "rollns", "rollew"], default=None,
                    help="null test: rearrange the observed rain inside the regional window (same values, same "
                         "number of observations, storm elsewhere): mirror N-S / E-W, rotate 180 deg, or wrap-shift "
                         "by 5 cells (~140 km) N-S / 6 cells (~110 km) E-W")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(TARGETS))[args.target]
    TARGET_VALID = [datetime.strptime(v.replace("T", " "), "%Y-%m-%d %H") for v in cfg["valid"]]
    TARGET_LAT, TARGET_LON = tuple(cfg["box"][:2]), tuple(cfg["box"][2:])
    BASIN = cfg["cells"]
    pbox = args.pert_box if args.pert_box else cfg["pert_box"]
    PERT_LAT, PERT_LON = tuple(pbox[:2]), tuple(pbox[2:])
    RADAR_NC = Path(args.truth_file) if args.truth_file else CODE_ROOT / cfg["truth"]
    log(f"target {args.target}: truth {cfg['truth_name']}, {len(TARGET_VALID)} target steps")
    if not args.no_block_remat:
        enable_block_remat()
    t0 = datetime.strptime(args.t0.replace("T", " "), "%Y-%m-%d %H")
    target_valid = [tv for tv in TARGET_VALID if tv > t0]
    nsteps = int((target_valid[-1] - t0).total_seconds() // 21600)          # steps the loss needs
    t_end = max(datetime.strptime(args.extend_to.replace("T", " "), "%Y-%m-%d %H"), target_valid[-1])
    nsteps_all = int((t_end - t0).total_seconds() // 21600)                # steps for the flood model
    times = [t0 - timedelta(hours=6), t0] + [t0 + timedelta(hours=6 * k) for k in range(1, nsteps_all + 1)]
    log(f"t0={t0:%Y-%m-%d %H}Z  optimised steps={nsteps}  forecast steps={nsteps_all} (to {t_end:%Y-%m-%d %H}Z)  "
        f"devices={jax.devices()}")

    # ── model ───────────────────────────────────────────────────────────────
    with open(WEIGHTS, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params, mcfg, tcfg = ckpt.params, ckpt.model_config, ckpt.task_config
    stats = {k: xr.load_dataset(BASE / f"weights/{k}.nc").compute()
             for k in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}

    # One GraphCast step, exactly as autoregressive.Predictor does it, but without its
    # scan: we roll out step by step ourselves so that the state between steps can be
    # kept in host RAM. GPU memory is then one step's worth whatever the lead time.
    # --fp32 = mixed precision: J (line search, stopping, reported rain) from float32 forwards, which
    # are ~10x less noisy (sd(J) 0.13-0.17 vs ~2 in bfloat16); gradients from bfloat16 passes, since a
    # float32 backward step does not fit in 40 GB.
    def one_step_predictor(bf16=True):
        p = graphcast.GraphCast(mcfg, tcfg)
        if bf16:
            p = casting.Bfloat16Cast(p)
        return normalization.InputsAndResiduals(p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                                                mean_by_level=stats["mean_by_level"],
                                                stddev_by_level=stats["stddev_by_level"])

    # ── data ────────────────────────────────────────────────────────────────
    log("loading ERA5 ...")
    src = era5_sources(cfg["era5"])
    ds = load_era5(times, src)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h", f"{6*nsteps_all}h"), **dataclasses.asdict(tcfg))
    opt_vars = [v for v in OPT_VARS if v in inputs]
    log(f"inputs {dict(inputs.sizes)}  optimising {len(opt_vars)} fields")

    # Split as autoregressive.Predictor does: constants are neither targets nor forcings.
    const_keys = [v for v in inputs.data_vars if v not in targets and v not in forcings]
    constants = inputs[const_keys]
    # A one-step target template (structure only) and per-step forcings, all carrying the
    # first target time, as autoregressive.Predictor does inside its scan.
    t1_coord = targets.coords["time"][:1]
    tmpl1_spec = {v: (targets[v].dims, targets[v].isel(time=[0]).shape, targets[v].dtype) for v in targets.data_vars}
    tmpl1_coords = targets.isel(time=[0]).coords
    step_forcings = [forcings.isel(time=[k]).assign_coords(time=t1_coord) for k in range(nsteps_all)]
    if args.mode == "balance":                    # ERA5 at the forecast valid times (large-scale check), 40-60 N
        era5_valid = targets[["geopotential", "temperature", "mean_sea_level_pressure"]].sel(
            level=[500, 850]).isel(lat=np.where((targets.lat.values >= 40) & (targets.lat.values <= 60))[0]).compute()
    del ds, targets

    def one_step_core(state, cst, frc, bf16=True):
        tmpl = xarray_jax.Dataset(
            {v: xarray_jax.Variable(d, jnp.full(sh, jnp.nan, dt)) for v, (d, sh, dt) in tmpl1_spec.items()},
            coords=tmpl1_coords)
        pred = one_step_predictor(bf16)(xr.merge([cst, state]), tmpl, forcings=frc)
        nxt_frame = xr.merge([pred, frc])
        nxt = (xr.concat([state, nxt_frame[list(state.keys())]], dim="time")
               .tail(time=state.sizes["time"]).assign_coords(time=state.coords["time"]))
        rain = xarray_jax.unwrap_data(pred["total_precipitation_6hr"].transpose("batch", "time", "lat", "lon"))
        return nxt, rain[0, 0] * 1000.0                                   # rain (lat, lon) mm

    one_step = hk.transform_with_state(lambda st, c, f: one_step_core(st, c, f, True))
    one_step32 = hk.transform_with_state(lambda st, c, f: one_step_core(st, c, f, False))
    rng = jax.random.PRNGKey(0)

    def step_fn(state, cst, frc):
        return one_step.apply(params, {}, rng, state, cst, frc)[0]

    step_fwd16 = jax.jit(step_fn)
    def step_fn32(st, c, f):
        return one_step32.apply(params, {}, rng, st, c, f)[0]
    step_fwd = jax.jit(step_fn32) if args.fp32 else step_fwd16
    # forward-mode tangent of one float32 step (no stored activations: fits where the fp32 adjoint does not)
    jvp_step = jax.jit(lambda s, t, c, f: jax.jvp(lambda x: step_fn32(x, c, f), (s,), (t,)))

    @jax.jit
    def step_bwd(state, cst, frc, g_next, g_rain):
        _, vjp = jax.vjp(lambda x: step_fn(x, cst, frc), state)
        return vjp((g_next, g_rain))[0]

    lat = inputs.lat.values; lon = inputs.lon.values
    lon180 = np.where(lon > 180, lon - 360, lon)
    plat = np.where((lat >= PERT_LAT[0]) & (lat <= PERT_LAT[1]))[0]
    plon = np.where((lon180 >= PERT_LON[0]) & (lon180 <= PERT_LON[1]))[0]
    # Longitudes are handled in -180..180 (lon180) and every lon index list is ordered west
    # to east, so boxes that cross 0 deg (e.g. Valencia) work; GraphCast stores 0..360.
    def lon_idx(lo0, lo1):
        i = np.where((lon180 >= lo0 - 1e-6) & (lon180 <= lo1 + 1e-6))[0]
        return i[np.argsort(lon180[i])]
    tlat = np.where((lat >= TARGET_LAT[0] - 1e-6) & (lat <= TARGET_LAT[1] + 1e-6))[0]
    tlon = lon_idx(*TARGET_LON)
    alat = np.where(np.isin(np.round(lat, 2), BASIN["lat"]))[0]
    alon = lon_idx(min(BASIN["lon"]), max(BASIN["lon"]))
    lead_idx = [int((tv - t0).total_seconds() // 21600) - 1 for tv in target_valid]
    wlat = np.arange(tlat[0] - 8, tlat[-1] + 9)
    wlon = lon_idx(lon180[tlon[0]] - 2.0, lon180[tlon[-1]] + 2.0)             # saved window (+-2 deg)
    tsl, wsl, asl = np.ix_(tlat, tlon), np.ix_(wlat, wlon), np.ix_(alat, alon)

    radar = xr.open_dataset(RADAR_NC)["precip_6h"]
    if any(args.truth_shift):
        dla, dlo = args.truth_shift
        radar = radar.assign_coords(lat=np.round(radar.lat + dla, 4), lon=np.round(radar.lon + dlo, 4))
        log(f"NULL TEST: truth moved by {dla:+.2f} deg lat, {dlo:+.2f} deg lon")
    # Regional window: every radar cell (2.5 deg x 3 deg around Luxembourg). The storm as a
    # whole must match radar, so rain cannot simply be pulled onto the basin box.
    reg = cfg["region"]
    rlat = np.where((lat >= max(reg[0], float(radar.lat.min())) - 1e-6) & (lat <= min(reg[1], float(radar.lat.max())) + 1e-6))[0]
    rlon = lon_idx(max(reg[2], float(radar.lon.min())), min(reg[3], float(radar.lon.max())))
    rsl = np.ix_(rlat, rlon)
    obs_win = np.stack([radar.sel(time=np.datetime64(tv)).sel(lat=lat[rlat], lon=lon180[rlon], method="nearest").values
                        for tv in target_valid])
    win_ok = np.isfinite(obs_win)
    obs_win = np.nan_to_num(obs_win)
    obs = np.stack([radar.sel(time=np.datetime64(tv), lat=lat[tlat], lon=lon180[tlon], method="nearest").values
                    for tv in target_valid])                                         # (T, ny, nx) mm
    box_ok = np.isfinite(obs)                    # truth may not cover every box cell (e.g. RADKLIM at borders)
    obs = np.nan_to_num(obs)
    obs_alz = np.array([radar.sel(time=np.datetime64(tv), lat=BASIN["lat"], lon=BASIN["lon"]).mean().item()
                        for tv in target_valid])
    if args.truth_transform:
        tf = {"flipns": lambda x: x[:, ::-1, :], "flipew": lambda x: x[:, :, ::-1], "rot180": lambda x: x[:, ::-1, ::-1],
              "rollns": lambda x: np.roll(x, 5, 1), "rollew": lambda x: np.roll(x, 6, 2)}[args.truth_transform]
        obs_win, win_ok = np.ascontiguousarray(tf(obs_win)), np.ascontiguousarray(tf(win_ok))
        ri, rj = list(rlat), list(rlon)
        sub = lambda x, la, lo: x[(slice(None),) + np.ix_([ri.index(i) for i in la], [rj.index(j) for j in lo])]
        obs, box_ok = sub(obs_win, tlat, tlon), sub(win_ok, tlat, tlon)
        obs_alz = sub(obs_win, alat, alon).mean(axis=(1, 2))
        log(f"NULL TEST: observed rain rearranged inside the window ({args.truth_transform})")
    holdout = {}
    if args.holdout:
        allc = json.load(open(TARGETS))
        main_cells = {(round(float(x), 2), round(float(y), 2)) for x in BASIN["lat"] for y in BASIN["lon"]}
        LA, LO = np.meshgrid(lat[rlat], lon180[rlon], indexing="ij")
        for name in args.holdout.split(","):
            hb = allc[name]["box"]
            m = (LA >= hb[0] - 1e-6) & (LA <= hb[1] + 1e-6) & (LO >= hb[2] - 1e-6) & (LO <= hb[3] + 1e-6)
            m &= ~np.array([[(round(float(a), 2), round(float(b), 2)) in main_cells for b in lon180[rlon]] for a in lat[rlat]])
            win_ok = win_ok & ~m[None]
            hc = allc[name]["cells"]
            holdout[name] = dict(lat=np.where(np.isin(np.round(lat, 2), hc["lat"]))[0],
                                 lon=lon_idx(min(hc["lon"]), max(hc["lon"])),
                                 truth=float(sum(radar.sel(time=np.datetime64(tv)).sel(lat=hc["lat"], lon=hc["lon"], method="nearest").mean().item()
                                                 for tv in target_valid)), n_masked=int(m.sum()))
            log(f"HOLD-OUT {name}: {int(m.sum())} window cells removed from J_o; truth {holdout[name]['truth']:.1f} mm")
    log(f"truth basin 6h rain {np.round(obs_alz,1)} mm  (sum {obs_alz.sum():.1f}); "
        f"regional window {len(rlat)}x{len(rlon)} cells, {int(np.isfinite(obs_win).sum())} valid")

    # Bound per field, restricted to the perturbation box: |delta| <= bound everywhere.
    bounds, shapes = {}, {}
    eda = load_eda_spread(times[:2], src["eda"]) if args.bound == "eda" else None
    for v in opt_vars:
        da = inputs[v]
        box = da.isel(lat=plat, lon=plon)
        if args.bound == "eda":
            kind, name = EDA_MAP[v]
            sp = eda[kind][name]
            if "level" in da.dims:
                sp = sp.sel(level=da.level)
            sp = sp.isel(lat=plat, lon=plon).assign_coords(time=da.time).expand_dims(batch=da.sizes["batch"])
            b = args.eda_k * sp.transpose(*da.dims).values
        else:
            sd = stats["stddev_by_level"][v]
            if "level" in da.dims:
                sd = sd.sel(level=da.level)
            b = args.bound_frac * sd.broadcast_like(box).transpose(*da.dims).values
        if not np.all(np.isfinite(b)) or b.shape != box.shape:
            raise ValueError(f"bad bound for {v}: shape {b.shape} vs {box.shape}")
        bounds[v] = jnp.asarray(b, jnp.float32)
        shapes[v] = b.shape
    log((f"bound = {args.eda_k} x EDA spread" if args.bound == "eda" else f"bound = {args.bound_frac} x clim std")
        + f", smoothing {args.smooth_km:.0f} km, taper {args.taper_deg} deg")

    state0 = inputs.drop_vars(const_keys)

    def ens_perturbations():
        """IFS ENS initial perturbations pf_m - cf at t0 on GraphCast's grid and levels (see ensic)."""
        ev = "2021" if cfg["era5"] == "2021" else cfg["era5"].split("_")[0]
        E = EDA_DIR.parent / "ens_ic"

        def ens(kind, typ):
            d = xr.open_dataset(E / f"{ev}_{typ}_{kind}.grib", engine="cfgrib", backend_kwargs={"indexpath": ""})
            d = d.rename({"latitude": "lat", "longitude": "lon"})
            if "isobaricInhPa" in d.dims:
                d = d.rename({"isobaricInhPa": "level"})
            d = d.sel(time=np.datetime64(t0)).drop_vars(["valid_time", "step", "surface", "time"], errors="ignore")
            return ascending(d.assign_coords(lon=np.round(d.lon.values % 360, 3)).sortby("lon"))
        cf = {k: ens(k, "cf") for k in ("pl", "sfc")}
        pf = {k: ens(k, "pf") for k in ("pl", "sfc")}
        elat, elon = cf["sfc"].lat.values, cf["sfc"].lon.values
        ilat = np.array([int(np.argmin(np.abs(lat - x))) for x in elat])
        ilon = np.array([int(np.argmin(np.abs(lon - x))) for x in elon])
        glev = inputs.level.values.astype(float)
        ens_lev = cf["pl"].level.values.astype(float)
        lev_ok = (glev >= ens_lev.min()) & (glev <= ens_lev.max())
        pert = {}
        for v in opt_vars:
            kind, name = EDA_MAP[v]
            if name == "w":
                continue
            dv = (pf[kind][name] - cf[kind][name]).transpose("number", ...)
            if kind == "pl":
                dv = dv.sortby("level")
                x = np.log(dv.level.values.astype(float))
                a_ = dv.values                                            # number, level, lat, lon
                out_ = np.zeros(a_.shape[:1] + (glev.size,) + a_.shape[2:], np.float32)
                for i, p in enumerate(glev):
                    if lev_ok[i]:
                        j = int(np.clip(np.searchsorted(x, np.log(p)), 1, x.size - 1))
                        w = (np.log(p) - x[j - 1]) / (x[j] - x[j - 1])
                        out_[:, i] = (1 - w) * a_[:, j - 1] + w * a_[:, j]
                pert[v] = out_
            else:
                pert[v] = dv.values.astype(np.float32)
        nmem = next(iter(pert.values())).shape[0]
        log(f"IFS ENS {ev}: {nmem} members, {len(ens_lev)} levels -> {int(lev_ok.sum())} GraphCast levels, "
            f"fields {list(pert)}")
        return pert, ilat, ilon, lev_ok, nmem, elat, elon, ev, glev

    def with_ens_pert(st, pert, ilat, ilon, m):
        new = st.copy()
        for v in pert:
            da = st[v]
            upd = np.array(da.values)
            upd[..., ilat[:, None], ilon[None, :]] += pert[v][m]
            if v == "specific_humidity":
                upd = np.maximum(upd, 0.0)
            new[v] = (da.dims, upd.astype(da.dtype))
        return new

    if args.bg_ens_member:
        # background = ERA5 + IFS ENS perturbation of member m (alternative starting point / twin truth)
        _p = ens_perturbations()
        state0 = with_ens_pert(state0, _p[0], _p[1], _p[2], args.bg_ens_member - 1)
        log(f"background = ERA5 + IFS ENS member {args.bg_ens_member} perturbation ({_p[7]})")

    # Spatially correlated increment, as in 4D-Var's background-error covariance: the
    # optimiser controls a field a; what is added is bound * tanh(S a) * taper, where S is a
    # Gaussian smoother with length scale smooth_km (true distances, so the longitudinal
    # width shrinks with cos(latitude)) and the taper brings the increment smoothly to zero
    # at the edges of the perturbation box. Without S, Adam moves every grid point by a
    # similar fraction and the increment is grid-scale noise.
    blat_deg, blon_deg = lat[plat], lon180[plon]
    R = 6371.0
    if args.smooth_km > 0:
        dy = (blat_deg[:, None] - blat_deg[None, :]) * np.pi / 180 * R
        S_lat = np.exp(-0.5 * (dy / args.smooth_km) ** 2)
        S_lat /= S_lat.sum(1, keepdims=True)
        dl = (blon_deg[:, None] - blon_deg[None, :]) * np.pi / 180
        coslat = np.cos(np.deg2rad(blat_deg))[:, None, None]
        S_lon = np.exp(-0.5 * (dl[None] * R * coslat / args.smooth_km) ** 2)
        S_lon /= S_lon.sum(2, keepdims=True)                     # (nlat, nlon_out, nlon_in)
        if args.cost == "4dvar":
            # B^1/2 rows with unit L2 norm: white z gives an increment with variance sigma^2
            S_lat = S_lat * (S_lat.sum(1, keepdims=True) / np.sqrt((S_lat ** 2).sum(1, keepdims=True)))
            S_lon = S_lon * (S_lon.sum(2, keepdims=True) / np.sqrt((S_lon ** 2).sum(2, keepdims=True)))
        S_lat, S_lon = jnp.asarray(S_lat, jnp.float32), jnp.asarray(S_lon, jnp.float32)

    def edge_taper(x, lo, hi, w):
        if w <= 0:
            return np.ones_like(x)
        t = np.clip(np.minimum(x - lo, hi - x) / w, 0, 1)
        return 0.5 * (1 - np.cos(np.pi * t))
    taper = jnp.asarray(edge_taper(blat_deg, blat_deg.min(), blat_deg.max(), args.taper_deg)[:, None]
                        * edge_taper(blon_deg, blon_deg.min(), blon_deg.max(), args.taper_deg)[None, :],
                        jnp.float32)

    def smooth(x):                                               # x (..., lat, lon) on the box
        if args.smooth_km <= 0:
            return x
        y = jnp.einsum("...ik,ijk->...ij", x, S_lon)
        return jnp.einsum("ab,...bj->...aj", S_lat, y)

    def increment(a_v, v):
        if args.cost == "4dvar":                                 # a_v has a length-1 time axis: same z both times
            return bounds[v] * smooth(a_v) * taper
        return bounds[v] * jnp.tanh(smooth(a_v)) * taper

    ctrl_shapes = {v: (shapes[v][:1] + (1,) + shapes[v][2:]) if args.cost == "4dvar" else shapes[v] for v in opt_vars}
    r0, r1 = args.obs_err

    def perturbed_state(a, st):
        new = st.copy()
        for v in opt_vars:
            da = st[v]
            full = xarray_jax.unwrap_data(da)
            d = increment(a[v], v)
            lat_ax, lon_ax = da.dims.index("lat"), da.dims.index("lon")
            idx = [slice(None)] * full.ndim
            idx[lat_ax] = slice(plat[0], plat[-1] + 1)
            idx[lon_ax] = jnp.asarray(plon)
            updated = full.at[tuple(idx)].add(d)
            if v == "specific_humidity":
                updated = jnp.maximum(updated, 0.0)
            new[v] = xarray_jax.DataArray(updated, dims=da.dims, coords=da.coords)
        return new

    init_fwd = jax.jit(perturbed_state)

    @jax.jit
    def init_bwd(a, st, g_state):
        _, vjp = jax.vjp(lambda x: perturbed_state(x, st), a)
        return vjp(g_state)[0]

    ntarget = box_ok.sum()
    if args.cost == "4dvar":
        inside = set(tlat) <= set(rlat) and set(tlon) <= set(rlon)
        log(f"J_o over the regional window only ({int(win_ok.sum())} observations); basin box inside window: {inside}")
        if not inside:
            raise ValueError("basin box is not inside the regional window")

    def evaluate(a, need_grad, n=None):
        """Roll out n steps (default: the optimised window), compute the loss; if need_grad,
        backpropagate step by step."""
        n = nsteps if n is None else n
        st = init_fwd(a, state0)
        saved, rains = [], []
        keep = need_grad and not args.fp32
        for k in range(n):
            if keep:
                saved.append(jax.device_get(st))                  # state in host RAM
            st, r = step_fwd(st, constants, step_forcings[k])
            rains.append(np.asarray(r))
        rains = np.stack(rains)                                   # (nsteps, lat, lon)
        box = rains[lead_idx][(slice(None),) + tsl]
        win = rains[lead_idx][(slice(None),) + rsl]
        nwin = win_ok.sum()
        if args.cost == "4dvar":                                  # J_o with observation error r = r0 + r1*obs
            rb, rw = r0 + r1 * obs, r0 + r1 * obs_win
            loss = float(0.5 * np.sum((((win - obs_win) / rw) * win_ok) ** 2))   # window only: basin cells once
        else:
            loss = float(np.sum(((box - obs) * box_ok) ** 2) / ntarget + args.reg_weight * np.sum(((win - obs_win) * win_ok) ** 2) / nwin)
        alz = rains[lead_idx][(slice(None),) + asl].mean(axis=(1, 2))
        if not need_grad:
            return loss, alz, rains, None
        if args.fp32:                                             # gradient from a bfloat16 pass (mixed precision)
            st = init_fwd(a, state0); rg = []
            for k in range(nsteps):
                saved.append(jax.device_get(st))
                st, r = step_fwd16(st, constants, step_forcings[k]); rg.append(np.asarray(r))
            rg = np.stack(rg)
            box, win = rg[lead_idx][(slice(None),) + tsl], rg[lead_idx][(slice(None),) + rsl]
        g_st = jax.tree_util.tree_map(jnp.zeros_like, st)
        for k in reversed(range(nsteps)):
            g_r = np.zeros(rains.shape[1:], np.float32)
            if k in lead_idx:
                j = lead_idx.index(k)
                if args.cost == "4dvar":
                    g_r[rsl] = (win[j] - obs_win[j]) * win_ok[j] / rw[j] ** 2
                else:
                    g_r[tsl] = 2.0 * (box[j] - obs[j]) * box_ok[j] / ntarget
                    g_r[rsl] += args.reg_weight * 2.0 * (win[j] - obs_win[j]) * win_ok[j] / nwin
            g_st = step_bwd(jax.device_put(saved[k]), constants, step_forcings[k], g_st, jnp.asarray(g_r))
            saved[k] = None
        return loss, alz, rains, init_bwd(a, state0, g_st)

    def pattern_corr(rains):
        w = rains[lead_idx][(slice(None),) + rsl][win_ok]
        return float(np.corrcoef(w, obs_win[win_ok])[0, 1])

    a = {v: jnp.zeros(ctrl_shapes[v], jnp.float32) for v in opt_vars}

    if args.mode == "members":
        # ERA5 + (member - mean) at both input times for the same fields the optimiser may
        # change (globally, no bound): the spread of outcomes across real analysis uncertainty.
        D = EDA_DIR / "eda_members_2021"
        pert = {}
        for kind, names in (("sfc", None), ("pl", None)):
            def opn(t):
                d = xr.open_dataset(D / f"eda_{t}_{kind}.grib", engine="cfgrib", backend_kwargs={"indexpath": ""})
                d = d.drop_vars(["valid_time", "step", "surface"], errors="ignore").rename({"latitude": "lat", "longitude": "lon"})
                if "isobaricInhPa" in d.dims:
                    d = d.rename({"isobaricInhPa": "level"}).assign_coords(level=lambda x: x.level.astype(np.int32))
                return ascending(d).sel(time=[np.datetime64(x) for x in times[:2]])
            an, em = opn("an"), opn("em")
            em = em.drop_vars("number", errors="ignore")
            for v in opt_vars:
                k, name = EDA_MAP[v]
                if k == kind:
                    dv = an[name] - em[name]
                    if "level" in inputs[v].dims:
                        dv = dv.sel(level=inputs[v].level)
                    pert[v] = dv.load()
        nmem = pert[opt_vars[0]].sizes["number"]

        def with_delta(st, m):
            new = st.copy()
            for v in opt_vars:
                da = st[v]
                d = pert[v].isel(number=m).assign_coords(time=da.time).expand_dims(batch=da.sizes["batch"])
                upd = da.values + d.transpose(*da.dims).values
                if v == "specific_humidity":
                    upd = np.maximum(upd, 0.0)
                new[v] = (da.dims, upd.astype(da.dtype))
            return new

        rains_all, sums = [], []
        for m in [-1] + list(range(nmem)):              # -1 = ERA5 control
            st0 = state0 if m < 0 else with_delta(state0, m)
            st = st0; rains = []
            for k in range(nsteps):
                st, r = step_fwd(st, constants, step_forcings[k]); rains.append(np.asarray(r))
            rains = np.stack(rains)
            rains_all.append(rains[(slice(None),) + wsl])
            alz = rains[lead_idx][(slice(None),) + asl].mean(axis=(1, 2))
            sums.append(float(alz.sum()))
            log(f"member {m:2d}: basin {np.round(alz,1)} sum {alz.sum():5.1f} mm  window r={pattern_corr(rains):.2f}")
        vt = [np.datetime64(t0 + timedelta(hours=6 * k)) for k in range(1, nsteps + 1)]
        xr.Dataset({"rain": (("member", "time", "lat", "lon"), np.stack(rains_all))},
                   coords={"member": np.arange(-1, nmem), "time": vt, "lat": lat[wlat], "lon": lon180[wlon]},
                   attrs={"t0": str(t0), "member_-1": "ERA5 control", "units": "mm per 6 h"}
                   ).to_netcdf(out / "members_rain.nc")
        json.dump(dict(t0=str(t0), target=args.target, basin_sum_by_member=sums,
                       truth_basin_mm=obs_alz.round(2).tolist()), open(out / "members_summary.json", "w"), indent=1)
        log(f"done -> {out}")
        return

    if args.mode == "ensic":
        # Physics check with the IFS ENS. (1) GraphCast from ERA5 + (pf_m - cf), the operational
        # ENS initial perturbation of member m at t0 (11 pressure levels interpolated in log p to
        # GraphCast's levels, zero outside 50-1000 hPa and for w; both input times): does GraphCast
        # respond to each perturbation as IFS did? (2) Projection of each perturbation onto the
        # optimised increment of --ref-run in the sigma_EDA-whitened metric, sum (d/s)(p/s) cos(lat),
        # and the same for 200 random increments s B^1/2 z (the null). IFS member rain is merged later.
        pert, ilat, ilon, lev_ok, nmem, elat, elon, ev, glev = ens_perturbations()

        # (2) projections onto the optimised increment and onto random B^1/2 z increments
        ref = np.load(Path(args.ref_run) / "increment_tanh_a.npz")
        bl = [int(np.argmin(np.abs(elat - x))) for x in lat[plat]]
        bo = [int(np.argmin(np.abs(elon - x))) for x in lon[plon]]
        wlat = np.cos(np.deg2rad(lat[plat]))[:, None]
        P, D = {}, {}
        for v in pert:
            sig = np.asarray(bounds[v])[0, 1]                             # ([level], lat, lon) at t0
            p_ = pert[v][(slice(None),) * (pert[v].ndim - 2) + np.ix_(bl, bo)] / sig
            d_ = ref[v].astype(np.float32)[0, 1]
            if p_.ndim == 4:
                p_, d_ = p_[:, lev_ok], d_[lev_ok]
            P[v], D[v] = p_ * wlat, d_
        dot = lambda d: np.array([sum(float(np.sum(P[v][m] * d[v])) for v in P) for m in range(nmem)])
        proj = dot(D)
        dn = np.sqrt(sum(float(np.sum(D[v] ** 2 * wlat)) for v in D))
        pn = np.sqrt(np.array([sum(float(np.sum(P[v][m] ** 2 / wlat)) for v in P) for m in range(nmem)]))
        rng_np = np.random.default_rng(0); proj_rand = []
        for k in range(args.n_random):
            R = {}
            for v in P:
                shp = (1, 1) + D[v].shape if D[v].ndim == 2 else (1, 1, int(lev_ok.sum())) + D[v].shape[1:]
                R[v] = np.asarray(smooth(jnp.asarray(rng_np.standard_normal(shp), jnp.float32)) * taper)[0, 0]
            rn = np.sqrt(sum(float(np.sum(R[v] ** 2 * wlat)) for v in R))
            proj_rand.append((dot(R) * dn / rn).tolist())                # same norm as the real increment
        log(f"projection onto increment: mean {proj.mean():.3g}, sd {proj.std():.3g}; random sd "
            f"{np.std(proj_rand):.3g}")

        # (1) GraphCast forecasts from ERA5 + IFS perturbation
        def with_pert(st, m):
            return with_ens_pert(st, pert, ilat, ilon, m)
        sums, steps = [], []
        for m in [-1] + list(range(nmem)):
            st = state0 if m < 0 else with_pert(state0, m); rains = []
            for k in range(nsteps):
                st, r = step_fwd(st, constants, step_forcings[k]); rains.append(np.asarray(r))
            alz = np.stack(rains)[lead_idx][(slice(None),) + asl].mean(axis=(1, 2))
            sums.append(float(alz.sum())); steps.append(alz.round(2).tolist())
            if m < 3 or m % 10 == 0:
                log(f"member {m:2d}: GraphCast basin {np.round(alz,1)} sum {alz.sum():5.1f} mm  proj "
                    f"{proj[m] if m >= 0 else 0:.3g}")
        json.dump(dict(t0=str(t0), target=args.target, ref_run=str(args.ref_run), ens=ev, members=list(range(1, nmem + 1)),
                       graphcast_basin_sum_control=sums[0], graphcast_basin_sum=sums[1:], graphcast_basin_steps=steps[1:],
                       proj=proj.tolist(), cos=(proj / (pn * dn)).tolist(), incr_norm=dn, pert_norm=pn.tolist(),
                       proj_random=proj_rand, levels_used=glev[lev_ok].tolist(), truth_basin_mm=obs_alz.round(2).tolist()),
                  open(out / "ensic_summary.json", "w"), indent=1)
        log(f"done -> {out}")
        return

    if args.mode == "robust":
        # Is the optimised increment an adversarial (fragile) perturbation? Forecasts from ERA5 + variants of
        # the --ref-run increment: (i) as optimised; (ii) with B-smoothed random noise of noise_sigma x sigma_EDA
        # added (n_noise draws); (iii) noise alone (5 draws); (iv) ablations removing field groups.
        ref = np.load(Path(args.ref_run) / "increment_tanh_a.npz")
        base = {v: jnp.asarray(ref[v].astype(np.float32)) * bounds[v] for v in opt_vars}      # delta = (delta/sigma) sigma

        def add_delta(st, D):
            new = st.copy()
            for v in opt_vars:
                da = st[v]; full = xarray_jax.unwrap_data(da)
                lat_ax, lon_ax = da.dims.index("lat"), da.dims.index("lon")
                idx = [slice(None)] * full.ndim
                idx[lat_ax] = slice(plat[0], plat[-1] + 1); idx[lon_ax] = jnp.asarray(plon)
                upd = full.at[tuple(idx)].add(D[v])
                if v == "specific_humidity":
                    upd = jnp.maximum(upd, 0.0)
                new[v] = xarray_jax.DataArray(upd, dims=da.dims, coords=da.coords)
            return new
        add_j = jax.jit(add_delta)

        def run(D):
            st = add_j(state0, D); rr = []
            for k in range(nsteps):
                st, r = step_fwd(st, constants, step_forcings[k]); rr.append(np.asarray(r))
            rr = np.stack(rr); w_ = rr[lead_idx][(slice(None),) + rsl]
            jo_ = float(0.5 * np.sum((((w_ - obs_win) / (r0 + r1 * obs_win)) * win_ok) ** 2))
            b_ = float(rr[lead_idx][(slice(None),) + asl].mean(axis=(1, 2)).sum())
            return dict(basin_mm=b_, basin_pct=100 * b_ / float(obs_alz.sum()), chi2=2 * jo_ / int(win_ok.sum()),
                        window_corr=pattern_corr(rr))
        zero = {v: jnp.zeros_like(base[v]) for v in opt_vars}
        keep = lambda vs: {v: (base[v] if v in vs else zero[v]) for v in opt_vars}
        surf = [v for v in opt_vars if v.startswith(("2m", "10m", "mean_sea"))]
        res = {"control": run(zero), "optimised": run(base),
               "no_vertical_velocity": run(keep([v for v in opt_vars if v != "vertical_velocity"])),
               "no_humidity": run(keep([v for v in opt_vars if v != "specific_humidity"])),
               "humidity_only": run(keep(["specific_humidity"])),
               "no_surface": run(keep([v for v in opt_vars if v not in surf])),
               "dynamics_only_zTuv": run(keep(["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"]))}
        for k_, v_ in res.items():
            log(f"{k_:22s} basin {v_['basin_pct']:5.1f} %  chi2 {v_['chi2']:.2f}  window r {v_['window_corr']:.2f}")
        rng_r = np.random.default_rng(11)

        def noise():
            return {v: bounds[v] * smooth(jnp.asarray(rng_r.standard_normal(ctrl_shapes[v]), jnp.float32)) * taper * args.noise_sigma
                    for v in opt_vars}
        noisy, alone = [], []
        for i in range(args.n_noise):
            n_ = noise(); noisy.append(run({v: base[v] + n_[v] for v in opt_vars}))
            if i < 5:
                alone.append(run(n_))
        pc = lambda L, key: np.percentile([x[key] for x in L], [5, 50, 95]).round(2).tolist()
        log(f"optimised + noise ({args.n_noise}x, {args.noise_sigma} sigma): basin % 5/50/95 = {pc(noisy, 'basin_pct')}, "
            f"chi2 {pc(noisy, 'chi2')}")
        log(f"noise alone (5x): basin % 5/50/95 = {pc(alone, 'basin_pct')}, chi2 {pc(alone, 'chi2')}")
        json.dump(dict(t0=str(t0), ref_run=str(args.ref_run), noise_sigma=args.noise_sigma, variants=res,
                       optimised_plus_noise=noisy, noise_alone=alone, truth_basin_mm=float(obs_alz.sum())),
                  open(out / "robust_summary.json", "w"), indent=1)
        log(f"done -> {out}")
        return

    if args.mode == "balance":
        # Referee checks on the optimised increment (--ref-run): (1) hydrostatic consistency - geopotential
        # increment recomputed from the T, q and MSLP increments (hypsometric integration upward from 1000 hPa);
        # forecasts with the balanced and with no geopotential increment. (2) Out-of-sample large-scale check:
        # RMSE of Z500, T850 and MSLP against ERA5 at the target valid times, 40-60 N, 10 W-20 E.
        ref = np.load(Path(args.ref_run) / "increment_tanh_a.npz")
        base = {v: np.asarray(ref[v].astype(np.float32)) * np.asarray(bounds[v]) for v in opt_vars}
        Rd, lev = 287.05, inputs.level.values.astype(float)                    # levels ascending (hPa)
        T0 = inputs["temperature"].isel(lat=plat, lon=plon).values             # (batch, time, level, lat, lon)
        q0 = inputs["specific_humidity"].isel(lat=plat, lon=plon).values
        dT, dq = base["temperature"], base["specific_humidity"]
        dTv = dT * (1 + 0.608 * q0) + 0.608 * T0 * dq
        i1000 = int(np.where(lev == 1000)[0][0])
        dphi = np.zeros_like(dT)
        dphi[:, :, i1000] = Rd * T0[:, :, i1000] * (1 + 0.608 * q0[:, :, i1000]) * base["mean_sea_level_pressure"] / 1e5
        for k in range(i1000 - 1, -1, -1):                                   # upward: p decreasing
            dphi[:, :, k] = dphi[:, :, k + 1] + Rd * 0.5 * (dTv[:, :, k] + dTv[:, :, k + 1]) * np.log(lev[k + 1] / lev[k])
        trop = (lev >= 100)[None, None, :, None, None]
        zo, zb = base["geopotential"] * trop, dphi * trop
        resid = float(np.sqrt(np.sum((zo - zb) ** 2) / np.sum(zb ** 2)))
        k500 = int(np.where(lev == 500)[0][0])
        log(f"hydrostatic check: |dphi_opt - dphi_bal| / |dphi_bal| = {resid:.2f}; |dphi_opt|/|dphi_bal| = "
            f"{np.sqrt(np.sum(zo ** 2) / np.sum(zb ** 2)):.2f}; max |dZ500| opt {np.abs(zo[:, :, k500]).max() / 9.80665:.2f} m, "
            f"balanced {np.abs(zb[:, :, k500]).max() / 9.80665:.2f} m")
        elat = era5_valid.lat.values; elon = era5_valid.lon.values; elon180 = np.where(elon > 180, elon - 360, elon)
        jj = np.where((elon180 >= -10) & (elon180 <= 20))[0]; ii = np.where(np.isin(lat, elat))[0]
        wts = np.cos(np.deg2rad(elat))[:, None]
        def add_delta(st, D):
            new = st.copy()
            for v in opt_vars:
                da = st[v]; full = xarray_jax.unwrap_data(da)
                idx = [slice(None)] * full.ndim
                idx[da.dims.index("lat")] = slice(plat[0], plat[-1] + 1); idx[da.dims.index("lon")] = jnp.asarray(plon)
                upd = full.at[tuple(idx)].add(jnp.asarray(D[v]))
                if v == "specific_humidity":
                    upd = jnp.maximum(upd, 0.0)
                new[v] = xarray_jax.DataArray(upd, dims=da.dims, coords=da.coords)
            return new
        add_j = jax.jit(add_delta)
        def run(D):
            st = add_j(state0, D); rr, err = [], {"z500": [], "t850": [], "msl": []}
            for k in range(nsteps):
                st, r = step_fwd(st, constants, step_forcings[k]); rr.append(np.asarray(r))
                if k in lead_idx:
                    for key, v, lv in (("z500", "geopotential", 500), ("t850", "temperature", 850), ("msl", "mean_sea_level_pressure", None)):
                        f = np.asarray(xarray_jax.unwrap_data(st[v]))[0, -1]
                        o = era5_valid[v].isel(time=k).values[0] if "batch" in era5_valid[v].dims else era5_valid[v].isel(time=k).values
                        if lv is not None:
                            f = f[list(lev).index(lv)]; o = o[[500, 850].index(lv)]
                        d_ = (f[np.ix_(ii, jj)] - o[:, jj]) * (1 / 9.80665 if key == "z500" else (0.01 if key == "msl" else 1))
                        err[key].append(float(np.sqrt(np.sum(wts * d_ ** 2) / np.sum(wts * np.ones_like(d_)))))
            rr = np.stack(rr); b_ = float(rr[lead_idx][(slice(None),) + asl].mean(axis=(1, 2)).sum())
            return dict(basin_pct=100 * b_ / float(obs_alz.sum()), window_corr=pattern_corr(rr),
                        rmse={k_: float(np.mean(v_)) for k_, v_ in err.items()})
        zero = {v: np.zeros_like(base[v]) for v in opt_vars}
        bal = dict(base); bal["geopotential"] = dphi
        noz = dict(base); noz["geopotential"] = np.zeros_like(dphi)
        res = {"control": run(zero), "optimised": run(base), "balanced_geopotential": run(bal), "no_geopotential": run(noz)}
        for k_, v_ in res.items():
            log(f"{k_:22s} basin {v_['basin_pct']:5.1f} %  window r {v_['window_corr']:.2f}  RMSE Z500 {v_['rmse']['z500']:.2f} m  "
                f"T850 {v_['rmse']['t850']:.3f} K  MSLP {v_['rmse']['msl']:.3f} hPa")
        json.dump(dict(t0=str(t0), ref_run=str(args.ref_run), hydrostatic_residual=resid,
                       dphi_ratio_opt_to_balanced=float(np.sqrt(np.sum(zo ** 2) / np.sum(zb ** 2))), variants=res),
                  open(out / "balance_summary.json", "w"), indent=1)
        return

    if args.mode == "forward":
        # control forecast twice (determinism check) and the 6-h rain over every target basin's
        # cells for each step (used for GraphCast's own 6-h forecasts during the event)
        res, losses = [], []
        for k in range(max(2, args.noise_reps)):
            t1 = time.time(); loss, alz, rains, _ = evaluate(a, False)
            res.append(rains); losses.append(loss)
            log(f"forward {k}: {time.time()-t1:.1f}s  loss {loss:.6f}  basin {np.round(alz,3)} mm")
        log(f"determinism: max |rain run1 - run2| = {float(np.max(np.abs(res[0] - res[1]))):.3e} mm; "
            f"sd(J) over {len(losses)} = {np.std(losses, ddof=1):.4f}")
        if args.save_truth:                                       # identical-twin truth, radar-file format
            tv = [np.datetime64(t) for t in target_valid]
            xr.Dataset({"precip_6h": (("time", "lat", "lon"), np.mean([r_[lead_idx][(slice(None),) + rsl] for r_ in res], 0)
                                      .astype(np.float32), {"units": "mm"})},
                       coords={"time": tv, "lat": lat[rlat], "lon": lon180[rlon]},
                       attrs={"source": f"GraphCast forecast from t0 {t0}, bg_ens_member {args.bg_ens_member}, mean of {len(res)} repeats",
                              "use": "identical-twin truth"}).to_netcdf(args.save_truth)
            log(f"twin truth written: {args.save_truth}")
        allc = json.load(open(TARGETS)); basins = {}
        for name, c in allc.items():
            if not isinstance(c, dict):
                continue
            la = np.where(np.isin(np.round(lat, 2), c["cells"]["lat"]))[0]
            lo = lon_idx(min(c["cells"]["lon"]), max(c["cells"]["lon"]))
            basins[name] = res[0][(slice(None),) + np.ix_(la, lo)].mean(axis=(1, 2)).round(3).tolist()
        vt = [str(t0 + timedelta(hours=6 * (k + 1))) for k in range(res[0].shape[0])]
        json.dump(dict(t0=str(t0), valid=vt, basin_rain_6h=basins, fp32=args.fp32, J_repeats=losses,
                       max_abs_diff_repeat=float(np.max(np.abs(res[0] - res[1])))),
                  open(out / "forward_summary.json", "w"), indent=1)
        return

    if args.mode == "gradtest":
        # Taylor test of the adjoint: (J(z + e d) - J(z)) / (e g.d) -> 1 as e -> 0
        sizes = [int(np.prod(ctrl_shapes[v])) for v in opt_vars]
        # direction = steepest descent (|d| = 10): a random direction is ~orthogonal to g in 1e7
        # dimensions, so g.d (~0.1) would be far below the evaluation noise (~2-7 in J)
        j0, _, _, g_ = evaluate(a, True)
        ng = np.sqrt(sum(float(jnp.sum(g_[v] ** 2)) for v in opt_vars))
        d_ = {v: -g_[v] * (10.0 / ng) for v in opt_vars}
        gd = sum(float(jnp.sum(g_[v] * d_[v])) for v in opt_vars)
        # exact float32 directional derivative along d (forward mode): separates a precision-limited bf16
        # adjoint from an intrinsically rough / non-linear cost
        # jit with the state as an argument (inside jit it is traced; outside, perturbed_state sees numpy)
        st, tst = jax.jit(lambda a_, d__, s_: jax.jvp(lambda x: perturbed_state(x, s_), (a_,), (d__,)))(a, d_, state0)
        rr, tr = [], []
        for k in range(nsteps):
            (st, r), (tst, t_r) = jvp_step(st, tst, constants, step_forcings[k])
            rr.append(np.asarray(r)); tr.append(np.asarray(t_r))
        rr, tr = np.stack(rr), np.stack(tr)
        rw_ = r0 + r1 * obs_win
        win_ = rr[lead_idx][(slice(None),) + rsl]; twin_ = tr[lead_idx][(slice(None),) + rsl]
        jvp_dir = float(np.sum((win_ - obs_win) * win_ok / rw_ ** 2 * twin_))
        log(f"JVP (fp32 exact) dJ/de {jvp_dir:+.4e}   bf16 adjoint g.d {gd:+.4e}   ratio adjoint/jvp {gd/jvp_dir:.3f}")
        jvp_rec = dict(jvp_fp32=jvp_dir, adjoint_bf16=gd, ratio_adjoint_over_jvp=gd / jvp_dir)
        groups = {"all": list(opt_vars), "specific_humidity": ["specific_humidity"], "geopotential": ["geopotential"],
                  "temperature": ["temperature"], "wind": ["u_component_of_wind", "v_component_of_wind"],
                  "surface": [v for v in opt_vars if v.startswith(("2m", "10m", "mean_sea"))]}
        out_t = []
        for gname, vs in groups.items():
            vs = [v for v in vs if v in opt_vars]
            ngv = np.sqrt(sum(float(jnp.sum(g_[v] ** 2)) for v in vs))
            dv_ = {v: (-g_[v] * (10.0 / ngv) if v in vs else jnp.zeros_like(a[v])) for v in opt_vars}
            gdv = -10.0 * ngv
            for e in (0.3, 0.1, 0.03):
                cd = []
                for _ in range(3):                                         # central differences, 3 repeats
                    jp, _, _, _ = evaluate({v: a[v] + e * dv_[v] for v in opt_vars}, False)
                    jm, _, _, _ = evaluate({v: a[v] - e * dv_[v] for v in opt_vars}, False)
                    cd.append((jp - jm) / (2 * e))
                ratio = float(np.mean(cd) / gdv)
                out_t.append(dict(group=gname, eps=e, dJ_de_mean=float(np.mean(cd)), dJ_de_sd=float(np.std(cd, ddof=1)),
                                  g_dot_d=gdv, ratio=ratio))
                log(f"taylor {gname:17s} eps {e:5.2f}  central dJ/de {np.mean(cd):+.4e} (sd {np.std(cd, ddof=1):.2e})  "
                    f"g.d {gdv:+.4e}  ratio {ratio:.3f}")
        fd_all = [t["dJ_de_mean"] for t in out_t if t["group"] == "all"]
        jvp_rec["fd_over_jvp"] = [fd / jvp_rec["jvp_fp32"] for fd in fd_all]
        log(f"central FD / fp32 JVP (all fields, eps 0.3/0.1/0.03): {np.round(jvp_rec['fd_over_jvp'], 3)}")
        json.dump(dict(t0=str(t0), J0=j0, taylor=out_t, jvp=jvp_rec), open(out / "gradtest.json", "w"), indent=1)
        return

    m = {v: jnp.zeros_like(a[v]) for v in opt_vars}
    s = {v: jnp.zeros_like(a[v]) for v in opt_vars}
    b1, b2, eps = 0.9, 0.999, 1e-8

    hist = []
    if args.cost == "4dvar":
        # J(z) = J_o + 1/2|z|^2 minimised with L-BFGS (8 pairs) and a backtracking (Armijo)
        # line search, as in variational data assimilation. Vectors live on the host.
        sizes = [int(np.prod(ctrl_shapes[v])) for v in opt_vars]

        def flat(d):
            return np.concatenate([np.asarray(d[v], np.float64).ravel() for v in opt_vars])

        def unflat(x):
            out, i = {}, 0
            for v, n in zip(opt_vars, sizes):
                out[v] = jnp.asarray(x[i:i + n].reshape(ctrl_shapes[v]), jnp.float32); i += n
            return out

        def fg(x):
            t1 = time.time()
            jo, alz, rains, g = evaluate(unflat(x), need_grad=True)
            jb = 0.5 * float(x @ x)
            return jo + jb, jo, jb, flat(g) + x, alz, rains, time.time() - t1

        x = np.zeros(sum(sizes))
        if args.init_seed >= 0:                                    # different starting point
            x = np.random.default_rng(args.init_seed).standard_normal(x.size)
            if args.init_zrms > 0:                                 # B-unit rows: white z of rms k -> increment ~ k sigma
                x *= args.init_zrms
                log(f"starting from random z (seed {args.init_seed}, z rms {args.init_zrms}, J_b = {0.5*x@x:.3g})")
            else:
                x *= np.sqrt(2 * args.init_jb) / np.linalg.norm(x)
                log(f"starting from random z (seed {args.init_seed}, J_b = {args.init_jb})")
        f, jo, jb, g, alz, rains, sec = fg(x)
        g0 = np.linalg.norm(g); nobs = int(win_ok.sum()); stop = "max iterations"
        # evaluation noise (non-deterministic GPU scatter-adds): sd of J over repeated forwards of the same z
        reps = [f] + [evaluate(unflat(x), need_grad=False)[0] + jb for _ in range(args.noise_reps - 1)]
        noise = max(float(np.std(reps, ddof=1)), 1e-6 * abs(f))
        log(f"evaluation noise at start: sd(J) = {noise:.3f} over {len(reps)} repeats  (range {np.ptp(reps):.2f})")
        best = dict(x=x.copy(), f=f, jo=jo, jb=jb, g=g.copy(), alz=alz, rains=rains, it=0)
        ref_f, stall = f, 0          # ref_f: J at the last improvement larger than 2 sd
        S, Y = [], []; restarts = 0
        for it in range(args.iters + 1):
            pc = pattern_corr(rains)
            hist.append(dict(iter=it, loss=f, Jo=jo, Jb=jb, basin_mm=alz.round(2).tolist(), basin_sum=float(alz.sum()),
                             window_corr=pc, sec=round(sec, 1), grad_norm=float(np.linalg.norm(g))))
            log(f"it {it:3d}  J {f:8.2f} (Jo {jo:8.2f} Jb {jb:7.2f})  |g| {np.linalg.norm(g):.2e}  basin {np.round(alz,1)} "
                f"sum {alz.sum():5.1f} mm  window r={pc:.2f}  ({sec:5.1f}s)")
            if it == args.iters:
                break
            if not np.isfinite(g).all():
                log("NON-FINITE GRADIENT - stopping"); stop = "non-finite gradient"; break
            if np.linalg.norm(g) < args.gtol * g0:
                log(f"converged: |g| fell below {args.gtol} x |g0|"); stop = "gradient"; break
            q = g.copy(); al = []                                  # two-loop recursion
            for s_, y_ in reversed(list(zip(S, Y))):
                r_ = 1.0 / (y_ @ s_); a_ = r_ * (s_ @ q); q -= a_ * y_; al.append((r_, a_, s_, y_))
            # initial Hessian scaling; first step limited to |dz| = 10 (J_b = 50)
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else min(1.0, 10.0 / np.linalg.norm(g))
            for r_, a_, s_, y_ in reversed(al):
                q += s_ * (a_ - r_ * (y_ @ q))
            d = -q; slope = float(g @ d)
            if slope >= 0:
                d = -g * min(1.0, 10.0 / np.linalg.norm(g)); slope = float(g @ d); S, Y = [], []
            step, ok = 1.0, False
            for _ in range(8):
                xn = x + step * d
                fn, jon, jbn, gn_, alzn, rainsn, sec = fg(xn)
                if np.isfinite(fn) and fn <= f + 1e-4 * step * slope + noise:   # Armijo, 1-sd noise tolerance
                    ok = True; break
                log(f"    line search: J {fn:.2f} > {f:.2f}, halving step {step:.3g}")
                step *= 0.5
            if not ok:
                # restart instead of stopping: drop the L-BFGS memory and take steepest-descent steps;
                # stop only when steepest descent itself finds no decrease, or after max restarts
                if S and restarts < args.max_restarts:
                    restarts += 1; S, Y = [], []
                    log(f"    line search failed: L-BFGS memory reset (restart {restarts}/{args.max_restarts})")
                    continue
                log("converged: line search found no decrease along steepest descent"); stop = "line search"; break
            s_, y_ = xn - x, gn_ - g
            if s_ @ y_ > 1e-10:
                S.append(s_); Y.append(y_); S, Y = S[-8:], Y[-8:]
            x, f, jo, jb, g, alz, rains = xn, fn, jon, jbn, gn_, alzn, rainsn
            # keep the best iterate; stop when the best J has not improved by more than the noise
            # tolerance for 8 accepted iterations
            if f < best["f"]:
                best = dict(x=x.copy(), f=f, jo=jo, jb=jb, g=g.copy(), alz=alz, rains=rains, it=it + 1)
            if best["f"] < ref_f - 2 * noise:                      # real (cumulative) improvement
                ref_f, stall = best["f"], 0
            else:
                stall += 1
            if stall >= 8:
                log(f"converged: J not improved by more than 2 sd ({2*noise:.2f}) below {ref_f:.2f} in 8 "
                    f"iterations (best {best['f']:.2f} at iteration {best['it']})"); stop = "noise floor"; break
        x, f, jo, jb, g = best["x"], best["f"], best["jo"], best["jb"], best["g"]
        jo_reps = [evaluate(unflat(x), need_grad=False)[0] for _ in range(args.noise_reps)]
        jo = float(np.mean(jo_reps)); f = jo + jb                  # report the repeat mean, not the lucky draw
        log(f"best iterate: {best['it']}  J {best['f']:.2f} (single) -> repeat mean J {f:.2f} (sd {np.std(jo_reps, ddof=1):.2f})")
        a = unflat(x)
        conv = dict(stop=stop, iterations=len(hist) - 1, restarts=restarts, J_start=hist[0]["loss"], best_iter=best["it"],
                    noise_J=noise, J_reduction=hist[0]["loss"] - f, Jo_repeats=jo_reps, grad_ratio=float(np.linalg.norm(g) / g0),
                    n_obs=nobs, chi2_per_obs=2 * jo / nobs, Jo_final=jo, Jb_final=jb)
        log(f"J_o {jo:.1f}  J_b {jb:.1f}  2 J_o / N_obs = {2*jo/nobs:.2f}  |g|/|g0| = {np.linalg.norm(g)/g0:.3f}  ({stop})")
    else:
        conv = {}
    for it in (range(args.iters + 1) if args.cost != "4dvar" else []):
        t1 = time.time()
        loss, alz, rains, g = evaluate(a, need_grad=(it < args.iters))
        jo = loss
        jb = float(sum(0.5 * jnp.sum(a[v] ** 2) for v in opt_vars)) if args.cost == "4dvar" else 0.0
        loss = jo + jb
        if args.cost == "4dvar" and g is not None:
            g = {v: g[v] + a[v] for v in opt_vars}                # grad J_b = z
        gn = {v: float(jnp.sqrt(jnp.sum(g[v] ** 2))) for v in opt_vars} if g is not None else {}
        pc = pattern_corr(rains)
        hist.append(dict(iter=it, loss=loss, Jo=jo, Jb=jb, basin_mm=alz.round(2).tolist(), basin_sum=float(alz.sum()),
                         window_corr=pc, sec=round(time.time() - t1, 1), grad_norm=gn))
        log(f"it {it:3d}  loss {loss:8.2f} (Jo {jo:8.2f} Jb {jb:7.2f})  basin {np.round(alz,1)} sum {alz.sum():5.1f} mm  "
            f"window r={pc:.2f}  ({time.time()-t1:5.1f}s)")
        if g is None:
            break
        if it == 0:
            log("grad norms: " + ", ".join(f"{k.split('_')[0]}:{val:.2e}" for k, val in gn.items()))
            if not all(np.isfinite(list(gn.values()))):
                log("NON-FINITE GRADIENT - stopping"); break
        k = it + 1
        lr = args.lr * (0.5 * (1 + np.cos(np.pi * it / args.iters)) if args.lr_decay == "cosine" else 1.0)
        for v in opt_vars:
            m[v] = b1 * m[v] + (1 - b1) * g[v]
            s[v] = b2 * s[v] + (1 - b2) * g[v] ** 2
            a[v] = a[v] - lr * (m[v] / (1 - b1 ** k)) / (jnp.sqrt(s[v] / (1 - b2 ** k)) + eps)

    # ── full forecasts for the flood model (no gradients) ───────────────────
    log(f"forecasting {nsteps_all} steps to {t_end:%Y-%m-%d %H}Z for control and optimised ...")
    a0 = {v: jnp.zeros(ctrl_shapes[v], jnp.float32) for v in opt_vars}
    _, _, rains_ctl, _ = evaluate(a0, False, n=nsteps_all)
    _, _, rains_opt, _ = evaluate(a, False, n=nsteps_all)
    alz_ctl = rains_ctl[(slice(None),) + asl].mean(axis=(1, 2))
    alz_opt = rains_opt[(slice(None),) + asl].mean(axis=(1, 2))
    # the routed forecast is rains_opt; repeat twice more to report the evaluation spread
    rep_sums = [float(alz_opt[lead_idx].sum())]
    for _ in range(2):
        _, _, r_, _ = evaluate(a, False)
        rep_sums.append(float(r_[lead_idx][(slice(None),) + asl].mean(axis=(1, 2)).sum()))
    final_fc = dict(basin_sum_routed=rep_sums[0], basin_sum_repeats=rep_sums,
                    basin_pct_routed=100 * rep_sums[0] / float(obs_alz.sum()),
                    basin_pct_spread=100 * (max(rep_sums) - min(rep_sums)) / float(obs_alz.sum()),
                    window_corr_routed=pattern_corr(rains_opt[:nsteps]))
    log(f"routed forecast: basin {rep_sums[0]:.1f} mm ({final_fc['basin_pct_routed']:.0f} %), repeats {np.round(rep_sums,1)}")
    for name, h in holdout.items():
        hs = np.ix_(h["lat"], h["lon"])
        c_, o_ = float(rains_ctl[lead_idx][(slice(None),) + hs].mean(axis=(1, 2)).sum()), float(rains_opt[lead_idx][(slice(None),) + hs].mean(axis=(1, 2)).sum())
        final_fc[f"holdout_{name}"] = dict(truth=h["truth"], control=c_, optimised=o_, n_masked=h["n_masked"],
                                           control_pct=100 * c_ / h["truth"], optimised_pct=100 * o_ / h["truth"])
        log(f"HOLD-OUT {name}: control {c_:.1f} -> optimised {o_:.1f} mm (truth {h['truth']:.1f}; "
            f"{100*c_/h['truth']:.0f} -> {100*o_/h['truth']:.0f} %)")

    # ── outputs ─────────────────────────────────────────────────────────────
    blat = lat[wlat]; blon = lon180[wlon]
    vt = [np.datetime64(t0 + timedelta(hours=6 * k)) for k in range(1, nsteps_all + 1)]
    xr.Dataset({"rain_control": (("time", "lat", "lon"), rains_ctl[(slice(None),) + wsl]),
                "rain_optimised": (("time", "lat", "lon"), rains_opt[(slice(None),) + wsl]),
                "basin_control": (("time",), alz_ctl), "basin_optimised": (("time",), alz_opt)},
               coords={"time": vt, "lat": blat, "lon": blon},
               attrs={"t0": str(t0), "units": "mm per 6 h"}).to_netcdf(out / "rain_control_vs_optimised.nc")
    frac = {v: float(jnp.mean(jnp.abs(increment(a[v], v) / bounds[v]))) for v in opt_vars}
    over = {v: float(jnp.mean(jnp.abs(increment(a[v], v)) > bounds[v])) for v in opt_vars}
    zrms = float(np.sqrt(sum(float(jnp.sum(a[v] ** 2)) for v in opt_vars) / sum(int(np.prod(ctrl_shapes[v])) for v in opt_vars)))
    # Physical increments (delta = bound * tanh a) at selected levels, both input times.
    inc = {}
    for v in opt_vars:
        dlt = np.asarray(increment(a[v], v))[0]                               # drop batch
        dims = [d for d in inputs[v].dims if d != "batch"]
        coords = {"time": ["t0-6h", "t0"], "lat": lat[plat], "lon": lon[plon]}
        if "level" in dims:
            li = [list(inputs.level.values).index(L) for L in SAVE_LEVELS]
            dlt = dlt[:, li]; coords["level"] = SAVE_LEVELS
        inc[v] = (dims, dlt.astype(np.float32))
    xr.Dataset(inc, coords=coords | {"level": SAVE_LEVELS},
               attrs={"t0": str(t0), "bound": args.bound, "eda_k": args.eda_k,
                      "note": "delta added to ERA5 at the two GraphCast input times"}
               ).to_netcdf(out / "increment.nc")
    json.dump(dict(t0=str(t0), nsteps=nsteps, nsteps_all=nsteps_all, lr=args.lr, lr_decay=args.lr_decay, bound=args.bound,
                   eda_k=args.eda_k, reg_weight=args.reg_weight, bound_frac=args.bound_frac,
                   truth_shift=args.truth_shift,
                   smooth_km=args.smooth_km, taper_deg=args.taper_deg,
                   cost=args.cost, obs_err=args.obs_err, tol=args.tol, gtol=args.gtol, init_seed=args.init_seed,
                   pert_box=list(pbox), convergence=conv, final_forecast=final_fc, truth_transform=args.truth_transform, z_rms=zrms, share_beyond_1sigma=over,
                   target=args.target, truth=(str(args.truth_file) if args.truth_file else cfg["truth_name"]), bg_ens_member=args.bg_ens_member,
                   fp32=args.fp32, truth_basin_mm=obs_alz.round(2).tolist(), mean_abs_tanh=frac, history=hist),
              open(out / "summary.json", "w"), indent=1)
    np.savez_compressed(out / "increment_tanh_a.npz",
                        **{v: np.asarray(increment(a[v], v) / bounds[v], np.float16) for v in opt_vars},
                        plat=lat[plat], plon=lon[plon])
    log(f"done -> {out}   mean |tanh a| per field: " + ", ".join(f"{k.split('_')[0]}:{x:.2f}" for k, x in frac.items()))


if __name__ == "__main__":
    main()
