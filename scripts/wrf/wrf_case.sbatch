#!/bin/bash
# One WRF transfer-test case: ERA5 (+ optional perturbation at t0) -> metgrid (t0 only; later times reuse the
# control met_em) -> real.exe -> wrf.exe, free forecast (no data assimilation) to 15 Jul 06 UTC.
# cumulus scheme via env CU (default 3 = Grell-Freitas; 11 = multi-scale Kain-Fritsch)
# usage: sbatch wrf_case.sbatch <case_name> <t0 YYYY-MM-DD_HH> <opt_run_dir|none> [perturb_intermediate options]
#   e.g. sbatch wrf_case.sbatch L175_delta 2021-07-12_12 $RUNS/v5lead_20210712T12
#        sbatch wrf_case.sbatch L175_null03 2021-07-12_12 $RUNS/v5lead_20210712T12 --null 3
#        sbatch wrf_case.sbatch L175_ctl 2021-07-12_12 none
#SBATCH --job-name=wrf_case
#SBATCH --qos=np --account=luuni
#SBATCH --nodes=1 --ntasks=128 --time=12:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
source /ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf/wrf_env.sh
NAME=$1 T0=$2 RUN=$3; shift 3; EXTRA="${PERT_ARGS:-$*}"   # ECMWF sbatch rejects "--" options: pass them via env PERT_ARGS
PY=/usr/local/apps/ai-models/0.49/env-jax/bin/python
C=$W/cases/$NAME; rm -rf $C; mkdir -p $C/fg; cd $C
END=2021-07-15_06
echo "== $NAME  t0 $T0  run $RUN  $EXTRA  $(date +%T)"
# 1. initial-time intermediate file (perturbed or copied)
if [ "$RUN" = none ] && [ -z "$EXTRA" ]; then cp $W/wps/FILE:${T0} fg/FILE:${T0}
else $PY $CODE/scripts/wrf/perturb_intermediate.py $W/wps/FILE:${T0} fg/FILE:${T0} $RUN $EXTRA; fi
# 2. metgrid for t0 only; later times from the control
sed -e "s|__START__|${T0}:00:00|" -e "s|__END__|${T0}:00:00|" -e "s|__GEOG__|$GEOG|" -e "s|__FG__|$C/fg/FILE|" \
    $CODE/scripts/wrf/namelist.wps.tmpl > namelist.wps
ln -sf $W/wps/geo_em.d01.nc .; ln -s $WPS/metgrid .; ln -sf $WPS/metgrid.exe .
./metgrid.exe > metgrid.out 2>&1 || { tail metgrid.out; exit 1; }
for f in $W/wps/met_em.d01.*; do b=$(basename $f); [ -e $b ] || ln -s $f .; done
# 3. real + wrf
NLEV=$(ncdump -h met_em.d01.${T0}:00:00.nc 2>/dev/null | awk '/num_metgrid_levels =/{print $3}' || true)
NLEV=${NLEV:-38}
H=$(( ( $(date -u -d "${END/_/ }:00" +%s) - $(date -u -d "${T0/_/ }:00" +%s) ) / 3600 ))
read Y0 M0 D0 H0 <<< "$(echo $T0 | tr '_-' '  ')"; read Y1 M1 D1 H1 <<< "$(echo $END | tr '_-' '  ')"
sed -e "s|__HOURS__|$H|" -e "s|__Y0__|$Y0|" -e "s|__M0__|$M0|" -e "s|__D0__|$D0|" -e "s|__H0__|$H0|" \
    -e "s|__Y1__|$Y1|" -e "s|__M1__|$M1|" -e "s|__D1__|$D1|" -e "s|__H1__|$H1|" -e "s|__NMETLEV__|$NLEV|" -e "s|__CU__|${CU:-3}|" \
    $CODE/scripts/wrf/namelist.input.tmpl > namelist.input
for f in $WRFRUN/*; do b=$(basename $f); case $b in namelist.input|wrfinput*|wrfbdy*|wrfout*|rsl.*) ;; *) ln -sf $f . ;; esac; done
mpiexec -n 32 ./real.exe; tail -2 rsl.error.0000; mkdir -p real_logs; mv rsl.* real_logs/
$PY $CODE/scripts/wrf/fix_soil.py wrfinput_d01      # undefined coastal soil values (identical treatment for all cases)
mkdir -p $W/soil_ref; $PY $CODE/scripts/wrf/copy_soil.py wrfinput_d01 $W/soil_ref/soil_${T0}.nc   # one soil state per t0
mpiexec -n $SLURM_NTASKS ./wrf.exe; tail -2 rsl.error.0000
grep -q "SUCCESS COMPLETE WRF" rsl.error.0000 && echo "== WRF OK $(date +%T)"
# 4. keep rain + coordinates only
$PY $CODE/scripts/wrf/wrf_rain.py $C > rain_summary.txt && cat rain_summary.txt
[ -n "${KEEP_WRFOUT:-}" ] || case $NAME in *ctl*) ;; *) rm -f wrfout_d01_* ;; esac   # controls keep wrfout (diagnose the storm track); t0 met_em + wrfinput kept
