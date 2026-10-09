#!/bin/bash
# WRF transfer-test pilot at 1.75 d (t0 12 Jul 2021 12 UTC): WPS (geogrid + metgrid control), then
# 2 controls (determinism), +delta, -delta, 5 envelope-matched nulls, each starting when WPS has succeeded.
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
RUN=/ec/res4/scratch/lux0804/gc_flood_predictability/runs/v5lead_20210712T12
T0=2021-07-12_12
WPSJ=$(sbatch --parsable $S/wps_setup.sbatch geogrid_metgrid)
echo "WPS $WPSJ"
sub() { local n=$1 t=$2 r=$3; shift 3; PERT_ARGS="$*" sbatch --parsable --export=ALL --dependency=afterok:$WPSJ --job-name=wrf_$n $S/wrf_case.sbatch $n $t $r; }
for c in "P175_ctl $T0 none" "P175_ctl2 $T0 none" "P175_delta $T0 $RUN" "P175_mdelta $T0 $RUN --scale -1" \
         "P175_null01 $T0 $RUN --null 1" "P175_null02 $T0 $RUN --null 2" "P175_null03 $T0 $RUN --null 3" \
         "P175_null04 $T0 $RUN --null 4" "P175_null05 $T0 $RUN --null 5"; do
  set -- $c; echo "$1 $(sub "$@")"
done
