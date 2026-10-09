#!/bin/bash
# resubmit the pilot cases whose sbatch was rejected (options passed via PERT_ARGS); WPS job id as $1
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
RUN=/ec/res4/scratch/lux0804/gc_flood_predictability/runs/v5lead_20210712T12; T0=2021-07-12_12; WPSJ=$1
sub() { local n=$1 t=$2 r=$3; shift 3; PERT_ARGS="$*" sbatch --parsable --export=ALL --dependency=afterok:$WPSJ --job-name=wrf_$n $S/wrf_case.sbatch $n $t $r; }
echo "P175_mdelta $(sub P175_mdelta $T0 $RUN --scale -1)"
for k in 1 2 3 4 5; do echo "P175_null0$k $(sub P175_null0$k $T0 $RUN --null $k)"; done
