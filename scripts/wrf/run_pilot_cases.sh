#!/bin/bash
# Submit the 1.75 d pilot cases (WPS already done). Options go via PERT_ARGS (ECMWF sbatch rejects "--").
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
RUN=/ec/res4/scratch/lux0804/gc_flood_predictability/runs/v5lead_20210712T12; T0=2021-07-12_12
sub() { local n=$1 t=$2 r=$3; shift 3; PERT_ARGS="$*" sbatch --parsable --export=ALL --job-name=wrf_$n $S/wrf_case.sbatch $n $t $r; }
echo "P175_ctl $(sub P175_ctl $T0 none)"; echo "P175_ctl2 $(sub P175_ctl2 $T0 none)"
echo "P175_delta $(sub P175_delta $T0 $RUN)"; echo "P175_mdelta $(sub P175_mdelta $T0 $RUN --scale -1)"
for k in 1 2 3 4 5; do echo "P175_null0$k $(sub P175_null0$k $T0 $RUN --null $k)"; done
