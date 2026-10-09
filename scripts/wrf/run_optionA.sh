#!/bin/bash
# Option A: 0.75 d set (t0 13 Jul 12 UTC: control, +delta, -delta, 5 nulls) and 1.75 d amplitude sweep
# (+-3x, +-10x delta). Options via PERT_ARGS (ECMWF sbatch rejects "--").
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
R=/ec/res4/scratch/lux0804/gc_flood_predictability/runs
sub() { local n=$1 t=$2 r=$3; shift 3; PERT_ARGS="$*" sbatch --parsable --export=ALL --job-name=wrf_$n $S/wrf_case.sbatch $n $t $r; }
T=2021-07-13_12; RUN=$R/v5lead_20210713T12
echo "P075_ctl $(sub P075_ctl $T none)"; echo "P075_ctl2 $(sub P075_ctl2 $T none)"; echo "P075_delta $(sub P075_delta $T $RUN)"; echo "P075_mdelta $(sub P075_mdelta $T $RUN --scale -1)"
for k in 1 2 3 4 5; do echo "P075_null0$k $(sub P075_null0$k $T $RUN --null $k)"; done
T=2021-07-12_12; RUN=$R/v5lead_20210712T12
for s in 3 -3 10 -10; do n=P175_x${s/-/m}; echo "$n $(sub $n $T $RUN --scale $s)"; done
