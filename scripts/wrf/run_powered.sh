#!/bin/bash
# Powered 0.75 d transfer test (t0 13 Jul 12 UTC), all cases with the same reference soil (copy_soil.py):
#   +delta, -delta, 19 envelope-matched nulls (seeds 1-19), and micro-perturbation ensembles (white T noise,
#   0.05 K) of the control (5) and of +delta (5) for WRF's intrinsic spread. Control = P075_ctl (soil reference).
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
R=/ec/res4/scratch/lux0804/gc_flood_predictability/runs
sub() { local n=$1 t=$2 r=$3; shift 3; PERT_ARGS="$*" sbatch --parsable --export=ALL --job-name=wrf_$n $S/wrf_case.sbatch $n $t $r; }
T=2021-07-13_12; RUN=$R/v5lead_20210713T12
echo "P075p_delta $(sub P075p_delta $T $RUN)"; echo "P075p_mdelta $(sub P075p_mdelta $T $RUN --scale -1)"
for k in $(seq 1 19); do n=P075p_null$(printf %02d $k); echo "$n $(sub $n $T $RUN --null $k)"; done
for k in 1 2 3 4 5; do echo "P075p_ctlm$k $(sub P075p_ctlm$k $T none --micro $k)"; echo "P075p_deltam$k $(sub P075p_deltam$k $T $RUN --micro $k)"; done
