#!/bin/bash
# Option B: 19 structure-matched nulls at 0.75 d (perturb_intermediate.py --null-match: 250 km in true km both ways,
# delta-matched vertical coherence, same envelope and per-variable norm). Same reference soil as the powered test.
set -euo pipefail
S=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf
R=/ec/res4/scratch/lux0804/gc_flood_predictability/runs
T=2021-07-13_12; RUN=$R/v5lead_20210713T12
for k in $(seq 1 19); do n=P075m_null$(printf %02d $k)
  echo "$n $(PERT_ARGS="--null $k --null-match" sbatch --parsable --export=ALL --job-name=wrf_$n $S/wrf_case.sbatch $n $T $RUN)"; done
