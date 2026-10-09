#!/bin/bash
# Several optimisation runs one after the other on one A100 (GPU job limit is 10 per user).
# usage: sbatch run_chain.sbatch <list file>
# Each non-comment line of the list:  <t0> <tag> <target> [extra gc_optimise_ic.py options]
# Output of each run: runs/<tag>_<t0 without : and ->; a run whose summary.json exists is skipped.
# Non-login shell on purpose: on Atos a login shell (-l) redirects stdout and kills the job.
#SBATCH --job-name=gc_chain
#SBATCH --qos=ng --account=luuni
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem=400G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
CODE=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code
RUNS=/ec/res4/scratch/lux0804/gc_flood_predictability/runs
PY=/usr/local/apps/ai-models/0.49/env-jax/bin/python
# no module load: env-jax ships its own CUDA/cuDNN; loading cuda/11.8 breaks cuDNN init
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
nvidia-smi --query-gpu=name,memory.total --format=csv
grep -vE '^\s*(#|$)' "$1" | while read -r t0 tag target extra; do
  RUN=$RUNS/${tag}_$(echo "$t0" | tr -d ":-")
  if [ -s "$RUN/summary.json" ]; then echo "=== skip $RUN (done)"; continue; fi
  echo "=== $(date +%T) $t0 $tag $target $extra"
  "$PY" "$CODE/scripts/hpc/gc_optimise_ic.py" --t0 "$t0" --target "$target" --out "$RUN" \
      --cost 4dvar --iters ${GC_ITERS:-100} $extra < /dev/null || echo "=== FAILED $RUN"
done
echo "=== chain done $(date +%T)"
