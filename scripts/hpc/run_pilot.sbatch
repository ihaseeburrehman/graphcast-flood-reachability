#!/bin/bash
# Pilot: GraphCast initial-condition optimisation, one initialisation time, one A100.
# usage: sbatch run_pilot.sbatch 2021-07-13T12 <iters> <tag> [optimise|forward] [lr] [smooth_km] [target]   (extra options via env: GC_EXTRA="--cost 4dvar")   (no spaces: the ECMWF sbatch wrapper re-splits arguments)
# Non-login shell on purpose: on Atos a login shell (-l) redirects stdout and kills the job.
#SBATCH --job-name=gc_pilot
#SBATCH --qos=ng --account=luuni
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem=400G --time=08:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
CODE=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code
RUN=/ec/res4/scratch/lux0804/gc_flood_predictability/runs/${3:-pilot}_$(echo "$1" | tr -d ":-")
PY=/usr/local/apps/ai-models/0.49/env-jax/bin/python
# no module load: env-jax ships its own CUDA/cuDNN; loading cuda/11.8 breaks cuDNN init
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# exact-size allocations: no BFC pool, so no fragmentation (one step backward needs ~36 GB)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
exec "$PY" "$CODE/scripts/hpc/gc_optimise_ic.py" --t0 "$1" --iters "${2:-3}" --lr "${5:-0.05}" --mode "${4:-optimise}" --smooth-km "${6:-250}" --target "${7:-alzette_2021}" --out "$RUN" ${GC_EXTRA:-}
