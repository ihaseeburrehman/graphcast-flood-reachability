#!/bin/bash
# FRI pilot on one GPU: run fri_gradient.py in one mode for several inits (skips inits already done).
# usage: FRI_EXTRA="--targets-file F --repeats 2 --design" sbatch --export=ALL fri_run.sbatch <mode> <t0> [<t0> ...]
#SBATCH --job-name=fri
#SBATCH --qos=ng --account=luuni
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem=400G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
PY=/usr/local/apps/ai-models/0.49/env-jax/bin/python
CODE=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code
W=/ec/res4/scratch/lux0804/gc_flood_predictability
E5=${FRI_ERA5:-$W/era5/fri_jja2021}; OUT=${FRI_OUT:-$W/fri/jja2021}
MODE=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv
for T in "$@"; do
  tag=$(echo $T | tr -d '-' | sed 's/T\(..\)$/T\1/')
  f=$OUT/fri_${MODE}_${tag}; ls $f.* >/dev/null 2>&1 && { echo "== skip $T"; continue; }
  echo "== $MODE $T $(date +%T)"
  $PY $CODE/scripts/fri/fri_gradient.py --mode $MODE --t0 $T --era5-dir $E5 --out $OUT ${FRI_EXTRA:-} || echo "FAILED $T"
done
