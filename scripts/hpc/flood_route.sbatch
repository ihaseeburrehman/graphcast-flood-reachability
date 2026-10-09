#!/bin/bash
# Route one GraphCast rain field (control or optimised) through LISFLOOD-FP, identically
# to the multi-model paper: same patched binary, DEM, roughness, start state, boundary,
# 96 h window and forcing builder. Only the rain differs.
# usage: sbatch flood_route.sbatch <run_name e.g. lead_20210709T12> <control|optimised>
#SBATCH --job-name=gc_flood
#SBATCH --qos=ng --account=luuni
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem=120G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
RUN=$1 WHICH=$2
W=/ec/res4/scratch/lux0804/gc_flood_predictability
CODE=/ec/res4/hpcperm/lux0804/gc_flood_predictability/code
PAPER=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected
BIN=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_fp_8_2/build_patched_backup_v2/lisflood
PY=/usr/local/apps/ai-models/0.49/env-jax/bin/python
F=$W/flood/${RUN}_${WHICH}
mkdir -p $F/forecasts $F/results

echo "== rain -> forecast files -> 10 m dynamicrainfile  $(date +%T)"
$PY $CODE/scripts/hpc/split_for_lisflood.py $W/runs/$RUN $WHICH $F/forecasts
# builder copied verbatim from WRF_vs_AI_LISFLOOD_v1/analysis/data/lisflood_96h/scripts (writes the window attrs)
$PY $CODE/scripts/hpc/ai_to_lisflood_rain_96h.py graphcast $F/forecasts \
    $PAPER/inputs/Alzette_sub_basin_10m_bridge_burn.asc $F/rain_96h.nc

echo "== .par (paper's graphcast_corrected_18utc.par; only dirroot and dynamicrainfile changed)"
sed -e "s|^dirroot .*|dirroot                  $F/results|" \
    -e "s|^dynamicrainfile .*|dynamicrainfile          $F/rain_96h.nc|" \
    $PAPER/graphcast_corrected_18utc.par > $F/run.par
diff $PAPER/graphcast_corrected_18utc.par $F/run.par || true

echo "== LISFLOOD-FP  $(date +%T)"
module purge
module load prgenv/gnu gcc/8.5.0 cuda/11.6 netcdf4/4.9.2 cmake/3.28.3
"$BIN" $F/run.par
echo "== done $(date +%T)"
