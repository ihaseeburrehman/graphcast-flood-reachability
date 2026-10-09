#!/bin/bash
# Archive the data behind Paper A to ECFS (ECMWF's permanent, backed-up tape archive).
# One tar per component, copied with ecp to ec:/lux0804/gc_flood_predictability/, checked by size; originals untouched.
# ERA5 / E-OBS inputs are re-downloadable (MARS / ECA&D): only their request files are archived.
#SBATCH --job-name=archive_ecfs
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=4 --mem=16G --time=48:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
W=/ec/res4/scratch/lux0804/gc_flood_predictability
T=$W/archive_tmp; mkdir -p $T
EC=ec:/lux0804/gc_flood_predictability
emkdir -p $EC 2>/dev/null || true
arch() {   # name, base dir, paths...
  local name=$1 base=$2; shift 2
  echo "== $name $(date +%T)"; tar -C $base -cf $T/$name.tar "$@"
  ecp -o $T/$name.tar $EC/$name.tar
  local a=$(stat -c %s $T/$name.tar) b=$(els -l $EC/$name.tar | awk '{print $5}')
  [ "$a" = "$b" ] && echo "   OK $name $a bytes" || { echo "   SIZE MISMATCH $name $a vs $b"; exit 1; }
  tar -tf $T/$name.tar > $T/$name.list; ecp -o $T/$name.list $EC/$name.list; rm -f $T/$name.tar
}
arch graphcast_runs        $W runs
arch wrf_cases             $W wrf/cases wrf/soil_ref
arch wrf_wps               $W wrf/wps
arch lisflood_runs         $W flood
arch lisflood_inputs       /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected inputs
arch era5_requests_logs    $W $(cd $W && ls -d era5/*.req era5/*/*.req logs 2>/dev/null)
arch fri_pilot             $W fri
echo "== done $(date +%T)"; els -l $EC
