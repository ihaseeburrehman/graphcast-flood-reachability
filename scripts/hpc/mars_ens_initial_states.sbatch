#!/bin/bash
# IFS ENS starting states (step 0, control + 50 members) at the GraphCast start times, for the
# physics check of the optimised increments (ensemble sensitivity): do the IFS members whose
# initial perturbation projects onto the GraphCast increment produce more basin rain?
# Also the ENS 6-h rain for the 2024 events (2021 is already in operational_2021/).
# Levels: the 11 pressure levels present in ENS cf and pf in 2021 and 2024 (2021: no 600, no 150 in pf).
# Area: union of the perturbation boxes in configs/targets.json.
# usage: sbatch mars_ens_initial_states.sbatch
#SBATCH --job-name=mars_ens_ic
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=2 --mem=16G --time=48:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
O=/ec/res4/scratch/lux0804/gc_flood_predictability/ens_ic
mkdir -p $O; cd $O
LEV=1000/925/850/700/500/400/300/250/200/100/50
AREA=72/-45/20/45
declare -A DATES=([2021]=20210707/20210708/20210709/20210710/20210711/20210712/20210713
                  [boris]=20240906/20240908/20240910/20240911/20240912
                  [valencia]=20241022/20241024/20241026/20241027/20241028)

for ev in 2021 boris valencia; do
  for kind in cf pf; do
    NUM=""; [ $kind = pf ] && NUM="number=1/to/50,"
    cat > ${ev}_${kind}_pl.req <<EOF
retrieve, class=od, expver=1, stream=enfo, type=$kind, $NUM levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128, date=${DATES[$ev]}, time=12, step=0,
  area=$AREA, grid=0.25/0.25, target="${ev}_${kind}_pl.grib"
EOF
    cat > ${ev}_${kind}_sfc.req <<EOF
retrieve, class=od, expver=1, stream=enfo, type=$kind, $NUM levtype=sfc,
  param=151.128/165.128/166.128/167.128, date=${DATES[$ev]}, time=12, step=0,
  area=$AREA, grid=0.25/0.25, target="${ev}_${kind}_sfc.grib"
EOF
  done
done
# ENS rain for the 2024 events (6-hourly accumulations, to day 10)
for ev in boris valencia; do
  [ $ev = boris ] && A=53/12/47/20 || A=42/-3/37/2
  for kind in cf pf; do
    NUM=""; [ $kind = pf ] && NUM="number=1/to/50,"
    cat > ${ev}_${kind}_tp.req <<EOF
retrieve, class=od, expver=1, stream=enfo, type=$kind, $NUM levtype=sfc, param=228.128,
  date=${DATES[$ev]}, time=12, step=0/to/240/by/6, area=$A, grid=0.25/0.25, target="${ev}_${kind}_tp.grib"
EOF
  done
done

for r in *.req; do
  t=$(grep -o 'target="[^"]*"' $r | cut -d'"' -f2)
  [ -s "$t" ] || { echo "=== $r $(date +%T)"; mars $r; }; ls -la $t
done
echo DONE
