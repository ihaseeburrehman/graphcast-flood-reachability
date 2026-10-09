#!/bin/bash
# Diagnose the first-step v_cfl blow-up: same wrfinput/wrfbdy, 10 min of model time with 1, 64 and 128 MPI
# tasks. Blow-ups that sit on patch boundaries and move with the decomposition point to MPI halo exchange.
#SBATCH --job-name=wrf_mpitest
#SBATCH --qos=np --account=luuni
#SBATCH --nodes=1 --ntasks=128 --time=02:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
source /ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf/wrf_env.sh
SRC=$W/cases/P175_ctl
for n in 128; do
  D=$W/mpitest_$n; rm -rf $D; mkdir -p $D; cd $D
  for f in $SRC/*; do b=$(basename $f); case $b in rsl.*|wrfout*|real_logs|fg|namelist.input) ;; *) ln -s $f . ;; esac; done
  sed -e 's/run_hours = [0-9]*/run_hours = 0/' -e 's/run_minutes = 0/run_minutes = 10/' -e 's/history_interval = 360/history_interval = 10/' $SRC/namelist.input > namelist.input
  echo "== $n tasks $(date +%T)"
  if [ $n = 1 ]; then ./wrf.exe > wrf1.out 2>&1; L=rsl.error.0000; [ -f $L ] || L=wrf1.out; else mpiexec -n $n ./wrf.exe; L=rsl.error.0000; fi
  echo "steps: $(grep -c 'Timing for main' $L 2>/dev/null)   v_cfl lines: $(cat rsl.error.* wrf1.out 2>/dev/null | grep -c 'points exceeded')"
  cat rsl.error.* wrf1.out 2>/dev/null | grep -A1 'points exceeded' | grep 'Max   W' | awk '{print "   max W at i,j,k", $5, $6, $7}' | sort | uniq | head -5
  grep -h 'SUCCESS' rsl.error.0000 wrf1.out 2>/dev/null | head -1
done
