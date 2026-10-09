#!/bin/bash
# Isolate the first-step blow-up: same wrfinput/wrfbdy (P175_ctl), 10 min of model time, one change at a time.
#SBATCH --job-name=wrf_isolate
#SBATCH --qos=np --account=luuni
#SBATCH --nodes=1 --ntasks=128 --time=02:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
source /ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf/wrf_env.sh
export GFORTRAN_ERROR_BACKTRACE=0

SRC=$W/cases/P175_ctl
run() {   # name, sed expression applied to the namelist
  D=$W/isolate_$1; rm -rf $D; mkdir -p $D; cd $D
  for f in $SRC/*; do b=$(basename $f); case $b in rsl.*|wrfout*|real_logs|fg|namelist.input|wrfinput_d01) ;; *) ln -s $f . ;; esac; done; ln -s $W/wrfinput_fixed wrfinput_d01
  sed -e 's/run_hours = [0-9]*/run_hours = 0/' -e 's/run_minutes = 0/run_minutes = 10/' -e "$2" $SRC/namelist.input > namelist.input
  mpiexec -n ${NP:-128} ./wrf.exe > wrf_stdout.txt 2>&1
  echo "$1: steps $(grep -c 'Timing for main' rsl.error.0000)  v_cfl $(cat rsl.error.* | grep -c 'points exceeded')  segv $(cat rsl.error.* | grep -c SIGSEGV)  $(grep -c 'SUCCESS COMPLETE WRF' rsl.error.0000)"
}
cp --remove-destination $SRC/wrfinput_d01 $W/wrfinput_fixed; /usr/local/apps/ai-models/0.49/env-jax/bin/python $CODE/scripts/wrf/fix_soil.py $W/wrfinput_fixed
run base   's/x/x/'
