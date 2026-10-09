#!/bin/bash
# Is the ECMWF WRF build sound? Same ERA5 intermediate files, the author's validated 120x120 12 km Greater-Region
# domain, 1 h of model time with the same physics/numerics. Crash -> the build; clean run -> large-domain data.
#SBATCH --job-name=wrf_small
#SBATCH --qos=np --account=luuni
#SBATCH --nodes=1 --ntasks=64 --time=01:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -uo pipefail
source /ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf/wrf_env.sh
D=$W/small_test; rm -rf $D; mkdir -p $D; cd $D
ln -s $WPS/geogrid .; ln -s $WPS/metgrid .; ln -sf $WPS/geogrid.exe .; ln -sf $WPS/metgrid.exe .
cat > namelist.wps <<NL
&share
 wrf_core='ARW', max_dom=1, start_date='2021-07-12_12:00:00', end_date='2021-07-12_15:00:00', interval_seconds=10800, io_form_geogrid=2,
/
&geogrid
 parent_id=1, parent_grid_ratio=1, i_parent_start=1, j_parent_start=1, e_we=120, e_sn=120,
 geog_data_res='nesdis_greenfrac+default', dx=12000, dy=12000, map_proj='lambert',
 ref_lat=49.6255, ref_lon=6.208, truelat1=30.0, truelat2=60.0, stand_lon=6.208, geog_data_path='$GEOG'
/
&ungrib
 out_format='WPS', prefix='FILE',
/
&metgrid
 fg_name='$W/wps/FILE', io_form_metgrid=2,
/
NL
./geogrid.exe > geogrid.out 2>&1; tail -1 geogrid.out; ./metgrid.exe > metgrid.out 2>&1; tail -1 metgrid.out
for f in $WRFRUN/*; do b=$(basename $f); case $b in namelist.input|wrfinput*|wrfbdy*|wrfout*|rsl.*) ;; *) ln -sf $f . ;; esac; done
sed -e 's/__HOURS__/1/' -e 's/__Y0__/2021/;s/__M0__/07/;s/__D0__/12/;s/__H0__/12/' -e 's/__Y1__/2021/;s/__M1__/07/;s/__D1__/12/;s/__H1__/15/' \
    -e 's/__NMETLEV__/38/' -e 's/__CU__/3/' -e 's/e_we = 420, e_sn = 340/e_we = 120, e_sn = 120/' $CODE/scripts/wrf/namelist.input.tmpl > namelist.input
mpiexec -n 16 ./real.exe; tail -1 rsl.error.0000; mkdir -p rl; mv rsl.* rl/
/usr/local/apps/ai-models/0.49/env-jax/bin/python $CODE/scripts/wrf/fix_soil.py wrfinput_d01
mpiexec -n 64 ./wrf.exe > out.txt 2>&1
echo "small domain: steps $(grep -c 'Timing for main' rsl.error.0000)  segv $(cat rsl.error.* | grep -c SIGSEGV)  $(grep -c 'SUCCESS COMPLETE WRF' rsl.error.0000)"
