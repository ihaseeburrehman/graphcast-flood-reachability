#!/bin/bash
# WPS for the WRF transfer test: geogrid (new 12 km domain 30-65N, 35W-30E) and, when ERA5 is there,
# ungrib (Vtable.ECMWF) + metgrid for the control over 11 Jul 12 UTC - 15 Jul 06 UTC.
# usage: sbatch wps_setup.sbatch [geogrid|ungrib|metgrid|geogrid_metgrid]
#SBATCH --job-name=wps_setup
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=4 --mem=32G --time=12:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
source /ec/res4/hpcperm/lux0804/gc_flood_predictability/code/scripts/wrf/wrf_env.sh
D=$W/wps; mkdir -p $D; cd $D
for f in geogrid metgrid ungrib; do [ -e $f ] || ln -s $WPS/$f .; done
for f in geogrid.exe metgrid.exe ungrib.exe link_grib.csh; do ln -sf $WPS/$f .; done
mk() { sed -e "s|__START__|$1|" -e "s|__END__|$2|" -e "s|__GEOG__|$GEOG|" -e "s|__FG__|$3|" \
         $CODE/scripts/wrf/namelist.wps.tmpl > namelist.wps; }
case ${1:-geogrid} in
geogrid)
  mk 2021-07-11_12:00:00 2021-07-15_06:00:00 FILE
  ./geogrid.exe && ls -la geo_em.d01.nc ;;
ungrib)
  mk 2021-07-11_12:00:00 2021-07-15_06:00:00 FILE
  ln -sf $CODE/scripts/wrf/Vtable.ECMWF_q Vtable   # Vtable.ECMWF + specific humidity (no RH downloaded; ungrib derives RH)
  ./link_grib.csh $W/era5/era5_pl.grib $W/era5/era5_sfc.grib
  ./ungrib.exe > ungrib.out 2>&1; tail -3 ungrib.out; ls FILE:* | wc -l ;;
metgrid)
  mk 2021-07-11_12:00:00 2021-07-15_06:00:00 FILE
  ./metgrid.exe > metgrid.out 2>&1; tail -3 metgrid.out; ls met_em.d01.* | wc -l ;;
geogrid_metgrid)
  mk 2021-07-11_12:00:00 2021-07-15_06:00:00 FILE
  ./geogrid.exe > geogrid.out 2>&1; tail -2 geogrid.out; ls -la geo_em.d01.nc
  ./metgrid.exe > metgrid.out 2>&1; tail -3 metgrid.out; ls met_em.d01.* | wc -l ;;
esac
