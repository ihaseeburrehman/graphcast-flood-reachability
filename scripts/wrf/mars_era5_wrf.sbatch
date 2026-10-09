#!/bin/bash
# ERA5 initial and boundary data for the WRF transfer test (11 Jul 12 UTC - 15 Jul 06 UTC 2021, 3-hourly):
# pressure levels (37) T, u, v, q, z  -- NO relative humidity: ungrib derives RH from SPECHUMD, so the GraphCast
# humidity increment (applied to the intermediate files) is not overwritten by an ERA5 RH field;
# surface + soil fields for Vtable.ECMWF. Area covers the WRF Lambert domain (corners reach ~52W/47E) with margin.
# usage: sbatch mars_era5_wrf.sbatch
#SBATCH --job-name=mars_wrf
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=2 --mem=16G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
O=/ec/res4/scratch/lux0804/gc_flood_predictability/wrf/era5; mkdir -p $O; cd $O
DATES=20210711/to/20210715; TIMES=00/03/06/09/12/15/18/21; AREA=75/-70/20/65
LEV=1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000
cat > pl.req <<R
retrieve, class=ea, expver=1, stream=oper, type=an, levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128, date=$DATES, time=$TIMES,
  area=$AREA, grid=0.25/0.25, target="era5_pl.grib"
R
cat > sfc.req <<R
retrieve, class=ea, expver=1, stream=oper, type=an, levtype=sfc,
  param=31.128/34.128/39.128/40.128/41.128/42.128/129.128/134.128/139.128/141.128/151.128/165.128/166.128/167.128/168.128/170.128/172.128/183.128/235.128/236.128,
  date=$DATES, time=$TIMES, area=$AREA, grid=0.25/0.25, target="era5_sfc.grib"
R
for r in pl sfc; do t=era5_$r.grib; [ -s $t ] || mars $r.req; ls -la $t; done
echo DONE
