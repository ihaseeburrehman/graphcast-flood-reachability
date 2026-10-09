#!/bin/bash
# Download and prepare ERA5 GraphCast inputs + EDA spread for one event.
# Same products and conventions as the 2021 inputs: SL analysis (NetCDF), 37-level PL analysis
# (GRIB, read with cfgrib), and 6-hourly tp built from hourly ERA5 forecast steps with
# build_accum_v2.py (tp summed over six hours, tisr kept on the hourly convention).
# usage: sbatch prep_event.sbatch <event> <first_date YYYYMMDD> <last_date> <eda_dates d1/d2/...>
#SBATCH --job-name=prep_event
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=4 --mem=64G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
EV=$1 D0=$2 D1=$3 EDA_DATES=$4
D=/ec/res4/scratch/lux0804/gc_flood_predictability/era5/$EV
P=/usr/local/apps/ai-models/0.49/env-jax/bin/python
BUILD=/ec/res4/scratch/lux0804/ai_rerun_2026/v2_tisr_fix/scripts/build_accum_v2.py
LEV=1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000
mkdir -p $D && cd $D
# accumulation base times are 06 and 18 UTC; T=00/06 need the previous day's 18 UTC run
DPREV=$(date -d "$D0 -1 day" +%Y%m%d)

cat > sl.req <<EOF
retrieve, class=ea, expver=1, stream=oper, type=an, levtype=sfc,
  param=31.128/34.128/129.128/137.128/151.128/165.128/166.128/167.128/172.128,
  date=$D0/to/$D1, time=00/06/12/18, grid=0.25/0.25, target="era5_sl_instant.grib"
EOF
cat > accum.req <<EOF
retrieve, class=ea, expver=1, stream=oper, type=fc, levtype=sfc, param=212.128/228.128,
  date=$DPREV/to/$D1, time=06/18, step=1/2/3/4/5/6/7/8/9/10/11/12, grid=0.25/0.25,
  target="era5_accum_raw.grib"
EOF
cat > pl.req <<EOF
retrieve, class=ea, expver=1, stream=oper, type=an, levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128/135.128,
  date=$D0/to/$D1, time=00/06/12/18, grid=0.25/0.25, target="era5_pl.grib"
EOF
cat > eda_sfc.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=es, levtype=sfc,
  param=151.128/165.128/166.128/167.128, date=$EDA_DATES, time=06/12, grid=0.25/0.25,
  target="era5_eda_spread_sfc.grib"
EOF
cat > eda_pl.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=es, levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128/135.128, date=$EDA_DATES, time=06/12,
  grid=0.25/0.25, target="era5_eda_spread_pl.grib"
EOF

for r in sl accum eda_sfc eda_pl pl; do
  t=$(grep -o 'target="[^"]*"' $r.req | cut -d'"' -f2)
  [ -s "$t" ] && { echo "=== $r exists"; continue; }
  echo "=== $r $(date +%T)"; mars $r.req; ls -la $t
done

module load ecmwf-toolbox
[ -s era5_sl_instant.nc ] || grib_to_netcdf -k 4 -d 1 -o era5_sl_instant.nc era5_sl_instant.grib
T0="$(date -d "$D0" +%Y-%m-%d) 00:00"; T1="$(date -d "$D1" +%Y-%m-%d) 18:00"
[ -s era5_accum_v2.nc ] || $P $BUILD era5_accum_raw.grib sum6h era5_accum_v2.nc "$T0" "$T1"
ls -lh $D
echo "PREP_DONE_$EV"
