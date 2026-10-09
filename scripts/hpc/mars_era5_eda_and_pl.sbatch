#!/bin/bash
# ERA5 inputs for lead times beyond ~1.5 days, and the EDA spread that bounds the increment.
#   1. EDA spread (stream=enda, type=es): the analysis uncertainty at each initialisation
#      time, for the fields the optimiser may change. Only the input times are needed
#      (t0-6h and t0) for t0 = 12 UTC on 7-14 July.
#   2. ERA5 pressure levels (oper, an) for 7-11 July; 12-17 July already exist in
#      ai_rerun_2026/data/2021_floodwindow/era5_mars/era5_pl.nc.
# usage: sbatch mars_era5_eda_and_pl.sbatch
#SBATCH --job-name=mars_eda_pl
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=2 --mem=16G --time=24:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
D=/ec/res4/scratch/lux0804/gc_flood_predictability/era5
mkdir -p $D && cd $D
LEV=1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000

cat > eda_pl.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=es, levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128/135.128,
  date=20210707/to/20210714, time=06/12, grid=0.25/0.25, target="era5_eda_spread_pl.grib"
EOF
cat > eda_sfc.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=es, levtype=sfc,
  param=151.128/165.128/166.128/167.128,
  date=20210707/to/20210714, time=06/12, grid=0.25/0.25, target="era5_eda_spread_sfc.grib"
EOF
cat > oper_pl.req <<EOF
retrieve, class=ea, expver=1, stream=oper, type=an, levtype=pl, levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128/135.128,
  date=20210707/to/20210711, time=00/06/12/18, grid=0.25/0.25, target="era5_pl_0707_0711.grib"
EOF

for r in eda_sfc eda_pl oper_pl; do
  echo "=== $r $(date +%T)"; mars $r.req && ls -la $(grep -o 'target="[^"]*"' $r.req | cut -d'"' -f2)
done
echo DONE
