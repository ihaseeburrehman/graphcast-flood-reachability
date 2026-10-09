#!/bin/bash
# Two benchmarks for the July 2021 lead-time experiment.
#  1. ERA5 EDA members (10) and EDA mean at the GraphCast input times (06 and 12 UTC,
#     7-13 July), for the fields the optimiser may change. GraphCast started from
#     ERA5 + (member - mean) shows how the flood rain spreads across the *real* analysis
#     uncertainty, so the optimised start can be judged as plausible or merely possible.
#  2. ECMWF operational forecasts of total precipitation from 7-13 July (00 and 12 UTC):
#     HRES and the 51-member ENS - what forecasters actually had.
# usage: sbatch mars_ensemble_and_operational.sbatch
#SBATCH --job-name=mars_ens_op
#SBATCH --qos=nf --account=luuni
#SBATCH --cpus-per-task=2 --mem=16G --time=36:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/gc_flood_predictability/logs/%x_%j.log
set -euo pipefail
D=/ec/res4/scratch/lux0804/gc_flood_predictability/era5/eda_members_2021
O=/ec/res4/scratch/lux0804/gc_flood_predictability/operational_2021
mkdir -p $D $O
LEV=1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000

cd $O
cat > hres.req <<EOF
retrieve, class=od, expver=1, stream=oper, type=fc, levtype=sfc, param=228.128,
  date=20210707/to/20210713, time=00/12, step=0/to/240/by/6,
  area=52/2/47/10, grid=0.1/0.1, target="hres_tp.grib"
EOF
cat > ens_cf.req <<EOF
retrieve, class=od, expver=1, stream=enfo, type=cf, levtype=sfc, param=228.128,
  date=20210707/to/20210713, time=00/12, step=0/to/240/by/6,
  area=52/2/47/10, grid=0.2/0.2, target="ens_cf_tp.grib"
EOF
cat > ens_pf.req <<EOF
retrieve, class=od, expver=1, stream=enfo, type=pf, levtype=sfc, param=228.128,
  number=1/to/50, date=20210707/to/20210713, time=00/12, step=0/to/240/by/6,
  area=52/2/47/10, grid=0.2/0.2, target="ens_pf_tp.grib"
EOF
for r in hres ens_cf ens_pf; do
  t=$(grep -o 'target="[^"]*"' $r.req | cut -d'"' -f2)
  [ -s "$t" ] || { echo "=== $r $(date +%T)"; mars $r.req; }; ls -la $t
done

cd $D
for kind in em an; do
  NUM=""; [ $kind = an ] && NUM="number=0/1/2/3/4/5/6/7/8/9,"
  cat > ${kind}_sfc.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=$kind, levtype=sfc, $NUM
  param=151.128/165.128/166.128/167.128, date=20210707/to/20210713, time=06/12,
  grid=0.25/0.25, target="eda_${kind}_sfc.grib"
EOF
  cat > ${kind}_pl.req <<EOF
retrieve, class=ea, expver=1, stream=enda, type=$kind, levtype=pl, $NUM levelist=$LEV,
  param=129.128/130.128/131.128/132.128/133.128/135.128, date=20210707/to/20210713, time=06/12,
  grid=0.25/0.25, target="eda_${kind}_pl.grib"
EOF
done
for r in em_sfc em_pl an_sfc an_pl; do
  t=$(grep -o 'target="[^"]*"' $r.req | cut -d'"' -f2)
  [ -s "$t" ] || { echo "=== $r $(date +%T)"; mars $r.req; }; ls -la $t
done
echo DONE
