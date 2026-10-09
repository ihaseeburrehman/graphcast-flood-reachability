# Sensitive initial-state directions let an AI weather model reach Europe's 2021 flood rainfall

Code and selected derived data accompanying the manuscript by Rehman & Teferle.

Archived releases: [10.5281/zenodo.23270441](https://doi.org/10.5281/zenodo.23270441).
All versions: [10.5281/zenodo.23270441](https://doi.org/10.5281/zenodo.23270441).

GraphCast (0.25°) is made differentiable in JAX and its initial state is optimised in a 4D-Var framework
(background error = ERA5 ensemble spread, 250-km Gaussian B^1/2) to fit gauge-adjusted radar rainfall for the
July 2021 floods (Alzette, Ahr, Vesdre). The rainfall is routed through LISFLOOD-FP, the change is tested for
robustness, hydrostatic balance and specificity, and inserted into WRF 4.5.2.

## Layout
| Path | Content |
|---|---|
| `scripts/hpc/gc_optimise_ic.py` | optimiser and all GraphCast tests (`--mode optimise / forward / robust / balance / ensic / gradtest / members`) |
| `scripts/hpc/*.sbatch` | ECMWF Atos job scripts (ERA5/EDA retrieval from MARS, chains, flood routing) |
| `scripts/wrf/` | WRF transfer test: perturbing the WPS intermediate files, soil/SST repair, case runs, rain extraction, statistics |
| `scripts/local/` | verification, statistics and the figures/tables of the paper (`11_paper_figures.py`, `12_si.py`) |
| `scripts/fri/` | exploratory flood-reachability-index pilot (not part of the paper) |
| `configs/` | target definitions (`targets.json`) and the run lists |
| `results/runs/` | per-run summaries (`summary.json`, `robust_/balance_/ensic_summary.json`) and GraphCast rain (`rain_control_vs_optimised.nc`) |
| `results/wrf_cases/` | WRF basin rain per case (`rain_summary.json`, `rain_025.nc`) |
| `results/flood/` | simulated discharge at the gauges (`station_Q.csv`) |
| `results/*.csv` | ECMWF basin rain, ERA5 6-h test, null tests, flood and ensemble statistics |
| `manuscript/figures/` | figures of the paper |

The large optimised initial-state increments (`increment.nc`, approximately 500 MB per file) are not included in this repository or its automatic GitHub-to-Zenodo archive. They require a separate data deposit.

## Data not included
ERA5, its ensemble spread and ECMWF operational forecasts (ECMWF MARS / Copernicus CDS), RADKLIM (DWD open data),
RMI radar and AGE river discharge (on request from RMI and the Administration de la gestion de l'eau, Luxembourg),
GraphCast weights (Google DeepMind, CC BY-NC-SA 4.0). Scripts contain paths of the authors' ECMWF HPC account; adapt them.

## Software
Python 3.11, JAX 0.4.14, dm-haiku, graphcast (ai-models 0.49 environment), xarray; WRF/WPS 4.5.2/4.5; LISFLOOD-FP 8.

## Licence
Code: MIT (`LICENSE`). Derived data and figures: CC BY 4.0 (`LICENSE-DATA`).

## Funding
Fonds National de la Recherche Luxembourg (FNR), Industrial Fellowship, Project No. 17130773.
