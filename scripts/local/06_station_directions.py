"""Gauge cross-section orientation, computed once with the multi-model paper's own extractor.

Imports extract_discharge_line_integral.py unchanged, points RIVER at the local copy of the
same centreline shapefile, and saves station coordinates, widths and downstream unit
vectors (plus the 10 m grid header) so discharge can be extracted on the HPC with numpy only.
"""
import json, sys
from pathlib import Path
import numpy as np
PAPER = Path("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/analysis")
sys.path.insert(0, str(PAPER / "scripts"))
import extract_discharge_line_integral as X
X.RIVER = ("/Users/haseeb.rehman/Documents/Misc/Lisflood_Simulations/Lisflood_Alzette_river_basin/"
           "sub_basins/5m/sub_basin_complete/pre_processing/alzette_river.shp")
qx = sorted((PAPER / "data/lisflood_96h/runs/graphcast/results").glob("*.Qx"))[0]
h = X.header(qx)
bed = np.load(PAPER / "data/lisflood_96h/bed_elevation.npy")
dirs = X.centreline_dirs(bed, h)
out = dict(header=h, start_utc="2021-07-13T18:00:00Z", interval_h=X.INTERVAL_H,
           stations={n: dict(x=x, y=y, width_m=W, ux=dirs[n][0], uy=dirs[n][1]) for n, (x, y, W) in X.STATIONS.items()},
           source="extract_discharge_line_integral.centreline_dirs (paper code), local alzette_river.shp")
p = Path(__file__).resolve().parents[2] / "data/gauges/stations.json"
p.write_text(json.dumps(out, indent=1)); print(p); print(json.dumps(out["stations"], indent=1))
