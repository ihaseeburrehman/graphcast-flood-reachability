"""E-OBS daily precipitation (ensemble mean, 0.1 deg) from the Copernicus CDS, as the
independent gauge-based cross-check for the 2024 events. Uses ~/.cdsapirc."""
from pathlib import Path
import cdsapi
out = Path(__file__).resolve().parents[2] / "data/truth_2024"
c = cdsapi.Client()
for version in ("31_0e", "30_0e"):
    try:
        c.retrieve("insitu-gridded-observations-europe",
                   {"product_type": "ensemble_mean", "variable": ["precipitation_amount"],
                    "grid_resolution": "0_1deg", "period": "2011_2024", "version": [version]},
                   str(out / f"eobs_rr_{version}.zip"))
        print("downloaded E-OBS", version); break
    except Exception as e:
        print("version", version, "failed:", str(e)[:200])
