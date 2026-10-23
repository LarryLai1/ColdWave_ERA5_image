## Step 1: Setup CDSAPI
follow https://cds.climate.copernicus.eu/how-to-api

## Step 2: Execute
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# it may take around 30 mins
python download_era5_201601.py # 332 MB
python plot_weather_maps.py
```
