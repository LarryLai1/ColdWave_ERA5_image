```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# it may take around 30 mins
python download_era5_201601.py # 332 MB
python plot_weather_maps.py
```