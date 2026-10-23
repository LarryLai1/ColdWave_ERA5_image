import argparse
import time
from pathlib import Path
import cdsapi
import xarray as xr
from tqdm import tqdm

# CDS variable name -> short name in the netCDF file
var_sfc = {
    'mean_sea_level_pressure': 'msl',
}
var_upper = {
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'vertical_velocity': 'w',
    'temperature': 't',
    'specific_humidity': 'q',
    'geopotential': 'z',
    'divergence': 'd',
}
pressure_level = [1000, 850, 500, 200]

YEAR = '2016'
MONTH = '01'
DAYS = [f'{d:02}' for d in range(19, 27)]          # 01/19 ~ 01/26
TIMES = ['00:00', '06:00', '12:00', '18:00']       # every 6 hours

# [North, West, South, East]; 160°W = 200°E (crosses the dateline)
AREA = [70, 40, -10, 200]
GRID = [0.25, 0.25]

def retrieve_with_retry(
        client: cdsapi.Client,
        dataset: str,
        request: dict,
        file_path: str,
        max_retries = 10
    ) -> bool:
    retries = 0
    while retries < max_retries:
        try:
            client.retrieve(dataset, request, file_path)
            tqdm.write(f'Data successfully retrieved and saved to {file_path}.')
            return True
        except Exception as e:
            retries += 1
            tqdm.write(f'Attempt {retries} failed: {e}.')
            tqdm.write('Retrying in 10 seconds...')
            time.sleep(10)
    tqdm.write(f'Failed to retrieve data after {max_retries} attempts.')
    return False

def find_missing(file_path: Path, variables: dict) -> list:
    """Return the CDS variable names whose short names are absent in file_path."""
    if not file_path.exists():
        return list(variables)
    with xr.open_dataset(file_path) as ds:
        existing = set(ds.data_vars)
    return [name for name, short in variables.items() if short not in existing]

def download_missing(
        client: cdsapi.Client,
        dataset: str,
        variables: dict,
        file_path: Path,
        extra_request: dict = None,
    ):
    missing = find_missing(file_path, variables)
    if not missing:
        tqdm.write(f'{file_path.name}: all variables present, skip.')
        return
    tqdm.write(f'{file_path.name}: missing {[variables[m] for m in missing]}, downloading...')

    request = {
        'variable': missing,
        'product_type': 'reanalysis',
        'year': YEAR,
        'month': MONTH,
        'day': DAYS,
        'time': TIMES,
        'area': AREA,
        'grid': GRID,
        'download_format': 'unarchived',
        'format': 'netcdf',
        **(extra_request or {}),
    }

    if not file_path.exists():
        retrieve_with_retry(client, dataset, request, file_path)
        return

    # Download missing variables to a temp file, then merge into the existing file
    new_path = file_path.with_suffix('.new.nc')
    merged_path = file_path.with_suffix('.merged.nc')
    if not retrieve_with_retry(client, dataset, request, new_path):
        return
    with xr.open_dataset(file_path) as old, xr.open_dataset(new_path) as new:
        merged = xr.merge([old, new[[variables[m] for m in missing]]],
                          join='exact', compat='override')
        merged.to_netcdf(merged_path)
    merged_path.replace(file_path)
    new_path.unlink()
    tqdm.write(f'Merged {[variables[m] for m in missing]} into {file_path}.')

def main(save_dir: str):
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    C = cdsapi.Client()
    tag = f'{YEAR}{MONTH}{DAYS[0]}-{YEAR}{MONTH}{DAYS[-1]}'

    jobs = [
        ('reanalysis-era5-single-levels', var_sfc, output_dir / f'{tag}_sfc.nc', None),
        ('reanalysis-era5-pressure-levels', var_upper, output_dir / f'{tag}_upper.nc',
         {'pressure_level': pressure_level}),
    ]
    # outer bar: files done; cdsapi shows its own byte-level bar while downloading
    with tqdm(jobs, desc='ERA5 files', unit='file') as bar:
        for dataset, variables, file_path, extra in bar:
            bar.set_postfix_str(file_path.name)
            download_missing(C, dataset, variables, file_path, extra_request=extra)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', type=str, default='./era5_data',
        help='Directory to save the downloaded data.',
    )
    args = parser.parse_args()
    main(**vars(args))
