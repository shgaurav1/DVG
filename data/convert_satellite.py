from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

import json
import numpy as np
import pandas as pd
import re
import xarray as xr

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

def get_start_date(path: Path):
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", path.stem)
    if len(dates) != 2:
        raise ValueError(f"{uri} should have start and end date")
    start_date_str, _ = dates
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    return start_date

def load_tif(local_path: Path, days_per_timestep: int = 30) -> xr.DataArray:
    r"""
    The sentinel files exported from google earth have all the timesteps
    concatenated together. This function loads a tif files and splits the
    timesteps
    """

    # this mirrors the eo-learn approach
    # also, we divide by 10,000, to remove the scaling factor
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
    da = xr.open_rasterio(str(local_path)).rename("FEATURES") / 10000

    bands_per_timestep = len(BANDS)
    num_bands = len(da.band)

    assert (num_bands % bands_per_timestep == 0), "Total number of bands not divisible by the expected bands per timestep"

    da_split_by_time: List[xr.DataArray] = []
    cur_band = 0
    while cur_band + bands_per_timestep <= num_bands:
        time_specific_da = da.isel(band=slice(cur_band, cur_band + bands_per_timestep))
        time_specific_da["band"] = range(bands_per_timestep)
        da_split_by_time.append(time_specific_da)
        cur_band += bands_per_timestep

    start_date = get_start_date(local_path)
    timesteps = [start_date + timedelta(days=days_per_timestep) * i for i in range(len(da_split_by_time))]

    combined = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
    combined.attrs["band_descriptions"] = BANDS
    return combined

def tif_to_tiles(tif: xr.DataArray, tile_size: int = 64) -> xr.DataArray:
    tif_x_len = tif.shape[-2]
    tif_y_len = tif.shape[-1]
    assert tif_x_len > tile_size
    assert tif_y_len > tile_size

    x_ranges = [(x-tile_size, x) for x in range(tile_size, tif_x_len, tile_size)]
    y_ranges = [(y-tile_size, y) for y in range(tile_size, tif_y_len, tile_size)]
    tiles = [tif[:, :, x0:x1, y0:y1] for x0, x1 in x_ranges for y0, y1 in y_ranges]
    return tiles

def tile_name(tile: xr.DataArray):
    date = tile.time[0].dt.strftime("%Y-%m-%d").item()
    x = str(tile.x.mean().item()).replace(".", "_")
    y = str(tile.y.mean().item()).replace(".", "_")
    return f"{date}_{x}_{y}.nc"


def update_normalizing_values(norm_dict: Dict[str, Union[np.ndarray, int]], array: np.ndarray) -> Dict[str, Union[np.ndarray, int]]:
    # given an input array of shape [timesteps, bands]
    # update the normalizing dict
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # https://www.johndcook.com/blog/standard_deviation/

    # array will be 12,13,64,64
    num_timesteps = array.shape[0]
    num_bands = array.shape[1]

    # initialize
    if "mean" not in norm_dict:
        norm_dict["mean"] = np.zeros(num_bands)
        norm_dict["M2"] = np.zeros(num_bands)

    for time_idx in range(num_timesteps):
        norm_dict["n"] += 1

        x = array[time_idx].mean(axis=(1,2))

        delta = x - norm_dict["mean"]
        norm_dict["mean"] += delta / norm_dict["n"]
        norm_dict["M2"] += delta * (x - norm_dict["mean"])
        return norm_dict

def calculate_normalizing_dict(norm_dict: Dict[str, Union[np.ndarray, int]]
    ) -> Optional[Dict[str, np.ndarray]]:
        variance = norm_dict["M2"] / (norm_dict["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": norm_dict["mean"].tolist(), "std": std.tolist()}

def main(tif_dir: str, processed_dir: str, tile_size: int = 64):
    all_tif_paths = list(Path(tif_dir).glob("**/*.tif"))
    processed_dir_path = Path(processed_dir)
    
    print(f"Loading {len(all_tif_paths)} tifs")
    split = int(len(all_tif_paths)*0.7)

    train_test_tifs = {
        "train": all_tif_paths[:split],
        "test": all_tif_paths[split:]
    }

    normalizing_dict = {"n": 0}

    for subset, tif_paths in train_test_tifs.items():
        (processed_dir_path / subset).mkdir(exist_ok=True, parents=True)
        for p in tqdm(tif_paths):
            tif = load_tif(p)
            tiles = tif_to_tiles(tif, tile_size)
            for tile in tiles:
                if subset == "train":
                    normalizing_dict = update_normalizing_values(normalizing_dict, tile.values)
                save_path = processed_dir_path / subset / tile_name(tile)
                if save_path.exists():
                    continue
                tile.to_netcdf(str(save_path))

    json_friendly_dict = calculate_normalizing_dict(normalizing_dict)
    print(json_friendly_dict)
    with (processed_dir_path / "normalizing_dict.json").open("w") as fp:
        json.dump(json_friendly_dict, fp)
    
    
    

if __name__ == "__main__":
    tif_dir = "/cmlscratch/izvonkov/forecaster-data"
    processed_dir = "/cmlscratch/izvonkov/forecaster-data-processed-split"
    main(tif_dir, processed_dir)