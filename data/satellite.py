import json
import torch
import os
import numpy as np
import warnings
import xarray as xr

from pathlib import Path
from rasterio.plot import reshape_as_image
from typing import Optional
from .convert_satellite import BANDS_WITH_NDVI


RGB_BANDS = ["B4", "B3", "B2"]
ALL_BANDS = BANDS_WITH_NDVI
BANDS_WITH_NO_AIR = [b for b in BANDS_WITH_NDVI if b not in ["B1", "B10"]]

class SatelliteData(object):
    
    """Data Handler that loads satellite data."""

    def __init__(self, data_root, bands_to_keep=RGB_BANDS, train=True, seq_len=12, skip_normalize=False, no_randomization=False):

        self.bands_to_keep = bands_to_keep
        self.seq_len = seq_len
        self.skip_normalize = skip_normalize
        self.no_randomization = no_randomization

        with (Path(data_root) / "normalizing_dict.json").open() as f:
            normalizing_dict = json.load(f)
            bands = len(normalizing_dict["mean"])
            all_mean = np.array(normalizing_dict["mean"]).reshape(bands, 1, 1)
            all_std = np.array(normalizing_dict["std"]).reshape(bands, 1, 1)
            norm_index = [BANDS_WITH_NDVI.index(b) for b in bands_to_keep]
            self.std = all_std[norm_index]
            self.mean = all_mean[norm_index]

        if train:
            data_root_subset = Path(data_root) / "train" 

        else:
            data_root_subset = Path(data_root) / "test" 

        
        self.nc_files = [str(i) for i in data_root_subset.glob("*.nc")]

        print(f"Using: {len(self.nc_files)} for {'training' if train else 'testing'}")
        self.seed_is_set = True 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)


    @staticmethod
    def _maxed_nan_to_num(
        array: np.ndarray, nan: float, max_ratio: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if max_ratio is not None:
            num_nan = np.count_nonzero(np.isnan(array))
            if (num_nan / array.size) > max_ratio:
                return None
        return np.nan_to_num(array, nan=nan)

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std)

    def unnormalize(self, array: np.ndarray) -> np.ndarray:
        return (array * self.std) + self.mean

          
    def __len__(self):
        return len(self.nc_files)

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """This nested function is so that
        _remove_bands can be called from an unitialized
        dataset, speeding things up at inference while still
        keeping the convenience of not having to check if remove
        bands is true all the time.
        """

        if self.remove_bands:
            return self._remove_bands(x)
        else:
            return x


    def _remove_bands(self, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands, size, size]
        """
        indices_to_keep = [BANDS_WITH_NDVI.index(band) for band in self.bands_to_keep]  
        return x[:, indices_to_keep]


    def __getitem__(self, index):
        if not self.no_randomization:
            self.set_seed(index)
            rand_i = np.random.randint(len(self.nc_files))
            file = self.nc_files[rand_i]
        else:
            file = self.nc_files[index]
        tile = xr.open_dataarray(file).values
        assert tile.shape == (60, len(BANDS_WITH_NDVI), 64, 64)

        tile = tile[0:self.seq_len]
        assert tile.shape == (self.seq_len, len(BANDS_WITH_NDVI), 64, 64), tile.shape

        tile = self.remove_bands(tile)
        assert tile.shape == (self.seq_len, len(self.bands_to_keep), 64, 64), tile.shape

        if not self.skip_normalize:
            tile = self._normalize(tile)
        
        return torch.from_numpy(tile)

    def get_nc(self, index):
        if not self.no_randomization:
            self.set_seed(index)
            rand_i = np.random.randint(len(self.nc_files))
            file = self.nc_files[rand_i]
        else:
            file = self.nc_files[index]
        return xr.open_dataarray(file)


    @classmethod
    def normalize(cls, array):
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    def for_viewing(self, tile: np.ndarray, unnormalize: bool = True) -> np.ndarray:
        if unnormalize:
            tile = self.unnormalize(tile)

        # Extract reference to Red, Green, Blue in image
        rgb_index = np.array([self.bands_to_keep.index(b) for b in RGB_BANDS])
        colors = tile[rgb_index, :, :].astype(np.float64)
        colors = colors*10000

        # Enforce maximum and minimum values
        max_val = 5000
        min_val = 0
        colors[colors[:, :, :] > max_val] = max_val
        colors[colors[:, :, :] < min_val] = min_val

        # Normalize
        for b in range(colors.shape[0]):
            colors[b, :, :] = colors[b, :, :] * 1 / (max_val - min_val)

        return colors


