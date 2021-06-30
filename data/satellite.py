import json
import torch
import os
import numpy as np
import warnings
import xarray as xr

from pathlib import Path
from typing import List, Optional
from .convert_satellite import BANDS

class SatelliteData(object):
    
    """Data Handler that loads satellite data."""
    bands_to_remove = ["B1", "B10"]
    bands_to_keep=["B4", "B3", "B2"]

    def __init__(self, data_root, train=True, seq_len=12):

        with (Path(data_root) / "normalizing_dict.json").open() as f:
            normalizing_dict = json.load(f)
            bands = len(normalizing_dict["mean"])
            self.mean = np.array(normalizing_dict["mean"]).reshape(bands, 1, 1)
            self.std = np.array(normalizing_dict["std"]).reshape(bands, 1, 1)

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
    def _calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, bands]
        where bands == len(BANDS), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """
        b08 = input_array[:, BANDS.index("B8")]
        b04 = input_array[:, BANDS.index("B4")]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndvi = np.where((b08 + b04) > 0, (b08 - b04) / (b08 + b04), 0,)
        return ndvi

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
        return (array - self.mean / self.std)
          
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


    @classmethod
    def _remove_bands(cls, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands, size, size]
        """
        if len(cls.bands_to_keep) > 0:
            indices_to_keep = [BANDS.index(band) for band in cls.bands_to_keep]  
        else:
            indices_to_remove = [BANDS.index(band) for band in cls.bands_to_remove]
            indices_to_keep = [i for i in range(x.shape[1]) if i not in indices_to_remove]
        return x[:, indices_to_keep]


    def __getitem__(self, index):
        self.set_seed(index)
        rand_i = np.random.randint(len(self.nc_files))
        tile = xr.open_dataarray(self.nc_files[rand_i]).values
        assert tile.shape == (12, 13, 64, 64)

        tile = self._normalize(tile)
        
        ndvi = self._calculate_ndvi(tile)
        assert ndvi.shape == (12, 64, 64)
        
        tile = np.concatenate([tile, np.expand_dims(ndvi, axis=1)], axis=1)
        assert tile.shape == (12, 14, 64, 64)

        tile = self.remove_bands(tile)
        assert tile.shape == (12, 3, 64, 64)
        
        return torch.from_numpy(tile)



