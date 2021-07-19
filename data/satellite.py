import json
import torch
import os
import numpy as np
import warnings
import xarray as xr

from pathlib import Path
from typing import Optional
from .convert_satellite import BANDS


RGB_BANDS = ["B4", "B3", "B2"]
ALL_BANDS = BANDS
BANDS_WITH_NO_AIR = [b for b in BANDS if b not in ["B1", "B10"]]

class SatelliteData(object):
    
    """Data Handler that loads satellite data."""
    bands_to_keep = BANDS_WITH_NO_AIR

    def __init__(self, data_root, train=True, seq_len=12, skip_normalize=False, no_randomization=False):

        self.seq_len = seq_len
        self.skip_normalize = skip_normalize
        self.no_randomization = no_randomization

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
        return ((array - self.mean) / self.std)
          
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
        indices_to_keep = [BANDS.index(band) for band in cls.bands_to_keep]  
        return x[:, indices_to_keep]


    def __getitem__(self, index):
        if not self.no_randomization:
            self.set_seed(index)
            rand_i = np.random.randint(len(self.nc_files))
            file = self.nc_files[rand_i]
        else:
            file = self.nc_files[index]
        tile = xr.open_dataarray(file).values
        assert tile.shape == (self.seq_len, 14, 64, 64)
        
        #ndvi = self._calculate_ndvi(tile)
        #assert ndvi.shape == (self.seq_len, 64, 64)

        if not self.skip_normalize:
            tile = self._normalize(tile)

        tile = self.remove_bands(tile)
        assert tile.shape == (self.seq_len, len(self.bands_to_keep), 64, 64), tile.shape

        #tile = np.concatenate([tile, np.expand_dims(ndvi, axis=1)], axis=1)
        #assert tile.shape == (self.seq_len, len(self.bands_to_keep)+1, 64, 64), tile.shape
        
        return torch.from_numpy(tile)

    @classmethod
    def normalize(cls, array):
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    @classmethod
    def normalize_for_viewing(cls, tile: np.ndarray, no_transpose = False) -> np.ndarray:
        blue_norm = cls.normalize(tile[cls.bands_to_keep.index("B4")])
        green_norm = cls.normalize(tile[cls.bands_to_keep.index("B3")])
        red_norm = cls.normalize(tile[cls.bands_to_keep.index("B2")])
        img = np.dstack((red_norm, green_norm, blue_norm))
        if no_transpose:
            return img
        else:
            return img.transpose(2,0,1)