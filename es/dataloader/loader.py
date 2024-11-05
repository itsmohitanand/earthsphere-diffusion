import lightning as L

from es.dataloader.core import ZarrWindSpeed
from es.utils.aux import console
from torch.utils.data import DataLoader
from es.dataloader.scaler import NormScaler
import zarr
import xarray as xr
import numpy as np
from einops import rearrange
from datetime import datetime
from typing import Tuple
from pathlib import Path

import torch 

class HighResWindSpeed(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.static_features = self.load_static_features()
        self.batch_size = self.cfg.experiment.dataset.batch_size

    def setup(self, stage:str):
        
        scaler = self.load_scaler()

        if scaler:
            console.print("Scaling params found")
        else:
            console.print("Scaling params not found, writing them")
            self.write_scaling_params()
            scaler = self.load_scaler()

        self.scaler = scaler


    def train_dataloader(self):

        train_data = zarr.open(self.cfg.path.root / self.cfg.path.data.folder / self.cfg.path.data.train_file)
        
        start_ind, end_ind = self.get_start_end_ind(train_data, self.cfg.experiment.dataset.split.train)
        
        dataset = ZarrWindSpeed(
                data = train_data,
                time_step = 1,
                start_ind = start_ind,
                end_ind = end_ind,
                static_features = self.static_features,
                scaler = self.scaler,
                var_list = self.cfg.experiment.dataset.dynamic_variables
                )
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        train_data = zarr.open(self.cfg.path.root / self.cfg.path.data.folder / self.cfg.path.data.train_file)
        start_ind, end_ind = self.get_start_end_ind(train_data, self.cfg.experiment.dataset.split.val)

        dataset = ZarrWindSpeed(
                data = train_data,
                time_step = 1,
                start_ind = start_ind,
                end_ind = end_ind,
                static_features = self.static_features,
                scaler = self.scaler,
                var_list = self.cfg.experiment.dataset.dynamic_variables
                )
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        test_data = zarr.open(self.cfg.path.root / self.cfg.path.data.folder / self.cfg.path.data.test_file)
        start_ind, end_ind = self.get_start_end_ind(test_data, self.cfg.experiment.dataset.split.test)
        dataset = ZarrWindSpeed(
            data=test_data,
            time_step=1,
            start_ind=start_ind,
            end_ind=end_ind,
            static_features=self.static_features,
            scaler=self.scaler,
            var_list = self.cfg.experiment.dataset.dynamic_variables
        )

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
    
    def load_static_features(self, ):

        data_path = Path(self.cfg.path.root) / self.cfg.path.data.folder / self.cfg.path.data.train_file

        lsm = torch.tensor(zarr.open(data_path)["land_sea_mask"][:, :])

        assert lsm.shape == (721, 1440)
        
        
        lsm = rearrange(lsm, "h w -> 1 h w")
        return lsm

    def load_scaler(self, ):
        file_name = self.cfg.experiment.dataset.scaler.filename

        out_path = Path(self.cfg.path.root) / "aux_data" / file_name

        if out_path.exists():
            scaler = globals()[self.cfg.experiment.dataset.scaler.name](out_path)
            
            scaler.load_params(self.cfg.experiment.dataset.dynamic_variables)
            return scaler
        else:
            return None

    def write_scaling_params(self, ):
        file_name = self.cfg.experiment.dataset.scaler.filename

        out_path = Path(self.cfg.path.root) / "aux_data" / file_name

        high_res_wind_speed = xr.open_zarr(Path(self.cfg.path.root) / self.cfg.path.data.folder / self.cfg.path.data.train_file)

        variables = self.cfg.experiment.dataset.dynamic_variables
        high_res_wind_speed = high_res_wind_speed[variables]
        train_years = np.arange(self.cfg.experiment.dataset.split.train[0], self.cfg.experiment.dataset.split.train[0]+1)
        high_res_wind_speed = high_res_wind_speed.sel(time=high_res_wind_speed.time.dt.year.isin(train_years))

        scaler = globals()[self.cfg.experiment.dataset.scaler.name](out_path)

        scaler.write_params(high_res_wind_speed)

    
    def get_start_end_ind(self, data: zarr.hierarchy.Group, split: Tuple):
        """
        Get the start and end indices of the data based on the given split.
        Args:
            data (zarr.hierarchy.Group): The data group.
            split (Tuple): A tuple representing the split, where the first element is the start year and the second element is the end year.
        Returns:
            Tuple[int, int]: A tuple containing the start and end indices of the data.
        Raises:
            AssertionError: If the number of days between the start and end time does not match the difference between the start and end indices.
        """

        base = datetime(1959, 1, 1)
        start_time = datetime(split[0], 1, 1)
        end_time = datetime(split[1], 12, 31)

        start_stamp = int((start_time - base).total_seconds()/(60*60))
        end_stamp = int((end_time - base).total_seconds()/(60*60))


        start_ind = np.argwhere(data.time[:] == start_stamp)[0][0]
        end_ind = np.argwhere(data.time[:] == end_stamp)[0][0]

        assert (end_time - start_time).days*4 == (end_ind - start_ind)

        self.check_temporally_sorted(data.time[start_ind:end_ind], temporal_frequency=1/4)

        return start_ind, end_ind

    def check_temporally_sorted(self, time: np.array, temporal_frequency: float = 1):
        """
        Check if the given time array is sorted in a temporal order with a specified frequency.

        Parameters:
            time (np.array): The array of time values.
            temporal_frequency (float, optional): The temporal frequency in days. Defaults to 1.

        Raises:
            AssertionError: If the time array is not sorted in a temporal order with the specified frequency.

        Returns:
            None
        """


        assert (
            np.diff(time) == int(temporal_frequency * 24)
        ).sum() == time.shape[0] - 1
        console.print("Temporal check passed", style="info")


