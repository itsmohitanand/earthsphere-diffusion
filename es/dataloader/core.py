from datetime import datetime, timedelta
import torch
import zarr
from torch.utils.data import Dataset


# The dataloder does not output the difference but rather outputs the actual wind speed at a time interval t
# Only windspeed dataset so only required one scaler

import numpy as np

class ZarrWindSpeed(Dataset):
    def __init__(self, data, time_step, start_ind, end_ind, static_features, scaler, var_list):
        
        self.data = data
        self.time_step = time_step
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.static_features = static_features
        self.scaler = scaler
        self.var_list = var_list

        self.scaler.create_min_std_array(var_list)

    def __len__(self):
        return self.end_ind - self.start_ind - self.time_step
    
    def __getitem__(self, idx):
        idx = idx+self.start_ind

        x_cond = torch.zeros((2, 721, 1440))
        x_tar = torch.zeros((2, 721, 1440))

        # TO be used later
        # t_0 = datetime(1959, 1, 1) + timedelta(seconds=int(self.data.time[idx]))

        # day_0 = t_0.timetuple().tm_yday
        # hour_0 = t_0.timetuple().tm_hour

        # sin_day = np.sin(2 * np.pi * day_0 / 366)
        # cos_day = np.cos(2 * np.pi * day_0 / 366)
        # sin_hour = np.sin(2 * np.pi * hour_0 / 24)
        # cos_hour = np.cos(2 * np.pi * hour_0 / 24)

        for i, var in enumerate(self.var_list):
            x_cond[i] = torch.tensor(self.data[var][idx])
            x_tar[i] = torch.tensor(self.data[var][idx + self.time_step])

        x_cond = self.scaler.scale(x_cond)
        x_tar = self.scaler.scale(x_tar)

        x_cond = torch.cat((x_cond, self.static_features), dim=0)

        return x_cond[:, 1:, :], x_tar[:, 1:, :] # to convert 721 x 1440 to 720 x 1440