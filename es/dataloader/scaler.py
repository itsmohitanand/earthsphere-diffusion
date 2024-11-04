from abc import ABC, abstractmethod
import xarray as xr
from dask.diagnostics import ProgressBar
import torch
from es.utils.aux import console
from einops import rearrange

class Scaler(ABC):
    @abstractmethod
    def _calculate_params(
        self,
    ):
        pass

    def load_params(self, variable_list):
        """
        Load parameters from a zarr file.
        Args:
            variable_list (list): List of variables to load.
        Returns:
            None
        """

        if self.param_path.exists():
            param_var_list = []
            for params in self.param_list:
                param_var_list.extend([params + "_" + var for var in variable_list])

        data = xr.open_zarr(
            self.param_path,
            chunks={"longitude": 1440, "latitude": 721},
        )

        self.data = data.transpose("longitude", "latitude")

    def write_params(self, data):
        """
        Write the calculated parameters to a zarr file.
        Parameters:
        - data: The input data used to calculate the parameters.
        Returns:
        None
        """

        params = self._calculate_params(data)
        for p in params:
            p.to_zarr(self.param_path, mode="a")

    def __bool__(self):
        if self.data is None:
            return False
        else:
            return True

class NormScaler(Scaler):
    """
    A class for performing min-max scaling on data.
    Args:
        param_path (str): The path to the parameter file.
        scaling_interval (tuple): The scaling interval.
    Attributes:
        data: The data to be scaled.
        param_list (list): The list of parameter names.
        scaling_interval (tuple): The scaling interval.
        param_path (str): The path to the parameter file.
        min_max_data: The min-max scaled data.
    Methods:
        _calculate_params(data): Calculates the minimum and maximum values of the data.
        apply(data): Applies min-max scaling to the data.
        create_min_max_array(variable_list): Creates the min-max arrays for the specified variables.
    """

    def __init__(self, param_path):
        """
        Initializes an instance of the Scaling class.
        Parameters:
        - param_path (str): The path to the parameter file.
        - scaling_interval (float): The scaling interval.
        Attributes:
        - data: The data.
        - param_list (list): The list of parameters.
        - scaling_interval (float): The scaling interval.
        - param_path (str): The path to the parameter file.
        - min_max_data: The min-max data.
        """

        self.data = None
        self.param_list = ["mean", "std"]
        self.param_path = param_path
        self.mean_arr = None
        self.std_arr = None

    def _calculate_params(self, data):
        """
        Calculate the minimum and maximum values for each variable in the given data.
        Parameters:
        - data: xarray.Dataset
            The input data containing variables.
        Returns:
        - min_data: xarray.Dataset
            The dataset containing the minimum values for each variable.
        - max_data: xarray.Dataset
            The dataset containing the maximum values for each variable.
        """

        mean_data = data.mean(dim="time")
        std_data = data.std(dim="time")

        console.print("Calculating mean", style="info")
        with ProgressBar():
            mean_data = mean_data.compute()

        var_name_dict = {}
        for variable_names in list(data.keys()):
            var_name_dict[variable_names] = "mean_" + variable_names

        mean_data = mean_data.rename(var_name_dict)

        console.print("Calculating std", style="info")

        with ProgressBar():
            std_data = std_data.compute()

        var_name_dict = {}
        for variable_names in list(data.keys()):
            var_name_dict[variable_names] = "std_" + variable_names

        std_data = std_data.rename(var_name_dict)

        return mean_data, std_data

    def scale(self, data):
        """
        Apply scaling to the given data.
        Parameters:
        - data: A torch.Tensor object representing the data to be scaled.
        Returns:
        - std: A torch.Tensor object representing the scaled data.
        Note:
        - The scaling is performed using the minimum and maximum values stored in self.min_arr and self.max_arr.
        - The scaled data is normalized between 0 and 1.
        - To scale the data within a custom interval, uncomment the code block and modify the scaling_interval values.
        """
        assert isinstance(self.mean_arr, torch.Tensor)
        assert isinstance(self.std_arr, torch.Tensor)

        norm_data = (data - self.mean_arr) / self.std_arr
        
        return norm_data

    def create_min_std_array(self, variable_list):
        """
        Creates min and max arrays for the given list of variables.
        Args:
            variable_list (list): List of variables for which min and max arrays need to be created.
        Returns:
            None
        """
        
        mean_data = []
        std_data = []

        for feature in variable_list:
            mean_xr = torch.from_numpy(self.data["mean_" + feature].values)
            std_xr = torch.from_numpy(self.data["std_" + feature].values)
            
            mean_xr = rearrange(mean_xr, 'w h -> 1 w h')
            std_xr = rearrange(std_xr, 'w h -> 1 w h')

            mean_data.append(mean_xr)
            std_data.append(std_xr)

        self.mean_arr = torch.cat(mean_data).permute((0, 2, 1))
        self.std_arr = torch.cat(std_data).permute((0, 2, 1))