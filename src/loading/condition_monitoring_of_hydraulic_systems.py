"""
Module implementing a Dataloader for the Condition Monitoring of Hydraulic Systems dataset.

The dataset can be found under:
https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

Dataset description:
--------------------
General Information:
The dataset consists of 2205 samples. Each sample is a 60 second measurement of 17 sensors.

Features:
Three different sampling frequencies are used: 1 Hz, 10 Hz, 100 Hz
The following sensors are used:
- PS1, PS2, PS3, PS4, PS5, PS6: Pressure [bar] -> 100 Hz
- EPS1: Motor power [W] -> 100 Hz
- FS1, FS2: Volume flow [l/min] -> 10 Hz
- TS1, TS2, TS3, TS4: Temperature [°C] -> 1 Hz
- VS1: Vibration [mm/s] -> 1 Hz
- CE: Cooler efficiency [%] -> 1 Hz
- CP: Cooling power [kW] -> 1 Hz
- SE: Efficiency factor -> 1 Hz

Target Variables:
The dataset has 5 different target variables (all classification):
- Cooler condition / % efficiency -> 3 classes
- Valve condition / % efficiency -> 4 classes
- Internal pump leakage -> 3 classes
- Hydraulic accumulator / bar -> 4 classes
- Stable flag -> 2 classes

"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

class Dataloader:
    """
    Data loader for the Condition Monitoring of Hydraulic Systems dataset.

    The dataset can be found under:
    https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

    Dataset description:
    --------------------
    General Information:
    The dataset consists of 2205 samples. Each sample is a 60 second measurement of 17 sensors.

    Features:
    Three different sampling frequencies are used: 1 Hz, 10 Hz, 100 Hz
    The following sensors are used:
    - PS1, PS2, PS3, PS4, PS5, PS6: Pressure [bar] -> 100 Hz
    - EPS1: Motor power [W] -> 100 Hz
    - FS1, FS2: Volume flow [l/min] -> 10 Hz
    - TS1, TS2, TS3, TS4: Temperature [°C] -> 1 Hz
    - VS1: Vibration [mm/s] -> 1 Hz
    - CE: Cooler efficiency [%] -> 1 Hz
    - CP: Cooling power [kW] -> 1 Hz
    - SE: Efficiency factor -> 1 Hz

    Target Variables:
    The dataset has 5 different target variables (all classification):
    - Cooler condition / % efficiency -> 3 classes
    - Valve condition / % efficiency -> 4 classes
    - Internal pump leakage -> 3 classes
    - Hydraulic accumulator / bar -> 4 classes
    - Stable flag -> 2 classes

    Attributes
    ----------
    feature_sampling_rates : dict[str, int]
        A dictionary containing the sampling rate for each feature.
    target_file : str
        The name of the file containing the target data.
    url : str
        The URL to download the dataset from.

    Methods
    -------
    load_dataset(dir_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]
        Load the Condition Monitoring of Hydraulic Systems dataset.
    """

    feature_sampling_rates = {
        "PS1": 100,
        "PS2": 100,
        "PS3": 100,
        "PS4": 100,
        "PS5": 100,
        "PS6": 100,
        "EPS1": 100,
        "FS1": 10,
        "FS2": 10,
        "TS1": 1,
        "TS2": 1,
        "TS3": 1,
        "TS4": 1,
        "VS1": 1,
        "CE": 1,
        "CP": 1,
        "SE": 1,
    }
    target_file = "profile.txt"
    url = "https://archive.ics.uci.edu/static/public/447/" + "condition+monitoring+of+hydraulic+systems.zip"

    def load_dataset(self, dir_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the Condition Monitoring of Hydraulic Systems dataset. This method will download
        the dataset if it is not already downloaded.

        Parameters
        ----------
        dir_path : Path
            Path, where the dataset will be located. If this directory is empty,
            the dataset will be downloaded to this directory. If the directory does not exist,
            it will be created.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the feature data as a dictionary of DataFrames and the
            target data as a DataFrame.
        """
        # Check if the directory already exists, if not, create it
        dir_path = Path(dir_path)
        if not dir_path.exists():
            os.makedirs(dir_path)

        # Check if the directory is empty --> download necessary
        if not os.listdir(dir_path):
            self._download_dataset(dir_path)

        # Load the dataset
        features, targets = self._load_from_dir(dir_path)

        return features, targets

    def _download_dataset(self, dir_path: Path) -> None:
        """
        Download the Condition Monitoring of Hydraulic Systems dataset from the UCI
        Machine Learning Repository.

        Parameters
        ----------
        dir_path : Path
            The directory to save the dataset to.
        """
        print(f"Downloading the Condition Monitoring of Hydraulic Systems dataset to {dir_path}...")

        # Download the dataset
        response = requests.get(self.url, timeout=10)
        zip_file = dir_path / "data.zip"

        with open(zip_file, "wb") as file:
            file.write(response.content)

        # Unzip the dataset
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        # Remove the zip file
        zip_file.unlink()

    def _load_from_dir(self, dir_path: Path | str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the Condition Monitoring of Hydraulic Systems dataset from a directory.

        Parameters
        ----------
        dir_path : Path | str
            The directory containing the unzipped dataset.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the feature data as a dictionary of DataFrames and the
            target data as a DataFrame.
        """
        print(f"Loading the Condition Monitoring of Hydraulic Systems dataset from {dir_path}...")

        # Load the features
        features = {}
        for feature_name, sampling_rate in tqdm(self.feature_sampling_rates.items()):
            feature_file = Path(dir_path) / f"{feature_name}.txt"
            features[feature_name] = self._load_feature(feature_file, sampling_rate)

        # Load the target variables
        target_file = Path(dir_path) / self.target_file
        targets = self._load_targets(target_file)

        return features, targets

    def _load_feature(self, feature_file: Path, sampling_rate: int) -> pd.DataFrame:
        """
        Load a feature from a file.

        Parameters
        ----------
        feature_file : Path
            The file containing the feature data.
        sampling_rate : int
            The sampling rate of the feature.

        Returns
        -------
        pd.DataFrame
            The feature data as a DataFrame
        """
        # Load the feature data
        feature = np.loadtxt(feature_file)
        assert feature.shape[0] == 2205
        assert feature.shape[1] == 60 * sampling_rate

        # Convert to stacked dataframe with one column
        feature = pd.DataFrame(feature.flatten(), columns=["value"])

        # Add a measurement index using the sampling rate for tsfresh
        # Each measuremement is 60 seconds long
        feature["measurement_idx"] = ((feature.index // (60 * sampling_rate)) + 1).astype(str)

        return feature

    def _load_targets(self, target_file: Path) -> pd.DataFrame:
        """
        Load the target variables from a file.

        Parameters
        ----------
        target_file : Path
            The file containing the target data.

        Returns
        -------
        pd.DataFrame
            The target data as a DataFrame
        """
        # Load the target data
        targets = pd.read_csv(
            target_file,
            sep="\t",
            header=None,
            names=[
                "cooler_condition",
                "valve_condition",
                "pump_leakage",
                "accumulator_condition",
                "stable_flag",
            ],
        )
        assert targets.shape[0] == 2205
        assert targets.shape[1] == 5

        return targets
