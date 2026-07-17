"""
Module implementing a Dataloader for Bosch CNC Machining dataset.

The dataset can be found under:
https://github.com/boschresearch/CNC_Machining

Dataset description:
--------------------
General Information:
The dataset provided is a collection of real-world industrial vibration data collected from a brownfield CNC milling machine.
The acceleration has been measured using a tri-axial accelerometer (Bosch CISS Sensor) mounted inside the machine.
The X- Y- and Z-axes of the accelerometer have been recorded using a sampling rate equal to 2 kHz.
Thereby normal as well as anomoulous data have been collected for 6 different timeframes,
each lasting 6 months from October 2018 until August 2021 and labelled accordingly.
It can be used to investigate the scalability of models and research process variations as the anomaly impact differs.
In total there is data from three different CNC milling machines each executing 15 processes.
For a detailed description of the data and experimental set-up, please refer to the paper.

Features:
The following sensors are used:
- X,Y and Z

Target Variables:
The dataset has 1 target variable:
- machine: M01, M02, M03
- operation: OP00, OP01, ... , OP14
- label: good, bad

Notes
-----

"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import pandas as pd
from git import RemoteProgress, Repo
from tqdm.auto import tqdm

from src.loading.bosch_file_utils import enumerate_files

class CloneProgress(RemoteProgress):
    def __init__(self, total=1972, desc="cloning"):
        super().__init__()
        self.pbar = tqdm(
            ncols=500,
            total=total,
            desc=desc,
        )

    def update(self, op_code, cur_count, max_count=None, message=""):

        self.pbar.total = int(max_count)
        self.pbar.refresh()
        self.pbar.n = int(cur_count)
        self.pbar.refresh()

class Dataloader:
    """
    Dataloader for the Bosch CNC Machining dataset.

    The dataset can be found under:
    https://github.com/boschresearch/CNC_Machining

    Dataset description:
    --------------------
    General Information:
    The dataset provided is a collection of real-world industrial vibration data collected from a brownfield CNC milling machine.
    The acceleration has been measured using a tri-axial accelerometer (Bosch CISS Sensor) mounted inside the machine.
    The X- Y- and Z-axes of the accelerometer have been recorded using a sampling rate equal to 2 kHz.
    Thereby normal as well as anomoulous data have been collected for 6 different timeframes,
    each lasting 6 months from October 2018 until August 2021 and labelled accordingly.
    It can be used to investigate the scalability of models and research process variations as the anomaly impact differs.
    In total there is data from three different CNC milling machines each executing 15 processes.
    For a detailed description of the data and experimental set-up, please refer to the paper.

    Features:
    The following sensors are used:
    - X,Y and Z

    Target Variables:
    The dataset has 1 target variable:
    - machine: M01, M02, M03
    - operation: OP00, OP01, ... , OP14
    - label: good, bad

    Notes
    -----
    -

    Attributes
    ----------
    dataset_url : str
        The URL to the dataset.

    Methods
    -------
    load_dataset(dir_path: Path)
        -> tuple[dict[str, pd.DataFrame], pd.DataFrame]
        Load the Bosch CNC Machining dataset.
    """

    dataset_url = "https://github.com/boschresearch/CNC_Machining.git"
    feature_columns = ["x", "y", "z"]

    def load_dataset(self, dir_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the Bosch CNC Machining dataset.

        Parameters
        ----------
        dir_path : Path
            Path to the directory where the dataset is stored.
        snr_level : int
            The Signal-to-Noise ratio level of the dataset.
        machine_type : str
            The type of machine to load.
        machine_id : str
            The ID of the machine to load.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the features and target data.

        Raises
        ------
        ValueError
            If snr_level, machine_type, or machine_id are invalid.
        """

        # Check if the directory already exists, if not create it
        os.makedirs(dir_path, exist_ok=True)

        # Check if the directory is empty --> download necessary
        if not os.listdir(dir_path):
            self._download_dataset(dir_path)
        else:
            self._update_dataset(dir_path)

        # Load the dataset
        features, targets = self._load_from_dir(dir_path)

        return features, targets

    def _download_dataset(self, dir_path: Path) -> None:
        """
        Download the Bosch CNC MAchining dataset from Github.

        Parameters
        ----------
        dir_path : Path
            The directory to save the dataset to.
        """

        print(f"Downloading Bosch CNC Machining dataset to {dir_path}...")

        Repo.clone_from(
            self.dataset_url,
            dir_path,
            progress=CloneProgress(total=1972, desc="git clone"),
        )

    def _update_dataset(self, dir_path: Path) -> None:
        """
        Download the Bosch CNC MAchining dataset from Github.

        Parameters
        ----------
        dir_path : Path
            The directory to save the dataset to.
        """

        print(f"Updating Bosch CNC Machining dataset to {dir_path}...")

        repo = Repo(dir_path)
        repo.remotes.origin.pull()

    def _load_from_dir(self, dir_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the Bosch CNC Machining dataset from a directory.

        Parameters
        ----------
        dir_path : Path
            Directory containing the Bosch CNC Machining dataset.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the features and target data.
        """
        print(f"Loading Bosch CNC Machining dataset from {dir_path}...")

        # Enumerate all recordings and parse machine/operation/label/year.
        file_df = enumerate_files(dir_path)

        # Initialize dictionaries to store features and targets
        features = {feature: [] for feature in self.feature_columns}
        targets = []

        # Iterate over each file and extract features and targets
        for i, row in tqdm(file_df.iterrows(), total=len(file_df)):
            with h5py.File(row.path, "r") as row_file:
                vibration_data = row_file["vibration_data"]
                targets.append(pd.DataFrame(row[["machine", "operation", "label", "year"]]).T)

                for f_i, feature in enumerate(self.feature_columns):
                    feature_values = pd.Series(vibration_data[:, f_i])
                    feature_df = pd.DataFrame(
                        {
                            "measurement_idx": i,
                            "value": feature_values,
                        }
                    ).astype(
                        {
                            "measurement_idx": "uint16",
                            "value": "float64",
                        }
                    )
                    features[feature].append(feature_df)

        # Concatenate all feature DataFrames
        for feature in self.feature_columns:
            features[feature] = pd.concat(features[feature], axis=0)

        # Concatenate all target DataFrames
        targets = pd.concat(targets, axis=0)

        return features, targets
