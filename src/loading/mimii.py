"""
Module implementing a Dataloader for the Sound Dataset for Malfunctioning Industrial Machine
Investigation and Inspection (MIMII Dataset).

The dataset can be found under: 
https://zenodo.org/records/3384388

Dataset description:
--------------------
General Information:
The dataset consists of ~ 32.000 samples of 10 seconds each. The authors sampled the data from
4 different machine types: Valve, Pump, Fan, Slide rail. Each machine type has data from 7 different
machines. For each machine, data from normal and anomalous conditions were collected. A sampling
rate of 16 kHz was used. Each recording contains 8 channels (= 8 different microphones).
Additionally, the authors added noise with different SNR levels to the recordings. Three different
noise levels are available: -6 dB, 0 dB, 6 dB.

Features:
The following sensors are used:
- M1, M2, M3, M4, M5, M6, M7, M8: 8 different microphones

Target Variables:
The dataset has 1 target variable:
- Machine condition: normal, anomaly

Notes
-----
- As the entire dataset is very large (~ 40 GB per SNR-level), we suggest using only 
one machine type.
- The original authors have only released data from 4 machines per machine type.

"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from scipy.io import wavfile
from tqdm import tqdm

class Dataloader:
    """
    Dataloader for the MIMII dataset.

    The dataset can be found under:
    https://zenodo.org/records/3384388

    Dataset description:
    --------------------
    General Information:
    The dataset consists of ~ 32.000 samples of 10 seconds each. The authors sampled the data from
    4 different machine types: Valve, Pump, Fan, Slide rail. Each machine type has data from 7
    different machines. For each machine, data from normal and anomalous conditions were collected.
    A sampling rate of 16 kHz was used. Each recording contains 8 channels
    (= 8 different microphones). Additionally, the authors added noise with different SNR levels
    to the recordings. Three different noise levels are available: -6 dB, 0 dB, 6 dB.

    Features:
    The following sensors are used:
    - M1, M2, M3, M4, M5, M6, M7, M8: 8 different microphones

    Target Variables:
    The dataset has 1 target variable:
    - Machine condition: normal, anomaly

    Notes
    -----
    - As the entire dataset is very large (~ 40 GB per SNR-level), we suggest using only
    one machine type.
    - The original authors have only released data from 4 machines per machine type.

    Attributes
    ----------
    dataset_url : str
        The URL to the dataset.
    snr_levels : list[int]
        A list of available SNR levels.
    machine_types : list[str]
        A list of available machine types.
    machine_ids : list[str]
        A list of available machine IDs.

    Methods
    -------
    load_dataset(dir_path: Path, snr_level: int, machine_type: str, machine_id: str)
        -> tuple[dict[str, pd.DataFrame], pd.DataFrame]
        Load the MIMII dataset.
    """

    dataset_url = "https://zenodo.org/records/3384388"
    snr_levels = [-6, 0, 6]
    machine_types = ["valve", "pump", "fan", "slider"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]

    def load_dataset(
        self, dir_path: Path, snr_level: int, machine_type: str, machine_id: str
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the MIMII dataset.

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

        # Validate the inputs
        if snr_level not in self.snr_levels:
            raise ValueError(f"Invalid SNR level: {snr_level}.")
        if machine_type not in self.machine_types:
            raise ValueError(f"Invalid machine type: {machine_type}.")
        if machine_id not in self.machine_ids:
            raise ValueError(f"Invalid machine id: {machine_id}.")

        # Check if the directory already exists, if not create it
        dir_path.mkdir(parents=True, exist_ok=True)

        # Check if the directory is empty --> download necessary
        if not os.listdir(dir_path):
            self._download_dataset(
                dir_path, snr_level=snr_level, machine_type=machine_type
            )

        # Load the dataset
        subset_dir = dir_path / f"{machine_type}" / f"{machine_id}"
        features, targets = self._load_from_dir(subset_dir)

        return features, targets

    def _download_dataset(
        self, dir_path: Path, snr_level: int, machine_type: str
    ) -> None:
        """
        Download the MIMII dataset from Zenodo.

        Parameters
        ----------
        dir_path : Path
            The directory to save the dataset to.
        snr_level : int
            The SNR level of the dataset.
        machine_type : str
            The machine type to download.
        """
        print(f"Downloading the MIMII dataset to {dir_path}...")

        # Download the dataset
        url = f"{self.dataset_url}/files/{snr_level}_dB_{machine_type}.zip"
        response = requests.get(url, timeout=10)
        zip_file = dir_path / "data.zip"

        with open(zip_file, "wb") as file:
            file.write(response.content)

        # Unzip the dataset
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dir_path)

    def _load_from_dir(
        self, dir_path: Path
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the MIMII dataset from a directory.

        Parameters
        ----------
        dir_path : Path
            Directory containing the MIMII dataset.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the features and target data.
        """
        print(f"Loading MIMII dataset from {dir_path}...")

        # Iterate over the directories and load the features
        dirs = [d for d in dir_path.iterdir() if d.is_dir()]
        targets = []
        features = pd.DataFrame()
        measurement_index = 0
        for data_dir in dirs:
            if "abnormal" in str(data_dir):
                target_value = 1
                files = [f for f in data_dir.iterdir() if f.is_file()]
                for file in tqdm(files, desc="Loading abnormal data"):
                    _, data = wavfile.read(file)
                    df = pd.DataFrame(data)

                    # Add measurement index
                    df["measurement_idx"] = str(measurement_index)
                    measurement_index += 1

                    features = pd.concat([features, df], axis=0)
                    targets.append(target_value)
            elif "normal" in str(data_dir):
                target_value = 0
                files = [f for f in data_dir.iterdir() if f.is_file()]
                for file in tqdm(files, desc="Loading normal data"):
                    _, data = wavfile.read(file)
                    df = pd.DataFrame(data)

                    # Add measurement index
                    df["measurement_idx"] = str(measurement_index)
                    measurement_index += 1

                    features = pd.concat([features, df], axis=0)
                    targets.append(target_value)

        # Convert targets to DataFrame
        targets = pd.DataFrame(targets, columns=["machine_condition"])

        features = self._extract_individual_sensors(features)

        return features, targets

    def _extract_individual_sensors(
        self, features: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Extract the individual microphone data from the feature DataFrame."""
        sensor_data = {}
        for sensor in range(8):
            sensor_data[f"M{sensor}"] = features[[sensor]]
            # Rename the column to `value`
            sensor_data[f"M{sensor}"].columns = ["value"]

        return sensor_data
