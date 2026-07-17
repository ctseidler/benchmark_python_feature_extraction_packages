"""
Module implementing a Dataloader for the Turning Dataset for Chatter Diagnosis.

The dataset can be found under:     
https://data.mendeley.com/public-files/datasets/hvm4wh3jzx/files/73bc3318-3e8e-4566-b565-e726a3d75188/file_downloaded

"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
import scipy.io
from tqdm import tqdm

class Dataloader:
    """
    Data loader for the Turning Dataset for Chatter Diagnosis.

    Dataset URL:
    https://data.mendeley.com/public-files/datasets/hvm4wh3jzx/files/73bc3318-3e8e-4566-b565-e726a3d75188/file_downloaded

    Dataset description:
    --------------------
    This dataset consists of .mat files with acceleration signals related to turning chatter
    detection.

    Methods
    -------
    load_dataset(undersampling: bool = True)
        Load the Turning Chatter Dataset
    """

    url = (
        "https://data.mendeley.com/public-files/datasets/hvm4wh3jzx/files/"
        + "73bc3318-3e8e-4566-b565-e726a3d75188/file_downloaded"
    )
    target_file = "turning_dataset.zip"
    feature_columns = ["label", "time_window", "acc_x_window"]

    def load_dataset(self, perform_undersampling: bool = True) -> tuple:
        """
        Load the Turning Chatter dataset. Dataset will be downloaded it it is not already downloaded

        Parameters:
        - undersampling (bool): Whether to apply undersampling.

        Returns:
        - tuple: Reshaped data for tsfresh and corresponding labels.
        """

        dir_path = Path("data/turning_dataset")

        # Check if the directory already exists, if not, create it
        dir_path = Path(dir_path)
        if not dir_path.exists():
            os.makedirs(dir_path)

        # Check if the directory is empty, if so download the dataset
        if not os.listdir(dir_path):
            self._download_dataset(dir_path)

        # Load the dataset
        df_dataset = self._window_data(dir_path)

        # Perform Undersampling if selected
        if perform_undersampling:
            df_dataset = self._undersampling_dataset(df_dataset)

        features, targets = self._format_dataset(df_dataset)

        return features, targets

    def _check_memory_space(
        self, required_space_gb: float, path: Path = Path(".")
    ) -> bool:
        """
        Checks if the system has enough free disk space.

        Parameters:
        - required_space_gb (float): Required space in gigabytes.
        - path (Path): Path to check the disk space (default: current directory).

        Returns:
        - bool: True if enough space is available, False otherwise.
        """
        _, _, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)  # Convert bytes to GB
        return free_gb >= required_space_gb

    def _download_dataset(self, dir_path: Path, required_space_gb: float = 0.3) -> None:
        """
        Downloads and extracts the dataset if necessary.

        Parameters:
        - dir_path (Path): Directory to save the dataset.
        - required_space_gb (float): Minimum required space in GB.

        Returns:
        - Path: Path where the dataset is saved.
        """

        # First, check if there is enough memory available before downloading
        if not self._check_memory_space(required_space_gb, dir_path):
            raise OSError(
                f"[ERROR] Not enough disk space. At least {required_space_gb} GB is required."
            )

        # Download the Turning Chatter dataset
        response = requests.get(self.url, timeout=10)
        total_size = int(response.headers.get("content-length", 0))
        zip_file = dir_path / "turning_chatter.zip"

        with open(zip_file, "wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading Turning Chatter Dataset",
        ) as progress_bar:
            for chunk in response.iter_content(
                chunk_size=1024
            ):  # Download in 1KB chunks
                file.write(chunk)
                progress_bar.update(len(chunk))

        # Unzip the dowonladed dataset
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        # Remove the zip file
        zip_file.unlink()

    def _window_data(
        self, data_dir: Path, window_length: float = 1.0, sampling_rate: float = 10e3
    ) -> pd.DataFrame:
        """
        Loads .mat files and windows the signal into segments with equal length and annotate
        (label s = stable; label ns = not stable (regardless if unknown, intermediate chatter
        or chatter) each window.

        Parameters:
        - data_dir (Path): Directory containing .mat files.
        - window_length (float): Length of each window in seconds.
        - sampling_rate (float): Sampling rate of the signal.

        Returns:
        - list: List of tuples containing (label, windowed signal data).
        """
        windows = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".mat"):
                    file_path = Path(root) / file

                    # Load .mat file
                    mat_data = scipy.io.loadmat(file_path)
                    raw_data = mat_data["tsDS"]
                    time = raw_data[:, 0]
                    acc_x = raw_data[:, 1]

                    # Determine class based on filename
                    class_name = file[0].lower()
                    if class_name in ["u", "i", "c"]:
                        class_name = "ns"

                    samples_per_window = int(window_length * sampling_rate)
                    num_windows = len(time) // samples_per_window

                    for i in range(num_windows):
                        start_idx = i * samples_per_window
                        end_idx = start_idx + samples_per_window

                        time_window = time[start_idx:end_idx]
                        acc_x_window = acc_x[start_idx:end_idx]
                        windows.append((class_name, time_window, acc_x_window))

        df_windowed_data = pd.DataFrame(windows, columns=self.feature_columns)

        return df_windowed_data

    def _undersampling_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs undersampling on the dataset.

        Parameters:
        - df (pd.DataFrame): The original dataset.

        Returns:
        - pd.DataFrame: Undersampled dataset.
        """
        class_counts = df["label"].value_counts()
        min_class_count = class_counts.min()

        undersampled_data = []
        for class_label in class_counts.index:
            class_data = df[df["label"] == class_label]
            sampled_data = class_data.sample(n=min_class_count, random_state=42)
            undersampled_data.append(sampled_data)

        undersampled_df = pd.concat(undersampled_data)
        undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        return undersampled_df

    def _format_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Reshapes the dataset for tsfresh input as a dictionary with feature data and labels.

        Returns:
        - tuple: A dictionary with feature data and a DataFrame with corresponding labels.
            features['acc-x']: pd.DataFrame with columns ['id', 'time', 'value'].
            labels: pd.DataFrame with columns ['id', 'label'].
        """
        acc_x_data = []
        labels_data = []

        for idx, row in df.iterrows():
            time_window = row["time_window"]
            acc_x_window = row["acc_x_window"]

            # Create DataFrame for 'acc-x'
            acc_x_data.append(
                pd.DataFrame(
                    {
                        "measurement_idx": [idx] * len(time_window),
                        "value": acc_x_window,
                    }
                )
            )

            # Create DataFrame for labels
            labels_data.append(
                pd.DataFrame({"measurement_idx": [idx], "label": [row["label"]]})
            )

        # Combine all the 'acc-x' feature data into one DataFrame
        features = {"acc-x": pd.concat(acc_x_data, ignore_index=True)}

        # Combine all labels into one DataFrame
        targets = pd.concat(labels_data, ignore_index=True)

        return features, targets
