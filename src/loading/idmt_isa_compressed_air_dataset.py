"""
Module implementing a Dataloader for the IDMT-ISA-Compressed-Air Dataset.

The dataset can be found under:
https://zenodo.org/api/records/7551606/files-archive

Dataset description:
--------------------
General Information:
The dataset consists of 5,592 audio recordings, each lasting 30 seconds,
captured at a 48 kHz sampling rate with 32-bit resolution.
Recordings simulate compressed air leaks in an industrial setting,
combining multiple leak types with various noise conditions.
The data were collected using four omnidirectional Earthworks M30 microphones
in different configurations.

Each combination of leak and noise type was recorded in 3 separate sessions.
Each session consists of 128 files, recorded with four microphones.
Leak types:
- Vent leak
- Vent leak low Pressure
- Tube leak

Noise types:
- Lab noise (no added background noise)
- Hydraulic machine noise (high volume)
- Hydraulic machine noise (low volume)
- General factory workshop noise (high volume)
- General factory workshop noise (low volume)

Features:
The following sensors are used:
- mic_1, mic_2, mic_3, mic_4: 4 different microphones

Target Variables:
The dataset has 1 target variable:
- Leakage type (4 classes): No leak, Vent leak, Vent leak low Pressure, Tube leak

Notes
-----
- As the entire dataset is very large (~ 22 GB), we suggest using only
one session type.

"""

from __future__ import annotations

import glob
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class Dataloader:
    """
    Data loader for the IDMT-ISA-Compressed-Air Dataset.

    Dataset URL:
    https://zenodo.org/api/records/7551606/files-archive

    Description:
    ------------
    The IICA dataset contains acoustic recordings of compressed air leaks
    under various leak types and noise conditions. It is designed to support
    research on industrial air leak detection using audio signals.

    Attributes:
    -----------
    - url (str): URL to download the dataset.
    - target_file (str): Name of the downloaded dataset archive.
    - feature_columns (list): Column names for extracted features.

    Methods:
    --------
    - load_dataset(dir_path, session_filter="1", mic_filter=1):
        Loads the dataset, downloading and extracting it.
    - _check_memory(required_space_gb, path):
        Checks if sufficient disk space is available.
    - _download_dataset(dir_path, required_space_gb=42):
        Downloads and extracts the dataset archive.
    - _load_from_dir(dir_path, session_filter, mic_filter):
        Loads and processes audio files into features and labels.
    """

    url = "https://zenodo.org/api/records/7551606/files-archive"
    target_file = "compressed_air_dataset.zip"
    feature_columns = ["label", "time_window", "signal_window"]

    def load_dataset(
        self, dir_path: Path, session_filter: int = 1, mic_filter: list | None = None
    ) -> tuple:
        """
        Loads the dataset, downloads and extracts it if not already present.

        Parameters:
        -----------
        - dir_path (Path): Path to the dataset directory.
        - session_filter (int): Filters sessions based on the given value.
        - mic_filter (list): Filters data for the specified microphone.

        Returns:
        --------
        - tuple: Dictionary of features and DataFrame of labels.
        """
        if mic_filter is None:
            mic_filter = [1, 2, 3, 4]

        session_filter = str(session_filter)

        # Create dataset directory if it doesn't exist
        if not dir_path.exists():
            os.makedirs(dir_path)

        # Download dataset if directory is empty
        if not os.listdir(dir_path):
            self._download_dataset(dir_path)

        # Load and process the dataset
        features, targets = self._load_from_dir(dir_path, session_filter, mic_filter)

        return features, targets

    def _check_memory(self, required_space_gb: float, path: Path = Path(".")) -> bool:
        """
        Checks if sufficient free disk space is available.

        Parameters:
        -----------
        - required_space_gb (float): Minimum required space in gigabytes.
        - path (Path): Path to check available disk space.

        Returns:
        --------
        - bool: True if sufficient space is available, False otherwise.
        """
        _, _, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        return free_gb >= required_space_gb

    def _download_dataset(self, dir_path: Path, required_space_gb: float = 23) -> None:
        """
        Downloads and extracts the dataset.

        Parameters:
        -----------
        - dir_path (Path): Directory to save the dataset.
        - required_space_gb (float): Minimum required disk space.
        """

        if not self._check_memory(required_space_gb, dir_path):
            msg = f"[ERROR] Not enough disk space. At least {required_space_gb} GB is required."
            raise MemoryError(msg)

        response = requests.get(self.url, stream=True, timeout=10)
        zip_file = dir_path / "compressed_air_dataset.zip"
        dataset_size_bytes = 22.1 * 1024**3
        total_downloaded = 0

        # Download the IICA dataset
        with open(zip_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total_downloaded += len(chunk)
                    percentage = (total_downloaded / dataset_size_bytes) * 100
                    print(
                        "[INFO] Downloaded: "
                        f"{total_downloaded / (1024**3):.2f} GB of "
                        f"{dataset_size_bytes / (1024**3):.1f} GB "
                        f"({percentage:.1f}%)",
                        end="\r",
                    )

        # Extract the downloaded file
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        # Remove zip file after extraction
        zip_file.unlink()

        # Extract the dataset
        for file in os.listdir(dir_path):
            file_path = dir_path / file

            if file.endswith(".zip"):
                # Extract the zip file
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(dir_path)

                # Delete the zip file after extraction
                file_path.unlink()

    def _load_from_dir(
        self,
        dir_path: Path,
        session_filter: str,
        mic_list: list,
        win_size: int = 2048,
        overlap: float = 0.0,
        window_audio: bool = False,
    ) -> tuple:
        """
        Loads and processes audio files into features and labels.

        Parameters:
        -----------
        - dir_path (Path): Path to the dataset directory.
        - session_filter (str): Filters sessions (1, 2, or 3).
        - mic_list (list): List of microphones (integers) to filter data.
        - win_size (int): Size of the window for splitting the audio.
        - overlap (float): Fraction of overlap between windows.
        - window_audio (bool): If False, use the original full-size
        audio instead of windowing.

        Returns:
        --------
        - tuple: Features dictionary (by microphone) and labels DataFrame.
        """

        sampling_rate = 48e3
        label_mapping = {
            "iO": 0,
            "tubeleak": 1,
            "ventleak": 2,
            "ventlow": 3,
            "unknown": 4,
        }

        features = {}
        for mic_filter in tqdm(mic_list, desc="Total progress", unit="mic_filter", leave=True):
            features_data = []
            labels_data_list = []
            mic_id = f"mic_{mic_filter}"

            # Iterate over the main folders
            for main_folder in tqdm(
                ["tubeleak", "ventleak", "ventlow"],
                desc=f"Progressing data of: {mic_id}",
                unit="main_folder",
                leave=False,
            ):
                main_folder_path = dir_path / main_folder

                # Iterate over the subfolders
                for subfolder in tqdm(
                    ["hydr", "hydr_low", "lab", "work", "work_low"],
                    desc=f"Processing data for leakage type: {main_folder}",
                    unit="subfolder",
                    leave=False,
                ):
                    subfolder_path = main_folder_path / subfolder
                    folder_path = subfolder_path / session_filter

                    if folder_path.exists():
                        wav_files = glob.glob(str(folder_path / "*.wav"))

                        for wav_file in tqdm(
                            wav_files,
                            desc=f"Processing data for noise type: {subfolder}",
                            unit="file",
                            leave=False,
                        ):
                            filename = os.path.basename(wav_file)
                            parts = filename.split("_")

                            # Extract mic number
                            mic_match = [
                                part for part in parts if part.endswith("l") or part.endswith("m")
                            ]
                            mic_number = int(mic_match[0][0]) if mic_match else -1

                            # If the mic_number is in mic_list, process the
                            # file, but only process iO files of tubeleak in
                            # order to get balanced dataset
                            if mic_number == mic_filter:
                                if len(parts) > 1:
                                    if main_folder == "tubeleak" and parts[1] in [
                                        "iO",
                                        "niO",
                                    ]:
                                        label = parts[1]
                                    elif main_folder != "tubeleak" and parts[1] == "niO":
                                        label = "niO"
                                    else:
                                        continue

                                # Map the label to an integer
                                label_encoded = label_mapping[
                                    (
                                        label
                                        if label == "iO"
                                        else (main_folder if label == "niO" else "unknown")
                                    )
                                ]

                                # Load audio data (using original sample rate)
                                # librosa is an optional audio dependency
                                # (requirements/audio_loaders.txt); import it
                                # lazily so this module loads without it.
                                import librosa

                                audio, _ = librosa.load(wav_file, sr=None)

                                if window_audio:
                                    total_samples = len(audio)
                                    step_size = int(win_size * (1 - overlap))

                                    # Ensure only full windows are included
                                    num_windows = (total_samples - win_size) // step_size + 1

                                    time_vector = np.arange(len(audio)) / sampling_rate

                                    # Window the audio
                                    for i in range(num_windows):
                                        start_idx = i * step_size
                                        end_idx = start_idx + win_size
                                        windowed_audio = audio[start_idx:end_idx]

                                        # Create DataFrame for this window of audio
                                        audio_df = pd.DataFrame(
                                            {
                                                "measurement_idx": [len(features_data)]
                                                * len(windowed_audio),
                                                "value": windowed_audio,
                                            }
                                        )

                                        features_data.append(audio_df)

                                        labels_data_list.append({"label": label_encoded})
                                else:
                                    # Use the original full audio signal
                                    # without windowing
                                    total_samples = len(audio)
                                    time_vector = np.arange(len(audio)) / sampling_rate

                                    # Create DataFrame for the entire audio
                                    audio_df = pd.DataFrame(
                                        {
                                            "measurement_idx": [len(features_data)] * total_samples,
                                            "value": audio,
                                        }
                                    )

                                    features_data.append(audio_df)

                                    labels_data_list.append({"label": label_encoded})

            features[mic_id] = pd.concat(features_data, ignore_index=True)

        targets = pd.DataFrame(labels_data_list)

        return features, targets
