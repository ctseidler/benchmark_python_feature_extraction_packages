"""
Module implementing a FeatureExtractor class for extracting features from time series data using
the TSFEL library.

"""

from pathlib import Path

import pandas as pd
import tsfel
from tqdm import tqdm

class FeatureExtractor:
    """
    Class for extracting features from time series data using the TSFEL library.

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        Dictionary containing the time-series data to extract features from. The keys are the names
        of the features and the values are the corresponding DataFrames with the time series data.
        Each DataFrame must have a column named "measurement_idx" containing the index of the
        measurement and a column named "value" containing the value of the measurement.
    sampling_frequency : int | dict[str, int]
        The sampling frequency of the time-series data. If a single value is provided, it is assumed
        that all features have the same sampling frequency. If a dictionary is provided, the keys
        are the names of the features and the values are the corresponding sampling frequencies.

    Attributes
    ----------
    features : dict[str, pd.DataFrame]
        Dictionary containing the time-series data to extract features from. The keys are the names
        of the features and the values are the corresponding DataFrames with the time series data.
        Each DataFrame must have a column named "measurement_idx" containing the index of the
        measurement and a column named "value" containing the value of the measurement.
    sampling_frequency : int | dict[str, int]
        The sampling frequency of the time-series data. If a single value is provided, it is assumed
        that all features have the same sampling frequency. If a dictionary is provided, the keys
        are the names of the features and the values are the corresponding sampling frequencies.
    window_size : int | dict[str, int]
        The window size for TSFEL feature extraction. If a single value is provided, it is assumed
        that all features have the same window size. If a dictionary is provided, the keys are the
        names of the features and the values are the corresponding window sizes.

    Methods
    -------
    extract_features(cfg=None, export_dir=None, **kwargs) -> pd.DataFrame
        Extract features from the time-series data using the TSFEL library.
    """

    def __init__(
        self,
        features: dict[str, pd.DataFrame],
        sampling_frequency: int | dict[str, int],
    ) -> None:
        """
        Constructor for the FeatureExtractor class.

        Parameters
        ----------
        features : dict[str, pd.DataFrame]
            Dictionary containing the time-series data to extract features from. The keys are the
            names of the features and the values are the corresponding DataFrames with the time
            series data. Each DataFrame must have a column named "measurement_idx" containing the
            index of the measurement and a column named "value" containing the value of the
            measurement.
        sampling_frequency : int | dict[str, int]
            The sampling frequency of the time-series data. If a single value is provided, it is
            assumed that all features have the same sampling frequency. If a dictionary is provided,
            the keys are the names of the features and the values are the corresponding sampling
            frequencies.
        """

        self.features = features
        self.sampling_frequency = sampling_frequency
        self.window_size = None
        self._prepare_features_for_extraction()

    def _prepare_features_for_extraction(self) -> None:
        """
        Prepare the data for feature extraction by calculating the window size for TSFEL and
        sanitizing the features.
        """
        # If the sampling rate is a single value, concat all the dataframes on the index
        if isinstance(self.sampling_frequency, int):
            data = pd.DataFrame()
            for sensor, sensor_data in self.features.items():
                self.window_size = self._calculate_window_size(sensor_data)
                sensor_data = self._sanitize_features(sensor_data, sensor)
                data = pd.concat([data, sensor_data], axis="columns")
            self.features = data
        # Else remove the measurement_idx column and rename the value column to the sensor name
        else:
            self.window_size = {}
            preprocessed_features = {}
            for sensor, sensor_data in self.features.items():
                self.window_size[sensor] = self._calculate_window_size(sensor_data)
                sensor_data = self._sanitize_features(sensor_data, sensor)
                preprocessed_features[sensor] = sensor_data
            self.features = preprocessed_features

    def _calculate_window_size(self, data: pd.DataFrame) -> int:
        """Calculate the window size for TSFEL based on the number of samples in the data."""
        num_samples = data["measurement_idx"].astype(int).max()
        return int(len(data) // num_samples)

    def _sanitize_features(self, features: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """
        Sanitize the features by removing the measurement_idx column and renaming the value column.
        """
        features = features.drop(columns=["measurement_idx"])
        features = features.rename(columns={"value": sensor})

        return features

    def extract_features(self, cfg: dict | None = None, export_dir: str | Path | None = None, **kwargs) -> pd.DataFrame:
        """
        Extract features from the time-series data using the TSFEL library.

        Parameters
        ----------
        cfg : dict | None, optional
            Configuration dictionary for the TSFEL library. If None, the default configuration is
            used (= all possible features are calculated), defaults to None.
        export_dir : str | Path | None, optional
            Directory to export the extracted features to. If None, the features are not exported,
            defaults to None.
        **kwargs
            Additional keyword arguments to pass to the tsfel.time_series_features_extractor
            functon.

        Returns
        -------
        pd.DataFrame
            The extracted features.
        """
        print("Extracting features using TSFEL...")
        if cfg is None:
            cfg = tsfel.get_features_by_domain()

        if isinstance(self.sampling_frequency, int):
            if self.sampling_frequency == 1:
                self.sampling_frequency = None
            extracted_features = tsfel.time_series_features_extractor(
                timeseries=self.features,
                config=cfg,
                fs=self.sampling_frequency,
                window_size=self.window_size,
                **kwargs,
            )

            if export_dir is not None:
                filename = "tsfel_extracted_features.csv"
                export_path = Path(export_dir) / filename
                print(f"Exporting extracted features to {export_path}...")
                extracted_features.to_csv(export_path, index=True, sep=";", decimal=",")

        if isinstance(self.sampling_frequency, dict):
            extracted_features = pd.DataFrame()
            for sensor, sensor_data in tqdm(self.features.items()):
                # Modify sampling rate if it is 1 Hz, as TSFEL cannot handle 1 Hz sampling rate
                if self.sampling_frequency[sensor] == 1:
                    self.sampling_frequency[sensor] = None

                # Get sampling frequency
                fs = self.sampling_frequency[sensor]
                window_size = self.window_size[sensor]
                sensor_extracted_features = tsfel.time_series_features_extractor(
                    timeseries=sensor_data,
                    config=cfg,
                    fs=fs,
                    window_size=window_size,
                    verbose=0,
                    **kwargs,
                )

                if export_dir is not None:
                    filename = f"tsfel_extracted_features_{sensor}.csv"
                    export_path = Path(export_dir) / filename
                    print(f"\nExporting extracted features to {export_path}...")
                    sensor_extracted_features.to_csv(export_path, index=True, sep=";", decimal=",")

                extracted_features = pd.concat(
                    [extracted_features, sensor_extracted_features],
                    axis="columns",
                    ignore_index=True,
                )

        return extracted_features
