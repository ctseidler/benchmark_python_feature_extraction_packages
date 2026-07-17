"""
Module implementing a FeatureExtractor class for extracting features from time series data using 
the pycatch22 library.

"""

from pathlib import Path

import pandas as pd
import pycatch22
from tqdm import tqdm

class FeatureExtractor:
    """
    Class for extracting features from time series data using the pycatch22 library.

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

    Attributes
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

    Methods
    -------
    extract_features(export_dir=None, **kwargs) -> pd.DataFrame
        Extract features from the time-series data using the pycatch22 library.
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
        self._prepare_features_for_extraction()

    def _prepare_features_for_extraction(self) -> None:
        """Prepare the features for feature extraction by sanitizing the data."""
        if isinstance(self.sampling_frequency, int):
            data = pd.DataFrame()
            for sensor, sensor_data in self.features.items():
                sensor_data = self._sanitize_features(sensor_data, sensor)
                data = pd.concat([data, sensor_data], axis="columns")
            # Drop duplicated measurement_idx columns
            data = data.loc[:, ~data.columns.duplicated()].copy()
            self.features = data
        else:
            self.window_size = {}
            preprocessed_features = {}
            for sensor, sensor_data in self.features.items():
                sensor_data = self._sanitize_features(sensor_data, sensor)
                preprocessed_features[sensor] = sensor_data
            self.features = preprocessed_features

    def _sanitize_features(self, features: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """
        Sanitize the features by renaming the value column and changing the dtype of the
        measurement_idx column to int.
        """
        features["measurement_idx"] = features["measurement_idx"].astype(int)
        features = features.rename(columns={"value": sensor})

        return features

    def extract_features(
        self, export_dir: str | Path | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the pycatch22 library.

        Parameters
        ----------
        export_dir : str | Path | None, optional
            Directory to export the extracted features to. If None, the features are not exported.
            The default is None.
        **kwargs
            Additional keyword arguments to pass to the pycatch22.catch22_all function.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted features.
        """
        print("Extracting features using pycatch22...")

        if isinstance(self.sampling_frequency, int):
            return self._extract_single_frequency_features(export_dir, **kwargs)

        return self._extract_different_frequencies_features(export_dir, **kwargs)

    def _extract_single_frequency_features(
        self, export_dir: str | Path | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the pycatch22 library when only one global
        sampling frequency exists.

        Parameters
        ----------
        export_dir : str | Path | None, optional
            Directory to export the extracted features to. If None, the features are not exported.
        **kwargs
            Additional keyword arguments to pass to the pycatch22.catch22_all function.

        Returns
        -------
        pd.DataFrame
            The extracted features.
        """
        extracted_features = pd.DataFrame()

        # Iterate over all sensors
        for column in tqdm(self.features.columns):
            if column == "measurement_idx":
                continue

            sensor_features = self._extract_features_for_sensor_data(
                self.features, column, export_dir=export_dir, **kwargs
            )

            extracted_features = pd.concat(
                [extracted_features, sensor_features], axis=1
            )
        extracted_features = extracted_features.reset_index(drop=True)

        return extracted_features

    def _extract_features_for_sensor_data(
        self,
        sensor_data: pd.DataFrame,
        sensor: str,
        export_dir: str | Path | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data of a single sensor using the pycatch22 library.

        Parameters
        ----------
        sensor_data : pd.DataFrame
            The time-series data of the sensor.
        sensor : str
            The name of the sensor.
        export_dir : str | Path | None, optional
            Directory to export the extracted features to as a CSV file. If not provided, the
            features are not exported. The default is None.
        **kwargs
            Additional keyword arguments to pass to the pycatch22.catch22_all function.

        Returns
        -------
        pd.DataFrame
            The extracted features.
        """
        extracted_features = pd.DataFrame()

        # Split dataframe based on the value of the measurement_idx column
        dataframes = [
            x
            for _, x in sensor_data.loc[:, [sensor, "measurement_idx"]].groupby(
                "measurement_idx"
            )
        ]

        # Extract features for every dataframe (= 1 time-series)
        for dataframe in dataframes:
            result = pycatch22.catch22_all(dataframe[sensor].values, **kwargs)

            result_df = pd.DataFrame(
                [result["values"]],
                columns=[f"{sensor}_{name}" for name in result["names"]],
            )
            extracted_features = pd.concat([extracted_features, result_df], axis=0)

        extracted_features = extracted_features.reset_index(drop=True)

        if export_dir is not None:
            filename = f"pycatch22_extracted_features_{sensor}.csv"
            export_path = Path(export_dir) / filename
            print(f"\nExporting extracted features to {export_path}...")
            extracted_features.to_csv(export_path, index=True, sep=";", decimal=",")

        return extracted_features

    def _extract_different_frequencies_features(
        self, export_dir: str | Path | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the pycatch22 library when different
        sampling frequencies exist.

        Parameters
        ----------
        export_dir : str | Path | None, optional
            Directory to export the extracted features to. If None, the features are not exported.
        **kwargs
            Additional keyword arguments to pass to the pycatch22.catch22_all function.

        Returns
        -------
        pd.DataFrame
            The extracted features.
        """
        extracted_features = pd.DataFrame()

        # Iterate over all sensors
        for sensor, sensor_data in tqdm(self.features.items()):
            # Rename value column to sensor name
            sensor_data = sensor_data.rename(columns={"value": sensor})

            sensor_features = self._extract_features_for_sensor_data(
                sensor_data, sensor, export_dir=export_dir, **kwargs
            )

            extracted_features = pd.concat(
                [extracted_features, sensor_features], axis=1
            )
        extracted_features = extracted_features.reset_index(drop=True)

        return extracted_features
