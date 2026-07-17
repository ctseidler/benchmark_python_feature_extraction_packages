"""
Module implementing a FeatureExtractor class for extracting features from time series data using
the tsfeatures library.

"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tsfeatures import tsfeatures

class FeatureExtractor:
    """
    Class for extracting features from time series data using the tsfeatures library.

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
    dict_freqs : dict[str, int]
        Dictionary containing the frequencies for the tsfeatures library.

    Methods
    -------
    extract_features(export_dir=None, **kwargs) -> pd.DataFrame
        Extract features from the time-series data using the tsfeatures library.
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
        self.dict_freqs = {}
        self._prepare_features_for_extraction()
        self._rename_measurement_idx()

    def _prepare_features_for_extraction(self) -> None:
        """Prepare the features for feature extraction by adding a time column."""
        if isinstance(self.sampling_frequency, int):
            for sensor, data in self.features.items():
                self.features[sensor] = self._create_time_column(data, self.sampling_frequency)
                # Rename value column to y
                self.features[sensor].rename(columns={"value": "y"}, inplace=True)
        else:
            for sensor, data in self.features.items():
                self.features[sensor] = self._create_time_column(data, self.sampling_frequency[sensor])
                # Rename value column to y
                self.features[sensor].rename(columns={"value": "y"}, inplace=True)

    def _create_time_column(self, data: pd.DataFrame, sampling_frequency: int) -> pd.DataFrame:
        """Create a time column for the given data."""
        # Create a time column. The time is calculated as a counter that starts at 0 and
        # increases by the sampling frequency.
        start_time = pd.Timestamp("2025-01-01 00:00:00")

        # Convert sampling frequency in Hz to milliseconds delta
        sampling_frequency = 1 / sampling_frequency * 1000

        # Create ds column for tsfeatures
        frequency = pd.to_timedelta(sampling_frequency, unit="ms")
        data["ds"] = pd.date_range(start=start_time, periods=len(data), freq=frequency)

        # Create the dict_freqs for tsfeatures
        if f"{int(sampling_frequency)}ms" not in self.dict_freqs:
            self.dict_freqs[f"{int(sampling_frequency)}ms"] = int(sampling_frequency)

        return data

    def _rename_measurement_idx(self) -> None:
        """Rename the measurement_idx column to unique_id."""
        for data in self.features.values():
            data.rename(columns={"measurement_idx": "unique_id"}, inplace=True)

    def extract_features(self, export_dir: str | Path | None = None, **kwargs) -> pd.DataFrame:
        """
        Extract features from the time-series data using the tsfeatures library.

        Parameters
        ----------
        export_dir : str | Path | None, optional
            Directory to export the extracted features to as a CSV file. If not provided, the
            features are not exported. The default is None.
        **kwargs
            Additional keyword arguments to pass to the tsfeatures.tsfeatures function.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted features.
        """
        print("Extracting features using tsfeatures...")

        result = pd.DataFrame()
        for sensor, data in tqdm(self.features.items()):
            try:
                # Extract features using tsfeatures
                extracted_features = tsfeatures(data, dict_freqs=self.dict_freqs, **kwargs)
                # extracted_features = tsfeatures(data, dict_freqs={"100us": 10_000}, **kwargs) # Turning Dataset
                # extracted_features = tsfeatures(data, dict_freqs={"500us": 2_000}, **kwargs) # Bosch CNC
                # extracted_features = tsfeatures(data, dict_freqs={"20833ns": 48_000}, **kwargs)  # IDMT-ISA
                # Rename the columns to include the sensor name
                extracted_features = extracted_features.add_prefix(sensor + "_")

                if export_dir is not None:
                    filename = f"tsfeatures_extracted_features_{sensor}.csv"
                    export_path = Path(export_dir) / filename
                    print(f"\nExporting extracted features to {export_path}...")
                    extracted_features.to_csv(export_path, decimal=",", sep=";", index=True)

                result = pd.concat([result, extracted_features], axis=1)
            except Exception as e:  # This exception is triggered if sampling frequency cannot be inferred
                print(f"Error extracting features for sensor {sensor}: {e}")
                continue

        return result
