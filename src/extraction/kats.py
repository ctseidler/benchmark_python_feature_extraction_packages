"""
Module implementing a FeatureExtractor class for extracting features from time series data using 
the kats library.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
from tqdm import tqdm

class FeatureExtractor:
    """
    Class for extracting features from time series data using the kats library.

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        Dictionary containing the time-series data to extract features from. The keys are the
        names of the features and the values are the corresponding DataFrames with the time
        series data. Each DataFrame must have a column named "measurement_idx" containing the
        index of the measurement and a column named "value" containing the value of the
        measurement.

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
    extract_features(**kwargs)
        Extract features from the time-series data using the kats library.
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
        """Prepare the features for feature extraction.

        We create a list of np.arrays for every sensor. Thus, we can later extract features per
        time-series individually.
        """
        # Iterate over the measurement_idx column -> each id is one sample
        for name, feature in self.features.items():
            array = []
            for idx in feature["measurement_idx"].unique():
                # Get the values for the current measurement_idx
                values = feature[feature["measurement_idx"] == idx]["value"]

                # Convert to TimeSeriesData
                time_col = pd.Series(range(0, len(values)))
                values = TimeSeriesData(time=time_col, value=values)

                array.append(values)

            # Add the array to the features dictionary
            self.features[name] = array

    def extract_features(
        self,
        export_dir: str | Path | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the kats library.

        Parameters
        ----------
        export_dir : str | Path | None
            Directory to export the extracted features to as a CSV file. If None, the features are
            not exported. The default is None.
        **kwargs
            Additional keyword arguments to pass to the kats.TsFeatures.transform method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted features.
        """
        print("Extracting features using kats...")

        result = pd.DataFrame()
        for sensor, data in tqdm(self.features.items()):
            result_per_ts = pd.DataFrame()
            for ts_data in data:
                sampling_frequency = (
                    self.sampling_frequency[sensor]
                    if isinstance(self.sampling_frequency, dict)
                    else self.sampling_frequency
                )
                transformer = TsFeatures(
                    window_size=len(ts_data.value),
                    spectral_freq=sampling_frequency,
                    **kwargs,
                )
                # Extract features using kats
                extracted_features = transformer.transform(ts_data)

                # Convert the dictionary to a DataFrame
                extracted_features = pd.DataFrame(data=extracted_features, index=[0])
                # Concatenate the extracted features
                result_per_ts = pd.concat([result_per_ts, extracted_features], axis=0)

            # Rename the columns to include the sensor name
            result_per_ts = result_per_ts.add_prefix(sensor + "_")

            if export_dir is not None:
                filename = f"kats_extracted_features_{sensor}.csv"
                export_path = Path(export_dir) / filename
                print(f"\nExporting extracted features to {export_path}...")
                result_per_ts.to_csv(export_path, decimal=",", sep=";", index=True)

            result = pd.concat([result, result_per_ts], axis=1)

        # Replace index with range(0, len(result))
        result.index = range(0, len(result))

        return result
