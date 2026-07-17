"""
Module implementing a FeatureExtractor class for extracting features from time series data using 
the seglearn library.

"""

from pathlib import Path

import numpy as np
import pandas as pd
from seglearn.transform import FeatureRep
from tqdm import tqdm

class FeatureExtractor:
    """
    Class for extracting features from time series data using the seglearn library.

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

    Methods
    -------
    extract_features(features_to_extract=None, verbose=False, export_dir=None) -> pd.DataFrame
        Extract features from the time-series data using the seglearn library.
    """

    def __init__(
        self,
        features: dict[str, pd.DataFrame],
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
        """
        self.features = features
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
                values = feature[feature["measurement_idx"] == idx]["value"].values
                array.append(values)

            # Add the array to the features dictionary
            self.features[name] = array

    def extract_features(
        self,
        features_to_extract: str | dict | None = None,
        verbose: bool = False,
        export_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the seglearn library.

        Parameters
        ----------
        features_to_extract : str | dict | None, optional
            Features to extract, default is None, which extracts the default setting.
        verbose : bool, optional
            Whether to print additional information during feature extraction, default is False.
        export_dir : str | Path | None, optional
            Directory to export the extracted features to, default is None. If None, the features
            are not exported.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted features.
        """
        print("Extracting features using seglearn...")

        if features_to_extract is None:
            features_to_extract = "default"

        transformer = FeatureRep(features=features_to_extract, verbose=verbose)

        result = pd.DataFrame()
        for sensor, data in tqdm(self.features.items()):
            result_per_ts = pd.DataFrame()
            for ts_data in data:
                # Add an additional dimension to the data
                ts_data = np.expand_dims(ts_data, axis=0)
                # Extract features using seglearn
                extracted_features = transformer.fit_transform(ts_data, [1])
                # Convert the extracted features to a DataFrame
                extracted_features = pd.DataFrame(
                    data=extracted_features, columns=transformer.f_labels
                )
                # Concatenate the extracted features
                result_per_ts = pd.concat([result_per_ts, extracted_features], axis=0)

            # Rename the columns to include the sensor name
            result_per_ts = result_per_ts.add_prefix(sensor + "_")

            if export_dir is not None:
                filename = f"seglearn_extracted_features_{sensor}.csv"
                export_path = Path(export_dir) / filename
                print(f"\nExporting extracted features to {export_path}...")
                result_per_ts.to_csv(export_path, index=True, sep=";", decimal=",")

            result = pd.concat([result, result_per_ts], axis=1)

        # Replace index with range(0, len(result))
        result.index = range(0, len(result))

        return result
