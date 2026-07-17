"""
Module implementing a FeatureExtractor class for extracting features from time series data using 
the tsfresh library.

"""

from pathlib import Path

import pandas as pd
from tsfresh import extract_features

class FeatureExtractor:
    """
    Class for extracting features from time series data using the tsfresh library.

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        Dictionary containing the time-series data to extract features from. The keys are the names
        of the features and the values are the corresponding DataFrames with the time series data.
        Each DataFrame must have a column named "measurement_idx" containing the index of the
        measurement and a column named "value" containing the value of the measurement.

    Attributes
    ----------
    features : dict[str, pd.DataFrame]
        Dictionary containing the time-series data to extract features from. The keys are the names
        of the features and the values are the corresponding DataFrames with the time series data.
        Each DataFrame must have a column named "measurement_idx" containing the index of the
        measurement and a column named "value" containing the value of the measurement.

    Methods
    -------
    extract_features(export_dir=None, **kwargs) -> pd.DataFrame
        Extract features from the time-series data using the tsfresh library.
    """

    def __init__(self, features: dict[str, pd.DataFrame]) -> None:
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
        """
        Prepare the data for feature extraction by adding a time column to each feature
        DataFrame.
        """
        for feature_name, feature_data in self.features.items():
            feature_data["time"] = feature_data.index
            self.features[feature_name] = feature_data

    def extract_features(
        self, export_dir: str | Path | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from the time-series data using the tsfresh library.

        Parameters
        ----------
        export_dir : str | Path | None, optional
            Directory to export the extracted features to. If None, the features are not exported.
            The default is None.
        **kwargs
            Additional keyword arguments to be passed to the tsfresh.extract_features function.

        Returns
        -------
        pd.DataFrame
            The extracted features.
        """
        print("Extracting features using tsfresh...")
        extracted_features = extract_features(
            self.features,
            column_id="measurement_idx",
            column_sort="time",
            column_value="value",
            **kwargs,
        )

        if export_dir is not None:
            filename = "tsfresh_extracted_features.csv"
            export_path = Path(export_dir) / filename
            print(f"Exporting extracted features to {export_path}...")
            extracted_features.to_csv(export_path, decimal=",", sep=";", index=True)

        return extracted_features
