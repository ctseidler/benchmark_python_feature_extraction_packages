"""
Test suite for the feature extraction using tsfresh.

"""

import os
from pathlib import Path

from tsfresh.feature_extraction import MinimalFCParameters

from src.extraction.tsfresh import FeatureExtractor
from src.loading.cnc_mill_tool_wear import Dataloader

def test_extraction() -> None:
    """Test feature extraction."""
    # Load the CNC Mill Tool Wear dataset
    temp_data_dir = Path(__file__).parent / "temp_data"
    dataloader = Dataloader()
    features, _ = dataloader.load_dataset(temp_data_dir)

    # Extract the features using tsfresh
    try:
        feature_extractor = FeatureExtractor(features)
        extracted_features = feature_extractor.extract_features(
            default_fc_parameters=MinimalFCParameters(), n_jobs=1
        )

        assert len(extracted_features) == 18
        assert (
            extracted_features.shape[1] == 45 * 10
        )  # We have 45 sensors and 10 features per sensor
    finally:
        # Delete feature files
        experiment_dir = temp_data_dir / "features"
        for file in experiment_dir.iterdir():
            file.unlink()
        experiment_dir.rmdir()

        # Delete target file
        os.remove(temp_data_dir / "train.csv")

        temp_data_dir.rmdir()
