"""
Test suite for the feature extraction using kats.

"""

import os
from pathlib import Path

from src.extraction.kats import FeatureExtractor
from src.loading.cnc_mill_tool_wear import Dataloader

def test_feature_extraction() -> None:
    """Test feature extraction."""
    # Load the CNC Mill Tool Wear dataset
    temp_data_dir = Path(__file__).parent / "temp_data"
    dataloader = Dataloader()
    features, _ = dataloader.load_dataset(temp_data_dir)

    # Extract features using kats
    try:
        selected_features = [
            "mean",
            "var",
            "length",
            "stability",
            "crossing_points",
        ]
        feature_extractor = FeatureExtractor(features, sampling_frequency=10)
        extracted_features = feature_extractor.extract_features(
            selected_features=selected_features
        )

        assert len(extracted_features) == 18
        assert extracted_features.shape[1] == 5 * 45

    finally:
        # Delete feature files
        experiment_dir = temp_data_dir / "features"
        for file in experiment_dir.iterdir():
            file.unlink()
        experiment_dir.rmdir()

        # Delete target file
        os.remove(temp_data_dir / "train.csv")

        temp_data_dir.rmdir()
