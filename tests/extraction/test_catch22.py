"""
Test suite for the feature extraction using pycatch22.

"""

import os
from pathlib import Path

import pytest

from src.extraction.catch22 import FeatureExtractor
from src.loading.cnc_mill_tool_wear import Dataloader as CMW_Dataloader
from src.loading.condition_monitoring_of_hydraulic_systems import (
    Dataloader as CMHS_Dataloader,
)

def test_extraction_single_sampling_frequency() -> None:
    """Test feature extraction if only one global sampling frequency exists."""
    # Load the CNC Mill Tool Wear dataset
    temp_data_dir = Path(__file__).parent / "temp_data"
    dataloader = CMW_Dataloader()
    features, _ = dataloader.load_dataset(temp_data_dir)

    # Extract features using pycatch22
    try:
        feature_extractor = FeatureExtractor(features, sampling_frequency=10)
        extracted_features = feature_extractor.extract_features()

        assert len(extracted_features) == 18
        assert extracted_features.shape[1] == 22 * 45

    finally:
        # Delete feature files
        experiment_dir = temp_data_dir / "features"
        for file in experiment_dir.iterdir():
            file.unlink()
        experiment_dir.rmdir()

        # Delete target file
        os.remove(temp_data_dir / "train.csv")

        temp_data_dir.rmdir()

@pytest.mark.skip(reason="Feature extraction takes roughly 15 minutes")
def test_extraction_different_sampling_frequencies() -> None:
    """Test feature extraction if different sampling frequencies exist."""
    feature_sampling_rates = {
        "PS1": 100,
        "PS2": 100,
        "PS3": 100,
        "PS4": 100,
        "PS5": 100,
        "PS6": 100,
        "EPS1": 100,
        "FS1": 10,
        "FS2": 10,
        "TS1": 1,
        "TS2": 1,
        "TS3": 1,
        "TS4": 1,
        "VS1": 1,
        "CE": 1,
        "CP": 1,
        "SE": 1,
    }

    # Load the Condition Monitoring of Hydraulic Systems dataset
    temp_data_dir = (
        Path(__file__).parent
        / "temp_data"
        / "condition_monitoring_of_hydraulic_systems"
    )
    dataloader = CMHS_Dataloader()
    features, _ = dataloader.load_dataset(temp_data_dir)

    # Extract features using pycatch22
    try:
        feature_extractor = FeatureExtractor(
            features, sampling_frequency=feature_sampling_rates
        )
        extracted_features = feature_extractor.extract_features(catch24=True)

        assert len(extracted_features) == 2205
        assert extracted_features.shape[1] == 24 * 17
    finally:
        # Delete feature files
        experiment_dir = temp_data_dir / "features"
        for file in experiment_dir.iterdir():
            file.unlink()
        experiment_dir.rmdir()

        # Delete target file
        os.remove(temp_data_dir / "train.csv")

        temp_data_dir.rmdir()
