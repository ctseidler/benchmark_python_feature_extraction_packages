"""
Test suite for the CNC Mill Tool Wear dataset.

"""

import os
from pathlib import Path

from src.loading.cnc_mill_tool_wear import Dataloader

def test_load_dataset() -> None:
    """Test loading of the dataset."""
    temp_data_dir = Path(__file__).parent / "temp_data"
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(temp_data_dir)

    # Guarantee that the data is deleted even if the test fails
    try:
        assert len(features) == 45
        assert len(targets) == 18

        for feature_data in features.values():
            assert len(feature_data) == 25286

        assert targets.shape[1] == 2
    finally:
        # Delete feature files
        experiment_dir = temp_data_dir / "features"
        for file in experiment_dir.iterdir():
            file.unlink()
        experiment_dir.rmdir()

        # Delete target file
        os.remove(temp_data_dir / "train.csv")

        temp_data_dir.rmdir()
