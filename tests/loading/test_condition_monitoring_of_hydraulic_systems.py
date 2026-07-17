"""
Test suite for the Condition Monitoring of Hydraulic Systems dataset.

"""

from pathlib import Path

from src.loading.condition_monitoring_of_hydraulic_systems import Dataloader

FEATURE_SAMPLING_RATES = {
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

def test_load_dataset() -> None:
    """Test loading of the dataset."""
    temp_data_dir = Path(__file__).parent / "temp_data"
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(temp_data_dir)

    # Guarantee that the data is deleted even if the test fails
    try:
        assert len(features) == 17
        assert len(targets) == 2205

        for feature_name, feature_data in features.items():
            assert len(feature_data) == 2205 * 60 * FEATURE_SAMPLING_RATES[feature_name]

        assert targets.shape[0] == 2205
        assert targets.shape[1] == 5
    finally:
        for file in temp_data_dir.iterdir():
            file.unlink()
        temp_data_dir.rmdir()
