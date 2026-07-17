import pytest

from src.loading.turning_chatter import Dataloader


@pytest.mark.skip(reason="Loads the full Turning dataset (~8M rows); run manually")
def test_load_dataset() -> None:
    """Test loading of the dataset."""
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset()

    assert len(features) == 1
    assert len(targets) == 808
    assert targets.shape[0] == 808
    assert targets.shape[1] == 2

    assert len(features["acc-x"]) == 8080000
