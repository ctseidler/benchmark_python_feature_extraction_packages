import os
from pathlib import Path

import pytest

from src.loading.idmt_isa_compressed_air_dataset import Dataloader

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "idmt-isa_compressed_air"


@pytest.mark.skip(reason="Download size = 22.1 GB")
def test_download() -> None:
    """Test downloading the dataset."""
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(DATA_DIR)

    assert len(features) == 4
    assert targets.shape[1] == 1


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data not found")
@pytest.mark.skip(reason="Very slow loading process, as dataset is very large")
def test_load_from_dir() -> None:
    """Test loading the dataset from a a directory."""
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(DATA_DIR, session_filter=1, mic_filter=[1, 2, 3, 4])

    assert len(features) == 4
    assert targets.shape[1] == 1
