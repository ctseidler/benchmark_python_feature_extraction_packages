"""
Test suite for the MIMII dataset.

Notes
-----
We use 6_db_valve for this test, as it is the smallest subset (6.9 GB). The download will still
take several minutes, depending on your internet connection. Thus, the download test is skipped by 
default.

"""

import os
from pathlib import Path

import pytest

from src.loading.mimii import Dataloader

DATA_DIR = Path(
    r"C:\Projects\Benchmarking_Python_Feature_Extraction_Packages\data\mimii"
)

@pytest.mark.skip(reason="Download size = 6.9 GB")
def test_download() -> None:
    """Test downloading the dataset."""
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(DATA_DIR, -6, "valve", "id_00")

    assert len(features) == 8
    assert targets.shape[1] == 1

@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data not found")
@pytest.mark.skip(reason="Very slow loading process, as dataset is very large")
def test_load_from_dir() -> None:
    """Test loading the dataset from a a directory."""
    dataloader = Dataloader()
    features, targets = dataloader.load_dataset(DATA_DIR, -6, "valve", "id_00")

    assert len(features) == 8
    assert targets.shape[1] == 1
