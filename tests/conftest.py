import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_resources() -> Path:
    os.environ["MBMBM_DATASET_DIR"] = str(Path(Path(__file__).parent / "resources" / "datasets"))
    return Path(__file__).parent / "resources" / "datasets"


@pytest.fixture(scope="session")
def ds_small_path(test_resources) -> Path:
    return test_resources / "ds_test_small"


@pytest.fixture(scope="session")
def ds_large_path(test_resources) -> Path:
    return test_resources / "ds_test_large"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return Path(__file__).parent / "resources" / "configs"


@pytest.fixture(scope="session")
def np_array_with_nan() -> np.array:
    return np.array([[1, 2, 3, 4], [5, 6, np.nan, 8], [9, 10, 11, 12]])


@pytest.fixture(scope="session")
def np_array_with_none() -> np.array:
    return np.array([[1, 2, 3, 4], [5, 6, None, 8], [9, 10, 11, 12]])


@pytest.fixture(scope="session")
def np_array() -> np.array:
    return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
