"""Put src/ on sys.path and share the expensive fixtures."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from disease_pred import data as data_mod          # noqa: E402
from disease_pred.config import MODEL_PATH         # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return data_mod.load_raw()


@pytest.fixture(scope="session")
def cleaned(raw):
    return data_mod.clean(raw)


@pytest.fixture(scope="session")
def prepared(cleaned):
    return data_mod.prepare(cleaned)


needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.joblib missing -- run `python -m disease_pred.train`",
)
