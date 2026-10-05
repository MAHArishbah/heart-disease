"""Paths, params, and the shared CV splitter, and creating the requried dirs .

"""

from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "model.joblib"
SPEC_PATH = MODELS_DIR / "feature_spec.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TRAIN_METRICS_PATH = REPORTS_DIR / "train_metrics.json"


def load_params(path: Path | str | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "params.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


PARAMS = load_params()

SEED = PARAMS["seed"]
MAX_ITER = PARAMS["model"]["max_iter"]
CV_FOLDS = PARAMS["model"]["cv_folds"]


def make_cv(seed_offset: int = 0) -> StratifiedKFold:
    """offset to change the default cv seed for OOFs."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=SEED + seed_offset
    )


cv = make_cv()


def ensure_dirs() -> None:
    """make the required directories if not exist already"""
    for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
