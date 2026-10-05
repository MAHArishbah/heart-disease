"""evaluate.py -- cell 34's helper, cells 72 and 76."""

import json

import numpy as np
import pytest

from conftest import needs_model
from disease_pred.config import METRICS_PATH
from disease_pred.evaluate import load_artifacts, score


def test_score_on_a_perfect_predictor():
    y = np.array([0, 0, 1, 1])
    row = score(y, np.array([0.01, 0.02, 0.98, 0.99]), 0.5, "perfect", "unit")
    assert row["roc_auc"] == 1.0
    assert row["recall"] == 1.0 and row["precision"] == 1.0
    assert row["accuracy"] == 1.0


def test_score_reports_the_threshold_it_used():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.3, 0.8, 0.25])
    assert score(y, p, 0.2, "", "")["recall"] == 1.0     # 0.3 and 0.8 both flagged
    assert score(y, p, 0.5, "", "")["recall"] == 0.5     # only 0.8
    assert score(y, p, 0.2, "", "")["threshold"] == 0.2


def test_score_survives_a_degenerate_prediction():
    """zero_division=0 -- a model that flags nothing must not raise."""
    y = np.array([0, 1, 0, 1])
    row = score(y, np.array([0.1, 0.1, 0.1, 0.1]), 0.5, "", "")
    assert row["precision"] == 0.0 and row["f1_score"] == 0.0


def test_score_keys_are_stable():
    row = score(np.array([0, 1]), np.array([0.2, 0.8]))
    assert set(row) == {"model", "split", "threshold", "accuracy", "precision",
                        "recall", "f1_score", "roc_auc", "avg_precision", "brier"}


@needs_model
def test_artifacts_load():
    model, spec = load_artifacts()
    assert hasattr(model, "predict_proba")
    assert list(model.named_steps) == ["pre", "clf"]
    assert spec["threshold"] == pytest.approx(0.2)


@needs_model
def test_metrics_json_is_present_and_sane():
    if not METRICS_PATH.exists():
        pytest.skip("run `python -m disease_pred.evaluate` first")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert 0.5 < metrics["test_auc"] <= 1.0
    assert metrics["test_recall"] > 0.9, "the 4:1 cost ratio should buy high recall"
    assert metrics["threshold"] == pytest.approx(0.2)


@needs_model
def test_model_beats_the_prevalence_baseline(prepared):
    """The floor cell 34's dummies established."""
    from sklearn.metrics import roc_auc_score

    model, spec = load_artifacts()
    X_test = prepared["X_test"].reindex(columns=spec["features"])
    p = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(prepared["y_test"], p) > 0.8
