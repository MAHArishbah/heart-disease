"""train.py -- cells 37, 51, 56, and the artifacts they write."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from conftest import needs_model
from disease_pred import train
from disease_pred.config import PARAMS, SEED, SPEC_PATH


def test_cost_threshold_is_the_bayes_rule():
    assert train.cost_threshold(1.0, 4.0) == pytest.approx(0.2)
    assert train.cost_threshold(1.0, 1.0) == pytest.approx(0.5)
    assert train.cost_threshold(1.0, 9.0) == pytest.approx(0.1)


def test_cost_threshold_defaults_to_params_yaml():
    cfg = PARAMS["decision"]
    expected = cfg["cost_fp"] / (cfg["cost_fp"] + cfg["cost_fn"])
    assert train.cost_threshold() == pytest.approx(expected)


def test_param_grid_rebuilt_from_yaml():
    start, stop, num = PARAMS["model"]["grid"]["C_logspace"]
    grid = train.param_grid()
    assert np.allclose(grid["clf__C"], np.logspace(start, stop, num))
    assert grid["clf__l1_ratio"] == PARAMS["model"]["grid"]["l1_ratio"]


def test_pipeline_shape(prepared):
    spec = prepared["spec"]
    pipe = train.build_pipeline(spec["numeric"], spec["binary"],
                                spec["categorical"], spec["indicators"])
    assert list(pipe.named_steps) == ["pre", "clf"]
    clf = pipe.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.solver == "saga"
    assert clf.l1_ratio is not None
    assert clf.random_state == SEED


def test_pipeline_consumes_raw_columns(prepared):
    """No hand-preprocessing between the CSV and the estimator."""
    spec = prepared["spec"]
    pipe = train.build_pipeline(spec["numeric"], spec["binary"],
                                spec["categorical"], spec["indicators"])
    X = prepared["X_train"]
    assert X.isna().any().any(), "fixture has no NaNs left to prove the point"
    pipe.fit(X, prepared["y_train"])
    p = pipe.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


@needs_model
def test_spec_records_the_threshold_as_an_artifact():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["threshold"] == pytest.approx(train.cost_threshold())
    assert spec["cost_fn"] / spec["cost_fp"] == pytest.approx(4.0)


@needs_model
def test_spec_column_lists_are_consistent():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["numeric"] + spec["binary"] + spec["categorical"] == spec["features"]
    assert not set(spec["features"]) & set(spec["dropped_high_missing"])
    assert len(spec["design_columns"]) >= len(spec["features"])


@needs_model
def test_saved_features_match_a_fresh_prepare(prepared):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["features"] == prepared["spec"]["features"]


@needs_model
def test_spec_records_the_indicator_list(prepared):
    """The indicator list is part of the contract, not an implementation detail.

    serve.py scores against the saved coefficients, so a spec that does not say
    which columns got a flag cannot be checked against the model it ships with.
    """
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert "indicators" in spec, (
        "models/feature_spec.json predates the indicator change -- "
        "rerun `python -m disease_pred.train`")
    assert spec["indicators"] == prepared["spec"]["indicators"]


@needs_model
def test_saved_design_columns_match_the_saved_indicator_list():
    """Cells 25-27: chol and fbs indicators were rejected as site proxies.

    design_columns is what the fitted model actually has coefficients for, so
    this is the assertion that catches a stale artifact after the yaml changes.
    """
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert "indicators" in spec, (
        "models/feature_spec.json predates the indicator change -- "
        "rerun `python -m disease_pred.train`")
    flagged = {c.removeprefix("missingindicator_")
               for c in spec["design_columns"]
               if c.startswith("missingindicator_")}
    assert flagged == set(spec["indicators"]), (
        "model.joblib was fitted with a different indicator list than the "
        "spec declares -- rerun `python -m disease_pred.train`")
