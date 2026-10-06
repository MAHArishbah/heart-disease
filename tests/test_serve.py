"""serve.py, plus the acceptance test from the refactor plan."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import needs_model
from disease_pred import serve
from disease_pred.data import load_raw

pytestmark = needs_model


@pytest.fixture(scope="module")
def client():
    return TestClient(serve.app)


@pytest.fixture(scope="module")
def artifacts():
    return serve.load_artifacts(force=True)


# --------------------------------------------------------------------------
# The acceptance test: load model.joblib, call predict_proba on five raw rows
# straight from the CSV, and get the same numbers the fitted pipeline gives.
# --------------------------------------------------------------------------

@needs_model
def test_raw_csv_rows_round_trip_through_the_service(artifacts, prepared):
    model, spec = artifacts

    raw = load_raw()
    idx = list(prepared["X_test"].index[:5])

    patients = [serve.Patient(**raw.loc[i].drop(labels=["num", "site"]).to_dict())
                for i in idx]
    via_service = model.predict_proba(serve.to_frame(patients, spec))[:, 1]

    # What the notebook's `final_model` produced for the same rows.
    via_pipeline = model.predict_proba(prepared["X_test"].loc[idx])[:, 1]

    assert np.allclose(via_service, via_pipeline, atol=1e-12)


@needs_model
def test_whole_test_split_round_trips(artifacts, prepared):
    """Not just five rows -- no row may disagree between the two paths."""
    model, spec = artifacts
    raw = load_raw()
    idx = list(prepared["X_test"].index)

    patients = [serve.Patient(**raw.loc[i].drop(labels=["num", "site"]).to_dict())
                for i in idx]
    via_service = model.predict_proba(serve.to_frame(patients, spec))[:, 1]
    via_pipeline = model.predict_proba(prepared["X_test"].loc[idx])[:, 1]

    assert np.allclose(via_service, via_pipeline, atol=1e-12)


# --------------------------------------------------------------------------
# Payload handling
# --------------------------------------------------------------------------

def test_to_frame_uses_the_saved_column_order(artifacts):
    _, spec = artifacts
    frame = serve.to_frame([serve.Patient(age=54, sex=1, cp=4)], spec)
    assert list(frame.columns) == spec["features"]


def test_to_frame_turns_impossible_zeros_into_nan(artifacts):
    _, spec = artifacts
    frame = serve.to_frame([serve.Patient(age=54, sex=1, chol=0, trestbps=0)], spec)
    assert np.isnan(frame.loc[0, "chol"])
    assert np.isnan(frame.loc[0, "trestbps"])


def test_unexpected_payload_key_cannot_reach_the_design_matrix(artifacts):
    _, spec = artifacts
    patient = serve.Patient.model_validate(
        {"age": 54, "sex": 1, "surprise_column": 999})
    frame = serve.to_frame([patient], spec)
    assert "surprise_column" not in frame.columns
    assert list(frame.columns) == spec["features"]


def test_dropped_high_missing_columns_are_not_requested(artifacts):
    """thal/slope/ca were dropped at training; the service must not need them."""
    _, spec = artifacts
    assert spec["dropped_high_missing"]
    for col in spec["dropped_high_missing"]:
        assert col not in spec["features"]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

def test_health(client, artifacts):
    _, spec = artifacts
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["threshold"] == spec["threshold"]


def test_schema_endpoint_reports_the_cost_ratio(client):
    body = client.get("/schema").json()
    assert body["cost_ratio_fn_to_fp"] == pytest.approx(4.0)
    assert body["threshold"] == pytest.approx(0.2)


def test_predict_returns_a_calibrated_probability(client):
    body = client.post("/predict", json={
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    }).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["flag"] == (body["probability"] >= body["threshold"])
    assert body["label"] in {"disease", "no disease"}


def test_threshold_comes_from_the_artifact_not_a_literal(client, artifacts):
    _, spec = artifacts
    body = client.post("/predict", json={"age": 55, "sex": 1}).json()
    assert body["threshold"] == spec["threshold"]


# --------------------------------------------------------------------------
# Mandatory intake fields.
#
# The fitted imputer is the missing-data policy for *measurements that may not
# have been taken* -- a missing trestbps is itself one of the three kept
# indicators, so the model has seen that case and has a coefficient for it.
# age and sex are not measurements: they are present in all 920 source rows,
# so the imputer's behaviour for them was never exercised in training. An
# absent age would be filled with the training median and scored as if it were
# a real patient. That is the notebook's "add check before predict" TODO.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("payload, reason", [
    ({}, "neither field"),
    ({"sex": 1}, "no age"),
    ({"age": 55}, "no sex"),
    ({"age": None, "sex": 1}, "explicit null age"),
    ({"age": 55, "sex": None}, "explicit null sex"),
])
def test_age_and_sex_are_mandatory(client, payload, reason):
    assert client.post("/predict", json=payload).status_code == 422, reason


@pytest.mark.parametrize("sex", [0.5, 2, -1])
def test_sex_outside_its_domain_is_rejected(client, sex):
    """most_frequent imputation has no answer for 0.5, and the binary block
    would scale it as if it meant something."""
    r = client.post("/predict", json={"age": 55, "sex": sex})
    assert r.status_code == 422


@pytest.mark.parametrize("age", [0, -5, 130])
def test_age_outside_its_range_is_rejected(client, age):
    assert client.post("/predict", json={"age": age, "sex": 1}).status_code == 422


def test_valid_boundary_values_are_accepted(client):
    for sex in (0, 1):
        assert client.post("/predict", json={"age": 1, "sex": sex}).status_code == 200
    assert client.post("/predict", json={"age": 120, "sex": 1}).status_code == 200


def test_the_rest_of_the_battery_stays_optional(client):
    """Requiring age and sex must not turn into requiring everything -- the
    imputer is still the policy for the tests that were not run."""
    body = client.post("/predict", json={"age": 55, "sex": 1}).json()
    assert 0.0 <= body["probability"] <= 1.0


def test_schema_advertises_the_required_fields(client):
    assert client.get("/schema").json()["required_fields"] == ["age", "sex"]


def test_batch_rejects_the_whole_payload_if_one_row_is_incomplete(client):
    """Fail loud: a screening batch must not come back silently one row short."""
    r = client.post("/predict_batch",
                    json=[{"age": 63, "sex": 1}, {"sex": 0}])
    assert r.status_code == 422
    assert any(1 in err["loc"] for err in r.json()["detail"]), \
        "the error must name the offending row index"


def test_batch_matches_one_by_one(client):
    payload = [
        {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233},
        {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204},
        {"age": 67, "sex": 1, "cp": 4, "thalach": 108, "exang": 1},
    ]
    batch = [row["probability"] for row in
             client.post("/predict_batch", json=payload).json()]
    single = [client.post("/predict", json=row).json()["probability"]
              for row in payload]
    assert batch == single


def test_empty_batch_is_rejected(client):
    assert client.post("/predict_batch", json=[]).status_code == 422


def test_higher_risk_profile_scores_higher(client):
    low = client.post("/predict", json={
        "age": 35, "sex": 0, "cp": 3, "thalach": 185, "exang": 0, "oldpeak": 0.0,
    }).json()["probability"]
    high = client.post("/predict", json={
        "age": 68, "sex": 1, "cp": 4, "thalach": 105, "exang": 1, "oldpeak": 3.2,
    }).json()["probability"]
    assert high > low

def test_live_does_not_depend_on_model(client):
    body=client.get("/live").json()
    assert body == {'status':'alive'}

def test_health_reports_the_model_version(client):
    body=client.get('/health').json()
    assert body['model_version']== serve.MODEL_VERSION

