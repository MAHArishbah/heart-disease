"""FastAPI prediction endpoint. The piece the notebook never had.

    uvicorn disease_pred.serve:app --reload

There is almost nothing to reimplement here, and that is the point. Cell 28
built a ColumnTransformer and cells 36-37 wrapped it with the estimator in a
Pipeline, so model.joblib takes *raw* columns and returns a probability. No
imputation, scaling or encoding is redone at request time -- doing so with
values recomputed from incoming rows would be train/serve skew.

Two guards on the design matrix:
  1. the preprocessor uses remainder='drop', so an unexpected column cannot be
     appended to the design matrix and silently break coefficient alignment;
  2. the payload is reindexed to the saved feature list before it ever reaches
     the model, so columns arrive in the trained order and absent ones become
     NaN for the fitted imputer to fill.

The threshold comes from feature_spec.json, never from a literal in this file:
it is a fitted quantity (cell 51), so a retrain that shifts it must shift what
the service serves.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field,field_validator

from .config import MODEL_PATH, SPEC_PATH
from .data import apply_zero_rules

app = FastAPI(
    title="Heart disease risk",
    description="Cost-weighted screening model over the UCI heart-disease battery.",
    version="1.0.0",
)

_MODEL: Any = None
_SPEC: dict | None = None


def load_artifacts(force: bool = False) -> tuple[Any, dict]:
    """Load model + spec once and memoise. Raises if training has not been run."""
    global _MODEL, _SPEC
    if _MODEL is None or _SPEC is None or force:
        if not MODEL_PATH.exists() or not SPEC_PATH.exists():
            raise FileNotFoundError(
                "models/model.joblib or models/feature_spec.json is missing -- "
                "run `python -m disease_pred.train` first."
            )
        _MODEL = joblib.load(MODEL_PATH)
        _SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return _MODEL, _SPEC


class Patient(BaseModel):
    """One raw record, exactly as it appears in the source CSVs.
    required columns: age,sex
    every other column is optional: the fitted imputer is the model's own answer to
    missing data .Unknown keys are ignored rather than
    rejected -- the reindex below is what decides the design matrix.
    """

    model_config = ConfigDict(extra="ignore")

    age: float = Field(gt=0,le=120 ,description="years;required")
    sex: float = Field( description="1 = male, 0 = female;required")
    cp: float | None = Field(None, description="chest pain type, 1-4")
    trestbps: float | None = Field(None, description="resting BP, mm Hg")
    chol: float | None = Field(None, description="serum cholesterol, mg/dl")
    fbs: float | None = Field(None, description="fasting blood sugar > 120 mg/dl")
    restecg: float | None = Field(None, description="resting ECG result, 0-2")
    thalach: float | None = Field(None, description="max heart rate achieved")
    exang: float | None = Field(None, description="exercise-induced angina")
    oldpeak: float | None = Field(None, description="ST depression vs rest")
    slope: float | None = Field(None, description="slope of peak exercise ST")
    ca: float | None = Field(None, description="major vessels coloured, 0-3")
    thal: float | None = Field(None, description="3 = normal, 6 = fixed, 7 = reversible")

    @field_validator('sex')
    @classmethod
    def sex_is_binary(cls,v:float) -> float:
        if v not in (0.0,1.0):
            raise ValueError('sex must be 0(female) or 1 (male)')
        return v


class Prediction(BaseModel):
    probability: float
    threshold: float
    flag: bool
    label: str


def to_frame(patients: list[Patient], spec: dict) -> pd.DataFrame:
    """Payload -> design matrix, in the trained column order.

    `apply_zero_rules` is the one piece of data.clean() that has to run here
    too: training never saw chol=0 or trestbps=0 (cell 8 turned them all into
    NaN), so a payload carrying a literal 0 must become NaN for the fitted
    imputer rather than being scaled as a real measurement.

    An absent field arrives as None, which in a pandas object column is NOT
    what SimpleImputer looks for -- it would sail past the imputer and reach
    the encoder as an unknown category, silently, scoring the patient against
    a design matrix nobody intended. Coercing to float turns every gap into a
    real NaN first.
    """
    frame = pd.DataFrame([p.model_dump() for p in patients],
                         columns=list(Patient.model_fields))
    frame = frame.astype("float64")
    return apply_zero_rules(frame).reindex(columns=spec["features"])


def predict_frame(frame: pd.DataFrame) -> list[Prediction]:
    model, spec = load_artifacts()
    threshold = spec["threshold"]
    probabilities = model.predict_proba(frame)[:, 1]
    return [
        Prediction(
            probability=round(float(p), 4),
            threshold=threshold,
            flag=bool(p >= threshold),
            label="disease" if p >= threshold else "no disease",
        )
        for p in probabilities
    ]


@app.get("/health")
def health() -> dict:
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "threshold": spec["threshold"],
        "n_features": len(spec["features"]),
        "trained_at": spec.get("trained_at"),
    }


@app.get("/schema")
def feature_schema() -> dict:
    """What the model actually consumes, straight from the saved spec."""
    _, spec = load_artifacts()
    return {
        "features": spec["features"],
        "dropped_high_missing": spec["dropped_high_missing"],
        "threshold": spec["threshold"],
        "cost_ratio_fn_to_fp": spec["cost_fn"] / spec["cost_fp"],
        "required_fields": ["age", "sex"],
    }


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predict_frame(to_frame([patient], spec))[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(patients: list[Patient]) -> list[Prediction]:
    if not patients:
        raise HTTPException(status_code=422, detail="empty batch")
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predict_frame(to_frame(patients, spec))
