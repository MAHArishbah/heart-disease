"""FastAPI prediction endpoint. The piece the notebook never had.

    uvicorn disease_pred.serve:app --reload

Two guards on the design matrix:
  1. the preprocessor uses remainder='drop', so an unexpected column cannot be
     appended to the design matrix and silently break coefficient alignment;
  2. the payload is reindexed to the saved feature list before it ever reaches
     the model, so columns arrive in the trained order and absent ones become
     NaN for the fitted imputer to fill.

The threshold comes from feature_spec.json, never from a literal in this file:
it is a fitted quantity , so a retrain that shifts it must shift what
the service serves.
"""

from __future__ import annotations

import json,os,math
from typing import Any,Literal
from contextlib import asynccontextmanager
import logging,time,uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException,Request,Response
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, ConfigDict, Field,field_validator,model_validator
from .logging_conf import configure_logging
from contextvars import ContextVar
from .predlog import record
from prometheus_client import CONTENT_TYPE_LATEST,generate_latest
from .metrics import FLAGGED,LATENCY,REQUESTS,SCORE
from .config import MODEL_PATH, SPEC_PATH
from .data import apply_zero_rules
import numpy as np

MODEL_VERSION=os.getenv("MODEL_VERSION","dev")
REQUIRE_MODEL=os.getenv("REQUIRE_MODEL","0")=="1"
CLINICAL_FIELDS = ["cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
MIN_CLINICAL = 3 
MAX_BATCH=500
SCORED_PATHS={"/predict","/predict_batch"}
_MODEL: Any = None
_SPEC: dict | None = None


request_id_var:ContextVar[str]=ContextVar("request_id",default="-")

def n_provided(p: "Patient") -> int:
    return sum(getattr(p, f) is not None for f in CLINICAL_FIELDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if REQUIRE_MODEL:
        load_artifacts()
    yield

app = FastAPI(
    title="Heart disease risk",
    description="Cost-weighted screening model over the UCI heart-disease battery.",
    version="1.0.0",
)
configure_logging()
log=logging.getLogger("disease_pred.serve")

@app.exception_handler(RequestValidationError)
async def log_validation_error(request:Request,exc: RequestValidationError):
    """logging which fields failed and why , not the values, then returns fastapi's normal 422"""
    errors= [{"loc":[str(p) for p in e["loc"]],"type":e["type"],"msg":e["msg"]} for e in exc.errors()]
    log.warning("validation_failed",extra={"request_id":request_id_var.get(),"path":request.url.path,"errors":errors})
    return await request_validation_exception_handler(request,exc)


@app.get("/metrics",include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def access_log(request: Request,call_next):
    request_id= request.headers.get("x-request-id") or uuid.uuid4().hex
    token=request_id_var.set(request_id)
    started=time.perf_counter()
    status=500
    try:
        response=await call_next(request)
        status=response.status_code      
        response.headers["x-request-id"] = request_id
        response.headers["x-model-version"] = MODEL_VERSION
        return response
    except Exception:
        log.exception("unhandled error",extra={"request_id":request_id})
        raise
    finally:
        log.info("request", extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        if request.url.path in SCORED_PATHS:
            REQUESTS.labels(request.url.path,str(status),MODEL_VERSION).inc()
            LATENCY.labels(request.url.path).observe(time.perf_counter()-started)
        request_id_var.reset(token)

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

    model_config = ConfigDict(extra="forbid")

    #required
    age: float = Field(ge=18, le=120, description="years; required")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female; required")

    # categorical codes: only these values are accepted
    cp: Literal[1, 2, 3, 4] | None = Field(None, description="chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    fbs: Literal[0, 1] | None = Field(None, description="fasting blood sugar > 120 mg/dl: 1 yes, 0 no")
    restecg: Literal[0, 1, 2] | None = Field(None, description="resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    exang: Literal[0, 1] | None = Field(None, description="exercise-induced angina: 1 yes, 0 no")
    slope: Literal[1, 2, 3] | None = Field(None, description="slope of peak exercise ST: 1 up, 2 flat, 3 down")
    ca: Literal[0, 1, 2, 3] | None = Field(None, description="major vessels coloured, 0-3")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 = normal, 6 = fixed, 7 = reversible")

    # continuous measurements: physiologically possible ranges
    trestbps: float | None = Field(None, ge=0, le=250, description="resting BP, mm Hg (0 = not measured)")
    chol: float | None = Field(None, ge=0, le=700, description="serum cholesterol, mg/dl (0 = not measured)")
    thalach: float | None = Field(None, ge=50, le=230, description="max heart rate achieved, bpm")
    oldpeak: float | None = Field(None, ge=-3, le=7, description="ST depression vs rest")

    @field_validator("*",mode="before")
    @classmethod
    def nan_to_none(cls,v):
        """treating nan like an omitted field"""
        return None if isinstance(v,float) and math.isnan(v) else v
    
    @field_validator('sex')
    @classmethod
    def sex_is_binary(cls,v:float) -> float:
        if v not in (0.0,1.0):
            raise ValueError('sex must be 0(female) or 1 (male)')
        return v
    
    @model_validator(mode='after')
    def plausible_comb(self):
        if self.thalach is not None and self.thalach >260- self.age:
            raise ValueError(f"thalach={self.thalach} is implausibly high for age={self.age}") #google says 220- age,+ 40 for generousity
        return self


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

    #output val check 
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities <0) | (probabilities >1)):
        log.error("invalid model output", extra={"probabilities": [float(p) for p in probabilities[:10]]})
        raise HTTPException(status_code=500, detail="model produced an invalid probability")

    record(request_id_var.get(),frame,probabilities,threshold,MODEL_VERSION)
    for p in probabilities:
        SCORE.observe(float(p))
    FLAGGED.inc(int((probabilities >= threshold).sum()))
    return [
        Prediction(
            probability=round(float(p), 4),
            threshold=threshold,
            flag=bool(p >= threshold),
            label="disease" if p >= threshold else "no disease",
        )
        for p in probabilities
    ]

@app.get("/live")
def live() ->dict:
    """Liveness: the process is up. Deliberately checks nothing else."""
    return {'status':'alive'}


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
        "model_version":MODEL_VERSION,
    }


@app.get("/schema")
def feature_schema() -> dict:
    """What the model actually consumes, straight from the saved spec."""
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "features": spec["features"],
        "dropped_high_missing": spec["dropped_high_missing"],
        "threshold": spec["threshold"],
        "cost_ratio_fn_to_fp": spec["cost_fn"] / spec["cost_fp"],
        "required_fields": [n for n, f in Patient.model_fields.items() if f.is_required()],
    }


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient,response:Response) -> Prediction:
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    provided=n_provided(patient)
    if provided < MIN_CLINICAL: #in headers and a warning , rather than rejection . dont want to write tests again and break Ui checkboxes
        response.headers["x-low-information"] = f"{provided} of {len(CLINICAL_FIELDS)} clinical fields provided"
        log.warning("low-information request", extra={"n_provided": provided})

    return predict_frame(to_frame([patient], spec))[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(patients: list[Patient]) -> list[Prediction]:
    # if not patients:
    #     raise HTTPException(status_code=422, detail="empty batch")
    if not 1<=len(patients) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"send between 1 and {MAX_BATCH} patients per request, got {len(patients)}")
    try:
        _, spec = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predict_frame(to_frame(patients, spec))
