"""Fit, tune, choose the threshold, write the artifacts.

    python -m disease_pred.train

writes models/model.joblib, models/feature_spec.json and
reports/train_metrics.json.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

from . import data as data_mod
from .config import (MAX_ITER, MODEL_PATH, PARAMS, SEED, SPEC_PATH,
                     TRAIN_METRICS_PATH, ensure_dirs, make_cv)
from .features import build_preprocessor, output_names


def build_pipeline(numeric, binary, categorical,indicators) -> Pipeline:
    """builds the pipeline with preprocessor and classifer"""
    # l1_ratio is always overwritten by the grid; a default is set anyway so
    # that a bare pipeline is fittable outside GridSearchCV.
    return Pipeline([
        ("pre", build_preprocessor(numeric, binary, categorical,indicators)),
        ("clf", LogisticRegression( solver="saga",
                                   l1_ratio=0.5,
                                   max_iter=4 * MAX_ITER, random_state=SEED)),
    ])


def param_grid() -> dict:
    """ `params_grid`, rebuilt from params.yaml."""
    start, stop, num = PARAMS["model"]["grid"]["C_logspace"]
    return {
        "clf__C": np.logspace(start, stop, num),
        "clf__l1_ratio": PARAMS["model"]["grid"]["l1_ratio"],
    }


def build_search(numeric, binary, categorical,indicators, n_jobs: int = -1) -> GridSearchCV:
    """ builds pipeline with CV . Exposed so 02-diagnostics can reuse it  instead of redefining the search it is meant to be validating."""
    return GridSearchCV(
        build_pipeline(numeric, binary, categorical,indicators),
        param_grid(),
        scoring={"log_loss": "neg_log_loss", "auc": "roc_auc"},
        cv=make_cv(),
        n_jobs=n_jobs,
        return_train_score=True,
        refit="log_loss",
    )


def cost_threshold(cost_fp: float | None = None,cost_fn: float | None = None) -> float:
    """ The Bayes-optimal cut for an asymmetric cost ratio based on the decision during eda."""
    cfg = PARAMS["decision"]
    cost_fp = cfg["cost_fp"] if cost_fp is None else cost_fp
    cost_fn = cfg["cost_fn"] if cost_fn is None else cost_fn
    return cost_fp / (cost_fp + cost_fn)


def oof_predictions(model, X_train, y_train) -> np.ndarray:
    """`p_oof`. A separate fold seed from the tuning CV on purpose."""
    return cross_val_predict(
        model, X_train, y_train, cv=make_cv(seed_offset=1), method="predict_proba"
    )[:, 1]


def main() -> dict:
    ensure_dirs()

    prepared = data_mod.prepare(verbose=True)
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    spec = prepared["spec"]

    print(f"train {X_train.shape}  test {prepared['X_test'].shape}")
    print(f"dropped for missingness > "
          f"{PARAMS['features']['missing_threshold']}: "
          f"{spec['dropped_high_missing']}")

    
    gs = build_search(spec["numeric"], spec["binary"], spec["categorical"],spec["indicators"])
    gs.fit(X_train, y_train)
    print(f" best params :{gs.best_params_}")
    print(f" best score : {-gs.best_score_}")

    final_model = gs.best_estimator_
    feature_names = output_names(final_model.named_steps["pre"])
    design_means = final_model.named_steps["pre"].transform(X_train).mean(axis=0)

    # 
    p_oof = oof_predictions(final_model, X_train, y_train)
    threshold = cost_threshold()

    # --- cell 56 -----------------------------------------------------------
    # THRESHOLD looks like a constant but it is a fitted quantity: derived from
    # out-of-fold predictions under a stated 4:1 cost ratio, and resting on a
    # calibration assumption validated at cells 54-55 and 70-71. It travels
    # with the model rather than being hardcoded in serve.py, so a retrain that
    # shifts the optimal cut cannot silently keep serving the old one.
    print("chosen threshold:", round(threshold, 3))

    feature_spec = {
        "features": spec["features"],
        "numeric": spec["numeric"],
        "binary": spec["binary"],
        "categorical": spec["categorical"],
        "dropped_high_missing": spec["dropped_high_missing"],
        "design_columns": feature_names,
        "threshold": threshold,
        "cost_fp": PARAMS["decision"]["cost_fp"],
        "cost_fn": PARAMS["decision"]["cost_fn"],
        "best_params": {k: float(v) for k, v in gs.best_params_.items()},
        "seed": SEED,
        "n_train": int(len(X_train)),
        "train_prevalence": float(y_train.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "indicators":spec["indicators"],
        "design_means":[float(v) for v in design_means],
    }
    # print(feature_names)

    joblib.dump(final_model, MODEL_PATH)
    X_train.to_csv(MODEL_PATH.parent / "reference.csv", index=False) #for drift detection
    SPEC_PATH.write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")

    train_metrics = {
        "cv_log_loss": float(-gs.best_score_),
        "oof_auc": float(roc_auc_score(y_train, p_oof)),
        "oof_brier": float(brier_score_loss(y_train, p_oof)),
        "threshold": threshold,
    }
    TRAIN_METRICS_PATH.write_text(json.dumps(train_metrics, indent=2),
                                  encoding="utf-8")

    print(f"wrote {MODEL_PATH.name}, {SPEC_PATH.name}, {TRAIN_METRICS_PATH.name}")
    print(pd.Series(train_metrics).round(4))
    return {"model": final_model, "search": gs, "p_oof": p_oof,
            "spec": feature_spec, "metrics": train_metrics}


if __name__ == "__main__":
    main()
