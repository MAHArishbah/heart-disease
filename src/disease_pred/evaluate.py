"""Scoring helper and the test-set report. Cells 34 (helper), 72, 76.

    python -m disease_pred.evaluate

reads models/model.joblib, scores the held-out split, writes
reports/metrics.json.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from . import data as data_mod
from .config import METRICS_PATH, MODEL_PATH, SPEC_PATH, ensure_dirs
from .train import oof_predictions


def score(y_true, p_hat, thresh: float = 0.5, name: str = "",split: str = "") -> dict:
    """evaluate the models with acc,pre,rec,f1,auc,ap,brier"""
    y_pred = (p_hat >= thresh).astype(int)
    return {"model": name, "split": split, "threshold": round(thresh, 3),
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_true, p_hat), 3),
            "avg_precision": round(average_precision_score(y_true, p_hat), 3),
            "brier": round(brier_score_loss(y_true, p_hat), 3)}


def load_artifacts():
    """loads the artifacts stored after traiing , model.joblib and spec.json"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found -- run `python -m disease_pred.train` first."
        )
    model = joblib.load(MODEL_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return model, spec


def main() -> dict:
    ensure_dirs()
    model, spec = load_artifacts()
    threshold = spec["threshold"]

    prepared = data_mod.prepare()
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    # The saved column list is the contract, not whatever prepare() happened to
    # produce this run. Reindexing here is the same guard serve.py applies.
    X_test = X_test.reindex(columns=spec["features"])
    p_test = model.predict_proba(X_test)[:, 1]
    p_oof = oof_predictions(model, X_train, y_train)

    rows = [
        score(y_test, p_test, 0.5, "FInal model 0.5 thresh", "test"),
        score(y_test, p_test, threshold, f"final model @ {threshold:.2f}", "test"),
        score(y_train, p_oof, 0.5, "final model (OOF) 0.5 thresh", "oof"),
        score(y_train, p_oof, threshold,
              f"final model (OOF) @ {threshold:.2f}", "oof"),
    ]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    
    y_pred = (p_test >= threshold).astype(int)
    report = classification_report(y_test, y_pred, digits=3)
    print()
    print(report)

    headline = next(r for r in rows
                    if r["split"] == "test" and r["threshold"] == round(threshold, 3))
    metrics = {
        "threshold": threshold,
        "test_auc": headline["roc_auc"],
        "test_ap": headline["avg_precision"],
        "test_brier": headline["brier"],
        "test_recall": headline["recall"],
        "test_precision": headline["precision"],
        "test_f1": headline["f1_score"],
        "test_accuracy": headline["accuracy"],
        "oof_auc": float(np.round(roc_auc_score(y_train, p_oof), 3)), #recomputed again , read from headline TODO
        "oof_brier": float(np.round(brier_score_loss(y_train, p_oof), 3)),
        "n_test": int(len(y_test)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (METRICS_PATH.parent / "classification_report.txt").write_text(
        report, encoding="utf-8")
    print(f"\nwrote {METRICS_PATH.name}")

    from .tracking import log_run
    log_run(model,X_test)

    return {"rows": rows, "table": table, "metrics": metrics,
            "p_test": p_test, "p_oof": p_oof}


if __name__ == "__main__":
    main()
