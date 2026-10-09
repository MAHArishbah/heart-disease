# File: src/disease_pred/drift.py · new
"""Weekly drift check: PSI of each continuous feature, training reference vs the last 7 days.

    python -m disease_pred.drift

Exits 1 when any feature's PSI exceeds ALERT, so the Cloud Run job shows as failed.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from .config import MODELS_DIR, PARAMS

REFERENCE = MODELS_DIR / "reference.csv"
TABLE = os.environ.get("PREDICTIONS_TABLE", "")
WINDOW_DAYS = 7
MIN_ROWS = 50          # below this the shares are too noisy to compare
ALERT = 0.25
EPS = 1e-4             # floor for empty bins; ln(0) is undefined


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(reference, edges)[0] / len(reference)
    a = np.histogram(current, edges)[0] / len(current)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_current() -> pd.DataFrame:
    from google.cloud import bigquery
    sql = (f"SELECT features FROM `{TABLE}` "
           f"WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {WINDOW_DAYS} DAY) "
           f"AND IFNULL(client,'api') != 'ui'")
    rows = bigquery.Client().query(sql).result()
    return pd.DataFrame([json.loads(r["features"]) if isinstance(r["features"], str)
                         else r["features"] for r in rows])


def main() -> int:
    reference = pd.read_csv(REFERENCE)
    current = load_current()
    if len(current) < MIN_ROWS:
        print(json.dumps({"status": "skipped", "n_current": len(current)}))
        return 0

    cont = [c for c in PARAMS["features"]["roles"]["cont"] if c in reference.columns]
    scores = {c: round(psi(reference[c], current[c]), 4) for c in cont}
    missing_shift = {c: round(float(current[c].isna().mean() - reference[c].isna().mean()), 3)
                     for c in reference.columns if c in current.columns}
    drifted = [c for c, v in scores.items() if v > ALERT]

    print(json.dumps({"status": "drift" if drifted else "ok", "n_current": len(current),
                      "psi": scores, "missing_rate_shift": missing_shift, "drifted": drifted}))
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())