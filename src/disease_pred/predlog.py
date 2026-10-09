"""Per prediction records: a hashed line in the app log, full row in bigquery"""

import hashlib
import json
import logging
import os
from datetime import datetime,timezone

log=logging.getLogger("disease_pred.predictions")
TABLE=os.getenv("PREDICTIONS_TABLE")
_client=None

def _bigquery():
    global _client
    if _client is None:
        from google.cloud import bigquery
        _client=bigquery.Client()
    return _client

def record(request_id,client,frame,probabilities,threshold,model_version)->None:
    now=datetime.now(timezone.utc).isoformat()
    rows=[]
    for (_,row),p in zip(frame.iterrows(),probabilities):
        features = {k:(None if v !=v else float(v)) for k, v in row.items()} #nan to None
        digest=hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()[:16]
        log.info("prediction", extra={
            "request_id": request_id,
            "client":client,
            "row_hash": digest,
            "probability": round(float(p), 4),
            "flag": bool(p >= threshold),
            "n_missing": sum(v is None for v in features.values()),
        })
        rows.append({
            "request_id": request_id, "ts": now, "model_version": model_version,"client":client,
            "probability": float(p), "flag": bool(p >= threshold),
            "features": json.dumps(features),
        })
    if TABLE:
        errors=_bigquery().insert_rows_json(TABLE,rows) #sets netwrok round trip to eacj process , move to background process later TODO
        if errors:
            log.error("bigquery insert failed", extra={"errors": str(errors)[:500]})


