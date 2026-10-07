from prometheus_client import Counter,Histogram

REQUESTS=Counter("predict_requests_total","Prediction requests",['path','status','model_version'])

LATENCY = Histogram(
    "predict_latency_seconds", "End-to-end request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))

SCORE = Histogram(
    "predict_score", "Predicted probability of disease",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))

FLAGGED = Counter("predict_flagged_total", "Predictions at or above the threshold")