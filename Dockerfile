#----------build stage : install dependencies----------
FROM python:3.14-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements-serve.txt /tmp/
RUN python -m venv /opt/venv && /ppt/venv/bin/pip install -r /tmp/requirements-serve.txt

#---------------runtime stage: only what serving needs----------
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    REQUIRE_MODEL=1
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY params.yaml ./params.yaml
COPY src ./src
COPY models/model.joblib models/feature_spec.json ./models/

USER appuser
EXPOSE 8080
CMD exec uvicorn disease_pred.serve:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1