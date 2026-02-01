# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    MEMORY_DB_PATH=/data/sqlite/memory.db \
    ASSET_CATALOG_PATH=/data/assets/catalog.json

CMD ["python", "-m", "orchestrator.cli", "--help"]
