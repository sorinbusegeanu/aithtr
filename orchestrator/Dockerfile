FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir mcp

ENV ARTIFACT_ROOT=/data/artifacts \
    MEMORY_DB_PATH=/data/sqlite/memory.db \
    ASSET_CATALOG_PATH=/data/assets/catalog.json

CMD ["python", "-m", "orchestrator.cli", "--help"]
