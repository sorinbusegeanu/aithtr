# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/assets/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    ASSET_CATALOG_PATH=/data/assets/catalog.json \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7101 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.assets.mcp_server"]
