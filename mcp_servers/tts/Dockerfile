# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install mcp

ENV ARTIFACT_ROOT=/data/artifacts \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7103 \
    MCP_PATH=/mcp

CMD ["python", "mcp_servers/tts/mcp_server.py"]
