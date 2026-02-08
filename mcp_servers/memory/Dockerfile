# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/memory/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt


COPY . /app

ENV MEMORY_DB_PATH=/data/sqlite/memory.db \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7102 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.memory.mcp_server"]
