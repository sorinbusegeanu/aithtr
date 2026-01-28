FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir mcp

ENV ARTIFACT_ROOT=/data/artifacts \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7104 \
    MCP_PATH=/mcp

CMD ["python", "mcp_servers/lipsync/mcp_server.py"]
