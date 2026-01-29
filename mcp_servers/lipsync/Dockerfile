FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/lipsync/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7104 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.lipsync.mcp_server"]
