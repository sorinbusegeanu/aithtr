FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/assets/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    ASSET_CATALOG_PATH=/data/assets/catalog.json \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7101 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.assets.mcp_server"]
