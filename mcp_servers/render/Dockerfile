FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir mcp

ENV ARTIFACT_ROOT=/data/artifacts \
    FFMPEG_BIN=ffmpeg \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7105 \
    MCP_PATH=/mcp

CMD ["python", "mcp_servers/render/mcp_server.py"]
