# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/render/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt


COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    FFMPEG_BIN=ffmpeg \
    SUBTITLES_MODE=off \
    SUBTITLES_LANG=en \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7105 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.render.mcp_server"]
