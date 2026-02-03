# syntax=docker/dockerfile:1.5
FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

COPY mcp_servers/qwen_tts/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    ARTIFACT_AUDIO_ROOT=/data/artifacts/audio \
    VOICE_MAP_PATH=/data/tts/voice_map.json \
    XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2 \
    XTTS_DEVICE=cuda \
    XTTS_LANGUAGE=en \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7203 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.qwen_tts.mcp_server"]
