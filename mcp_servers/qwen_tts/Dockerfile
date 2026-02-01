# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/qwen_tts/requirements.txt /tmp/requirements.txt
COPY data/models/flash_attn/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl /tmp/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl
RUN apt-get update && apt-get install -y ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    ARTIFACT_AUDIO_ROOT=/data/artifacts/audio \
    VOICE_MAP_PATH=/data/tts/voice_map.json \
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    QWEN_TTS_FLASH_ATTN=1 \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7203 \
    MCP_PATH=/mcp

CMD ["python", "-m", "mcp_servers.qwen_tts.mcp_server"]
