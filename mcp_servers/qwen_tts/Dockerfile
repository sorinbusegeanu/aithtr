FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/qwen_tts/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y ffmpeg sox && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /tmp/requirements.txt

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
