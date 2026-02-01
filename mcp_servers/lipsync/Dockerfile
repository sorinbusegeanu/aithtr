# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/lipsync/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip /opt/wav2lip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        librosa==0.8.1 \
        numpy==1.23.5 \
        scipy==1.10.1 \
        opencv-python==4.9.0.80 \
        tqdm==4.66.1 \
        numba==0.57.1

RUN mkdir -p /opt/wav2lip/face_detection/detection/sfd

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7104 \
    MCP_PATH=/mcp \
    WAV2LIP_SCRIPT=/opt/wav2lip/inference.py

CMD ["python", "-m", "mcp_servers.lipsync.mcp_server"]
