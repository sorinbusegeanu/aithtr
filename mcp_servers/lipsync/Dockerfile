# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/lipsync/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt



RUN python - <<'PY'
import io
import urllib.request
import zipfile
from pathlib import Path

url = "https://github.com/Rudrabha/Wav2Lip/archive/refs/heads/master.zip"
dest = Path("/opt/wav2lip")
tmp = Path("/tmp/wav2lip.zip")

with urllib.request.urlopen(url) as r:
    tmp.write_bytes(r.read())

with zipfile.ZipFile(tmp, "r") as z:
    z.extractall("/tmp")

extracted = Path("/tmp/Wav2Lip-master")
if dest.exists():
    for p in sorted(dest.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    dest.rmdir()
extracted.rename(dest)
tmp.unlink(missing_ok=True)
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2




RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        librosa==0.8.1 \
        numpy==1.23.5 \
        scipy==1.10.1 \
        opencv-python-headless==4.9.0.80 \
        tqdm==4.66.1 \
        numba==0.57.1 \
        imageio-ffmpeg==0.5.1

RUN python - <<'PY'
import os
import pathlib
import imageio_ffmpeg

src = imageio_ffmpeg.get_ffmpeg_exe()
dst = "/usr/local/bin/ffmpeg"
pathlib.Path("/usr/local/bin").mkdir(parents=True, exist_ok=True)
if os.path.lexists(dst):
    os.unlink(dst)
os.symlink(src, dst)
print("linked ffmpeg:", dst, "->", src)
PY

RUN mkdir -p /opt/wav2lip/face_detection/detection/sfd
RUN mkdir -p /opt/wav2lip/temp && chmod 0777 /opt/wav2lip/temp

COPY . /app

ENV ARTIFACT_ROOT=/data/artifacts \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=7104 \
    MCP_PATH=/mcp \
    WAV2LIP_SCRIPT=/opt/wav2lip/inference.py

CMD ["python", "-m", "mcp_servers.lipsync.mcp_server"]
