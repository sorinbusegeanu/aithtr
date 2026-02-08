import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest


def _wait_for_url(url: str, timeout_sec: int = 10) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _reset_voice_map() -> None:
    paths = [
        os.getenv("VOICE_MAP_PATH", ""),
        os.path.join("data", "tts", "voice_map.json"),
        "/data/tts/voice_map.json",
    ]
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")
        except Exception:
            pass


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.integration
def test_qwen_tts_smoke(tmp_path):
    url = os.getenv("QWEN_TTS_URL") or "http://localhost:7203/mcp"
    use_external = True
    proc = None
    try:
        if not _wait_for_url(url, timeout_sec=5):
            print(f"TTS server not reachable at {url}")
            pytest.skip(f"TTS server not reachable at {url}")
        _reset_voice_map()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "tts_synthesize",
                "arguments": {
                    "text": "Hello there",
                    "character_id": "char_1",
                    "voice_id": "char_1",
                    "emotion": "neutral",
                    "style": None,
                    "output_format": "wav",
                },
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    finally:
        if proc is not None:
            proc.terminate()

    assert "result" in data, data
    wav_path = data["result"]["wav_path"]
    duration_ms = data["result"]["duration_ms"]
    print(f"wav_path={wav_path}")
    assert duration_ms > 0
    host_path = wav_path
    if not os.path.exists(host_path) and host_path.startswith("/data/"):
        host_path = os.path.join("data", host_path[len("/data/"):].lstrip("/"))
    assert os.path.exists(host_path)

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        subprocess.run(
            [ffprobe, "-hide_banner", "-loglevel", "error", host_path],
            check=True,
        )
