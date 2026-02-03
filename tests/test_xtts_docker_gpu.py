from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


IMAGE = os.getenv("XTTS_DOCKER_IMAGE", "ghcr.io/coqui-ai/tts:latest")


@pytest.mark.integration
def test_xtts_docker_gpu_generates_wav(tmp_path: Path) -> None:
    if os.getenv("XTTS_DOCKER_TEST", "0") != "1":
        pytest.skip("Set XTTS_DOCKER_TEST=1 to run Docker GPU XTTS test")
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    gpu_check = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", IMAGE, "nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if gpu_check.returncode != 0:
        pytest.skip("docker GPU is unavailable for XTTS test")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample.wav"

    script = (
        "from TTS.api import TTS\n"
        "tts=TTS(model_name='tts_models/en/vctk/vits', gpu=True)\n"
        "tts.tts_to_file(text='Hello from XTTS docker test.', speaker='p225', file_path='/out/sample.wav')\n"
        "print('ok')\n"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{out_dir}:/out",
        IMAGE,
        "python3",
        "-c",
        script,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"docker xtts run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert out_path.exists(), f"expected generated wav at {out_path}"
    assert out_path.stat().st_size > 1024, "generated wav is unexpectedly small"
