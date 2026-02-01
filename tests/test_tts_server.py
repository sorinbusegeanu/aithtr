from __future__ import annotations

import os

from mcp_servers.tts.server import TTSService


def test_tts_list_voices(tmp_path, monkeypatch):
    voices_path = tmp_path / "voices.json"
    voices_path.write_text('["a","b"]', encoding="utf-8")
    monkeypatch.setenv("TTS_VOICES_PATH", str(voices_path))
    service = TTSService(artifact_root=str(tmp_path / "artifacts"))
    res = service.tts_list_voices()
    assert res["voices"] == ["a", "b"]


def test_tts_synthesize_piper(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "piper")
    monkeypatch.setenv("PIPER_BIN", "piper")

    def fake_run(cmd, input=None, check=False):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    service = TTSService(artifact_root=str(tmp_path / "artifacts"))
    res = service.tts_synthesize("hello", "voice.onnx", style={"speaker": 0}, tags=["tts"])
    assert "artifact_id" in res


def test_tts_synthesize_coqui(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "coqui")
    monkeypatch.setenv("COQUI_TTS_BIN", "tts")

    def fake_run(cmd, check=False):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    service = TTSService(artifact_root=str(tmp_path / "artifacts"))
    res = service.tts_synthesize("hello", "model", style=None, tags=["tts"])
    assert "artifact_id" in res
