from __future__ import annotations

import importlib
import os

import pytest


def test_qwen_tts_voice_map(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    server = importlib.import_module("mcp_servers.qwen_tts.server")
    vm = server.VoiceMap(str(tmp_path / "voice_map.json"), ["a", "b"])
    assert vm.resolve("char1") == "a"
    assert vm.resolve("char2") == "b"
    assert vm.resolve("char1") == "a"


def test_qwen_tts_emotion_instruct(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    server = importlib.import_module("mcp_servers.qwen_tts.server")
    os.environ["VOICE_MAP_PATH"] = str(tmp_path / "voice_map.json")
    os.environ["ARTIFACT_ROOT"] = str(tmp_path / "artifacts")
    os.environ["ARTIFACT_AUDIO_ROOT"] = str(tmp_path / "artifacts" / "audio")
    service = server.QwenTTSService()
    assert "Sad" in service._emotion_to_instruct("sad", None)
    assert "custom" in service._emotion_to_instruct("angry", "custom")
