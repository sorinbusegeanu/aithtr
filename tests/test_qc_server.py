from __future__ import annotations

import json

from mcp_servers.qc.server import QCService


def test_qc_timeline_validate():
    service = QCService(artifact_root="/tmp")
    res = service.qc_timeline_validate({"duration_sec": 1, "scenes": []})
    assert res["ok"] is True
    res = service.qc_timeline_validate({})
    assert res["ok"] is False


def test_qc_ffprobe_parse(monkeypatch):
    def fake_run(cmd, check=False, capture_output=False):
        class Result:
            def __init__(self):
                self.stdout = json.dumps(
                    {
                        "format": {"duration": "2.5"},
                        "streams": [{"r_frame_rate": "30000/1001"}],
                    }
                ).encode("utf-8")

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    service = QCService(artifact_root="/tmp")
    info = service._ffprobe("video.mp4")
    assert info["duration_sec"] == 2.5
    assert round(info["fps"], 2) == 29.97


def test_qc_detect_silence(monkeypatch):
    def fake_run(cmd, stderr=None, stdout=None):
        class Result:
            def __init__(self):
                self.stderr = (
                    "silence_start: 1.00\nsilence_end: 2.50\nsilence_start: 5.00\nsilence_end: 5.50"
                ).encode("utf-8")

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    service = QCService(artifact_root="/tmp")
    gaps = service._detect_silence("video.mp4")
    assert gaps == [{"start_sec": 1.0, "end_sec": 2.5}, {"start_sec": 5.0, "end_sec": 5.5}]


def test_qc_detect_black(monkeypatch):
    def fake_run(cmd, stderr=None, stdout=None):
        class Result:
            def __init__(self):
                self.stderr = (
                    "black_start:1.00 black_end:2.00\nblack_start:3.00 black_end:4.00"
                ).encode("utf-8")

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    service = QCService(artifact_root="/tmp")
    ranges = service._detect_black("video.mp4")
    assert ranges == [{"start_sec": 1.0, "end_sec": 2.0}, {"start_sec": 3.0, "end_sec": 4.0}]
