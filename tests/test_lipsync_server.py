from __future__ import annotations

import io
import wave

import cv2

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.lipsync.server import LipSyncService


def _png_bytes() -> bytes:
    img = cv2.merge(
        [
            cv2.UMat(2, 2, cv2.CV_8UC1, 0).get(),
            cv2.UMat(2, 2, cv2.CV_8UC1, 0).get(),
            cv2.UMat(2, 2, cv2.CV_8UC1, 0).get(),
        ]
    )
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode test png")
    return encoded.tobytes()


def _wav_silence_bytes(duration_sec: float = 0.1, sample_rate: int = 16000) -> bytes:
    frames = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_lipsync_render_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("WAV2LIP_SCRIPT", "wav2lip.py")
    monkeypatch.setenv("WAV2LIP_PYTHON", "python")

    def fake_run(cmd, check=False, **kwargs):
        out_path = cmd[cmd.index("--outfile") + 1]
        with open(out_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    store = ArtifactStore(root=str(tmp_path / "artifacts"))
    avatar_id = store.put(_png_bytes(), content_type="image/png", tags=["avatar"])
    wav_id = store.put(_wav_silence_bytes(), content_type="audio/wav", tags=["tts"])
    service = LipSyncService(artifact_root=str(tmp_path / "artifacts"))
    res = service.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_id, style=None)
    assert "artifact_id" in res


def test_lipsync_render_clip_uses_static_fallback_on_wav2lip_error(tmp_path, monkeypatch):
    monkeypatch.setenv("WAV2LIP_SCRIPT", "wav2lip.py")
    monkeypatch.setenv("WAV2LIP_PYTHON", "python")
    monkeypatch.setenv("LIPSYNC_ALLOW_STATIC_FALLBACK", "1")

    def fake_run(cmd, check=False, **kwargs):
        if "--outfile" in cmd:
            raise RuntimeError("wav2lip failed")
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    store = ArtifactStore(root=str(tmp_path / "artifacts"))
    avatar_id = store.put(_png_bytes(), content_type="image/png", tags=["avatar"])
    wav_id = store.put(_wav_silence_bytes(), content_type="audio/wav", tags=["tts"])
    service = LipSyncService(artifact_root=str(tmp_path / "artifacts"))
    res = service.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_id, style=None)
    meta = store.get_metadata(res["artifact_id"])
    assert "fallback" in meta.tags
