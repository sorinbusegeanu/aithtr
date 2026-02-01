from __future__ import annotations

import os

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.lipsync.server import LipSyncService


def test_lipsync_render_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("WAV2LIP_SCRIPT", "wav2lip.py")
    monkeypatch.setenv("WAV2LIP_PYTHON", "python")

    def fake_run(cmd, check=False):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    store = ArtifactStore(root=str(tmp_path / "artifacts"))
    avatar_id = store.put(b"avatar", content_type="image/png", tags=["avatar"])
    wav_id = store.put(b"audio", content_type="audio/wav", tags=["tts"])
    service = LipSyncService(artifact_root=str(tmp_path / "artifacts"))
    res = service.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_id, style=None)
    assert "artifact_id" in res
