from __future__ import annotations

import os

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.render.server import RenderService, _fmt_number, _fmt_time, _is_image, _is_subtitle


def test_render_helpers(tmp_path):
    img = tmp_path / "bg.png"
    vid = tmp_path / "clip.mp4"
    sub = tmp_path / "subs.srt"
    img.write_bytes(b"img")
    vid.write_bytes(b"vid")
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    assert _is_image(str(img))
    assert not _is_image(str(vid))
    assert _is_subtitle(str(sub))
    assert _fmt_time(1.23456) == "1.235"
    assert _fmt_number(3.14159) == "3.14"


def test_render_resolve(tmp_path):
    store = ArtifactStore(root=str(tmp_path / "artifacts"))
    artifact_id = store.put(b"data", content_type="video/mp4", tags=["render"])
    service = RenderService(artifact_root=str(tmp_path / "artifacts"))
    path = service._resolve(artifact_id)
    assert os.path.exists(path)
