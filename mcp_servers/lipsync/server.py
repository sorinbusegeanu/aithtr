"""
Minimal Wav2Lip wrapper. Expects an external Wav2Lip install.
Wire into MCP Python SDK in a later phase.
"""
import os
import shutil
import subprocess
import tempfile
import traceback
from typing import Any, Dict, Optional

import cv2

from mcp_servers.assets.artifact_store import ArtifactStore


class LipSyncService:
    def __init__(self, artifact_root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=artifact_root)

    def lipsync_render_clip(
        self,
        avatar_id: str,
        wav_id: str,
        style: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        temp_face_path = None
        try:
            wav2lip_py = os.getenv("WAV2LIP_PYTHON", "python")
            wav2lip_script = os.getenv("WAV2LIP_SCRIPT", "")
            wav2lip_ckpt = os.getenv("WAV2LIP_CHECKPOINT", "")
            if not wav2lip_script:
                raise ValueError("WAV2LIP_SCRIPT is required")
            wav2lip_root = os.path.dirname(wav2lip_script) or "."
            os.makedirs(os.path.join(wav2lip_root, "temp"), exist_ok=True)
            avatar_path = self.store.get_path(avatar_id)
            face_path = avatar_path
            _, ext = os.path.splitext(avatar_path)
            if not ext:
                suffix = ".png"
                try:
                    with open(avatar_path, "rb") as src:
                        header = src.read(8)
                    if header.startswith(b"\x89PNG\r\n\x1a\n"):
                        suffix = ".png"
                    elif header.startswith(b"\xff\xd8\xff"):
                        suffix = ".jpg"
                    else:
                        meta = self.store.get_metadata(avatar_id)
                        content_type = (meta.content_type or "").lower()
                        if "jpeg" in content_type or "jpg" in content_type:
                            suffix = ".jpg"
                except Exception:
                    pass
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_face:
                    img = cv2.imread(avatar_path)
                    if img is None:
                        raise ValueError(f"avatar image unreadable: {avatar_path}")
                    if not cv2.imwrite(tmp_face.name, img):
                        raise ValueError(f"avatar image re-encode failed: {avatar_path}")
                    temp_face_path = tmp_face.name
                face_path = temp_face_path
            wav_path = self.store.get_path(wav_id)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                out_path = tmp.name
            cmd = [
                wav2lip_py,
                wav2lip_script,
                "--face",
                face_path,
                "--audio",
                wav_path,
                "--outfile",
                out_path,
            ]
            if wav2lip_ckpt:
                cmd += ["--checkpoint_path", wav2lip_ckpt]
            if style and "pads" in style:
                cmd += ["--pads", str(style["pads"])]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=wav2lip_root,
                )
            except Exception as exc:
                print("[lipsync] Wav2Lip failed", flush=True)
                print(f"[lipsync] cmd: {' '.join(cmd)}", flush=True)
                if isinstance(exc, subprocess.CalledProcessError):
                    if exc.stdout:
                        print(f"[lipsync] stdout:\n{exc.stdout}", flush=True)
                    if exc.stderr:
                        print(f"[lipsync] stderr:\n{exc.stderr}", flush=True)
                else:
                    print(traceback.format_exc(), flush=True)
                raise
            with open(out_path, "rb") as f:
                data = f.read()
            artifact_id = self.store.put(data=data, content_type="video/mp4", tags=["lipsync"])
            return {"artifact_id": artifact_id}
        except Exception:
            print("[lipsync] lipsync_render_clip failed", flush=True)
            print(
                f"[lipsync] avatar_id={avatar_id} wav_id={wav_id} "
                f"WAV2LIP_SCRIPT={os.getenv('WAV2LIP_SCRIPT','')} "
                f"WAV2LIP_CHECKPOINT={os.getenv('WAV2LIP_CHECKPOINT','')}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            raise
        finally:
            if temp_face_path and os.path.exists(temp_face_path):
                try:
                    os.unlink(temp_face_path)
                except Exception:
                    pass
