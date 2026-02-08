"""
Minimal Wav2Lip wrapper. Expects an external Wav2Lip install.
Wire into MCP Python SDK in a later phase.
"""
import os
import shutil
import subprocess
import sys
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
            wav2lip_py = (os.getenv("WAV2LIP_PYTHON") or sys.executable).strip()
            wav2lip_script = (os.getenv("WAV2LIP_SCRIPT") or "").strip() or _discover_wav2lip_script()
            wav2lip_ckpt = (os.getenv("WAV2LIP_CHECKPOINT") or "").strip()
            allow_fallback = os.getenv("LIPSYNC_ALLOW_STATIC_FALLBACK", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not wav2lip_script:
                if allow_fallback:
                    return self._render_static_fallback(avatar_id=avatar_id, wav_id=wav_id)
                raise ValueError("WAV2LIP_SCRIPT is required")
            if not wav2lip_ckpt:
                wav2lip_ckpt = _discover_wav2lip_checkpoint(wav2lip_script)
            wav2lip_root = os.path.dirname(wav2lip_script) or "."
            _ensure_s3fd_weight(wav2lip_root)
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
                child_env = os.environ.copy()
                child_env.setdefault("NUMBA_DISABLE_CACHING", "1")
                child_env.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=wav2lip_root,
                    env=child_env,
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
                if allow_fallback:
                    return self._render_static_fallback(avatar_id=avatar_id, wav_id=wav_id)
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

    def _render_static_fallback(self, avatar_id: str, wav_id: str) -> Dict[str, Any]:
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        avatar_path = self.store.get_path(avatar_id)
        wav_path = self.store.get_path(wav_id)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loop",
                "1",
                "-i",
                avatar_path,
                "-i",
                wav_path,
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                out_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            with open(out_path, "rb") as f:
                data = f.read()
            artifact_id = self.store.put(data=data, content_type="video/mp4", tags=["lipsync", "fallback"])
            return {"artifact_id": artifact_id}
        finally:
            if os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except Exception:
                    pass


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _data_root() -> str:
    return os.getenv("DATA_ROOT", os.path.join(_project_root(), "data"))


def _discover_wav2lip_script() -> str:
    candidates = [
        os.path.join(_data_root(), "models", "Wav2Lip", "inference.py"),
        os.path.join(_data_root(), "models", "wav2lip", "inference.py"),
        "/opt/wav2lip/inference.py",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _discover_wav2lip_checkpoint(wav2lip_script: str) -> str:
    root = os.path.dirname(wav2lip_script) or "."
    candidates = [
        os.path.join(root, "checkpoints", "wav2lip_gan.pth"),
        os.path.join(root, "wav2lip_gan.pth"),
        os.path.join(_data_root(), "models", "Wav2Lip", "checkpoints", "wav2lip_gan.pth"),
        os.path.join(_data_root(), "models", "wav2lip", "wav2lip_gan.pth"),
        os.path.join(_data_root(), "models", "wav2lip_old", "wav2lip_gan.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _ensure_s3fd_weight(wav2lip_root: str) -> None:
    target = os.path.join(wav2lip_root, "face_detection", "detection", "sfd", "s3fd.pth")
    if os.path.exists(target):
        return
    candidates = [
        os.getenv("WAV2LIP_S3FD", "").strip(),
        os.path.join(_data_root(), "models", "wav2lip_old", "s3fd-619a316812.pth"),
        os.path.join(_data_root(), "models", "Wav2Lip", "checkpoints", "s3fd.pth"),
    ]
    source = next((p for p in candidates if p and os.path.exists(p)), "")
    if not source:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)
