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
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
import soundfile as sf

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
        out_path = None
        diagnostics: Dict[str, Any] = {
            "engine": "wav2lip",
            "avatar_id": avatar_id,
            "wav_id": wav_id,
            "style": style or {},
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
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
                    return self._render_static_fallback(
                        avatar_id=avatar_id,
                        wav_id=wav_id,
                        diagnostics=diagnostics,
                    )
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
                # Artifact paths are digest-only (no extension). Re-materialize with a suffix
                # so Wav2Lip can treat it as either an image or a video source.
                suffix = ".png"
                content_type = ""
                try:
                    meta = self.store.get_metadata(avatar_id)
                    content_type = (meta.content_type or "").lower()
                except Exception:
                    content_type = ""

                header = b""
                try:
                    with open(avatar_path, "rb") as src:
                        header = src.read(16)
                except Exception:
                    header = b""

                is_jpeg = header.startswith(b"\xff\xd8\xff") or "jpeg" in content_type or "jpg" in content_type
                is_png = header.startswith(b"\x89PNG\r\n\x1a\n") or "png" in content_type
                is_video = (
                    "video/" in content_type
                    or header[4:8] == b"ftyp"
                    or header.startswith(b"\x1aE\xdf\xa3")  # Matroska/WebM
                )

                if is_video:
                    suffix = ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_face:
                        shutil.copyfile(avatar_path, tmp_face.name)
                        temp_face_path = tmp_face.name
                    face_path = temp_face_path
                else:
                    if is_jpeg:
                        suffix = ".jpg"
                    elif is_png:
                        suffix = ".png"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_face:
                        img = cv2.imread(avatar_path)
                        if img is None:
                            raise ValueError(f"avatar image unreadable: {avatar_path}")
                        if not cv2.imwrite(tmp_face.name, img):
                            raise ValueError(f"avatar image re-encode failed: {avatar_path}")
                        temp_face_path = tmp_face.name
                    face_path = temp_face_path
            wav_path = self.store.get_path(wav_id)
            diagnostics["audio"] = _audio_stats(wav_path)
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
            diagnostics["render_command"] = cmd
            diagnostics["checkpoint_path"] = wav2lip_ckpt
            diagnostics["wav2lip_script"] = wav2lip_script
            try:
                child_env = os.environ.copy()
                child_env.setdefault("NUMBA_DISABLE_CACHING", "1")
                child_env.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
                proc = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=wav2lip_root,
                    env=child_env,
                )
                diagnostics["stdout_tail"] = (proc.stdout or "")[-4000:]
                diagnostics["stderr_tail"] = (proc.stderr or "")[-4000:]
            except Exception as exc:
                print("[lipsync] Wav2Lip failed", flush=True)
                print(f"[lipsync] cmd: {' '.join(cmd)}", flush=True)
                if isinstance(exc, subprocess.CalledProcessError):
                    diagnostics["subprocess_returncode"] = exc.returncode
                    diagnostics["stdout_tail"] = (exc.stdout or "")[-4000:]
                    diagnostics["stderr_tail"] = (exc.stderr or "")[-4000:]
                    if exc.stdout:
                        print(f"[lipsync] stdout:\n{exc.stdout}", flush=True)
                    if exc.stderr:
                        print(f"[lipsync] stderr:\n{exc.stderr}", flush=True)
                else:
                    print(traceback.format_exc(), flush=True)
                if allow_fallback:
                    diagnostics["fallback_reason"] = _format_exception(exc)
                    return self._render_static_fallback(
                        avatar_id=avatar_id,
                        wav_id=wav_id,
                        diagnostics=diagnostics,
                    )
                raise
            diagnostics["video_pre_split"] = _video_motion_stats(out_path)
            diagnostics["motion_signal"] = {
                # Wav2Lip does not currently expose viseme/blendshape internals.
                "type": "frame_diff_proxy",
                "mean_diff": diagnostics["video_pre_split"].get("mean_frame_diff", 0.0),
                "nonzero_ratio": diagnostics["video_pre_split"].get("nonzero_frame_diff_ratio", 0.0),
                "has_motion": diagnostics["video_pre_split"].get("mean_frame_diff", 0.0) > 1e-6,
                "landmark_tracking_ok": None,
                "blendshape_nonzero_ratio": None,
                "viseme_nonzero_ratio": None,
            }
            if os.getenv("LIPSYNC_DEBUG_ASSERT_MOTION", "0").strip().lower() in {"1", "true", "yes", "on"}:
                debug_min = float(os.getenv("LIPSYNC_DEBUG_MIN_MOTION_DIFF", "0.001"))
                if float(diagnostics["video_pre_split"].get("mean_frame_diff", 0.0)) < debug_min:
                    raise RuntimeError(
                        f"Lipsync debug assert failed: pre-split mean diff < {debug_min} "
                        f"({diagnostics['video_pre_split'].get('mean_frame_diff', 0.0):.6f})"
                    )
            _write_debug_artifacts(style=style, out_path=out_path, wav_path=wav_path, diagnostics=diagnostics)
            with open(out_path, "rb") as f:
                data = f.read()
            artifact_id = self.store.put(data=data, content_type="video/mp4", tags=["lipsync"])
            return {"artifact_id": artifact_id, "diagnostics": diagnostics}
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
            if out_path and os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except Exception:
                    pass

    def _render_static_fallback(
        self,
        avatar_id: str,
        wav_id: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            payload = diagnostics or {}
            payload["engine"] = "static_fallback"
            payload["audio"] = _audio_stats(wav_path)
            payload["video_pre_split"] = _video_motion_stats(out_path)
            return {"artifact_id": artifact_id, "diagnostics": payload}
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


def _audio_stats(path: str) -> Dict[str, Any]:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    duration = float(arr.shape[0]) / float(sr or 1)
    if arr.size:
        abs_arr = np.abs(arr)
        rms = np.sqrt(np.mean(np.square(arr)))
        return {
            "sample_rate": int(sr),
            "duration_sec": duration,
            "rms_min": float(np.min(abs_arr)),
            "rms_mean": float(rms),
            "rms_max": float(np.max(abs_arr)),
            "peak_abs": float(np.max(abs_arr)),
            "near_silent": bool(rms < float(os.getenv("TTS_MIN_RMS", "0.005"))),
        }
    return {
        "sample_rate": int(sr),
        "duration_sec": duration,
        "rms_min": 0.0,
        "rms_mean": 0.0,
        "rms_max": 0.0,
        "peak_abs": 0.0,
        "near_silent": True,
    }


def _video_motion_stats(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, prev = cap.read()
    diffs: list[float] = []
    while ok:
        ok2, cur = cap.read()
        if not ok2:
            break
        diffs.append(float(cv2.absdiff(cur, prev).mean()))
        prev = cur
    cap.release()
    duration = (float(frame_count) / fps) if fps > 0 else 0.0
    if diffs:
        arr = np.asarray(diffs, dtype=np.float32)
        nonzero_ratio = float(np.mean(arr > 1e-6))
        return {
            "frame_count": frame_count,
            "fps": fps,
            "duration_sec": duration,
            "width": width,
            "height": height,
            "mean_frame_diff": float(arr.mean()),
            "min_frame_diff": float(arr.min()),
            "max_frame_diff": float(arr.max()),
            "nonzero_frame_diff_ratio": nonzero_ratio,
        }
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "mean_frame_diff": 0.0,
        "min_frame_diff": 0.0,
        "max_frame_diff": 0.0,
        "nonzero_frame_diff_ratio": 0.0,
    }


def _write_debug_artifacts(
    *,
    style: Optional[Dict[str, Any]],
    out_path: str,
    wav_path: str,
    diagnostics: Dict[str, Any],
) -> None:
    debug_dir = ""
    if isinstance(style, dict):
        debug_dir = str(style.get("debug_dir") or "").strip()
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        shutil.copy2(out_path, os.path.join(debug_dir, "rendered.mp4"))
        shutil.copy2(wav_path, os.path.join(debug_dir, "input.wav"))
        with open(os.path.join(debug_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
            import json

            json.dump(diagnostics, f, ensure_ascii=True, indent=2)
    except Exception:
        # Diagnostics export must never alter render behavior.
        pass


def _format_exception(err: Exception) -> str:
    if isinstance(err, subprocess.CalledProcessError):
        return f"CalledProcessError(returncode={err.returncode})"
    return f"{type(err).__name__}: {err}"
