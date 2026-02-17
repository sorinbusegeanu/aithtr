"""Assets stage helpers: portraits, idle videos, and scene backgrounds."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import cv2

from .comfyui_client import ComfyUIClient, ComfyUIConfig
from mcp_servers.assets.artifact_store import ArtifactStore


def ensure_character_portrait(role: Dict[str, Any], episode_id: str, comfy_cfg: ComfyUIConfig) -> Tuple[str, Optional[str]]:
    character_id = str(role.get("character_id") or "").strip()
    if not character_id:
        raise RuntimeError("ensure_character_portrait: missing character_id")

    existing = str(role.get("avatar_image_path") or "").strip()
    if existing and os.path.exists(existing):
        return existing, None

    prompt_text = str(role.get("prompt") or "").strip() or (
        f"cinematic portrait of {character_id}, centered face, neutral expression, detailed skin, studio light"
    )
    negative_text = getattr(
        comfy_cfg,
        "negative_text_default",
        "blurry, low quality, deformed, extra limbs, watermark, text",
    )
    seed = _deterministic_seed(f"{episode_id}:{character_id}")
    filename_prefix = f"portraits/{character_id}"
    client = ComfyUIClient(config=comfy_cfg, episode_id=episode_id, character_id=character_id)
    return client.run_portrait(
        prompt_text=prompt_text,
        negative_text=negative_text,
        seed=seed,
        filename_prefix=filename_prefix,
    )


def ensure_character_idle_video(
    character_id: str,
    portrait_path: str,
    out_mp4_path: str,
    *,
    episode_id: str,
    comfy_cfg: ComfyUIConfig,
) -> Tuple[str, Optional[str]]:
    if not character_id:
        raise RuntimeError("ensure_character_idle_video: missing character_id")
    if not portrait_path or not os.path.exists(portrait_path):
        raise RuntimeError(f"ensure_character_idle_video: portrait missing for '{character_id}': {portrait_path}")
    if bool(getattr(comfy_cfg, "idle_avatar_enabled", False)):
        seed = _deterministic_seed(f"{episode_id}:{character_id}:idle")
        client = ComfyUIClient(config=comfy_cfg, episode_id=episode_id, character_id=character_id)
        idle_path, prompt_id = client.run_idle_avatar_video(
            episode_id=episode_id,
            character_id=character_id,
            portrait_png_path=portrait_path,
            seed=seed,
        )
        return idle_path, prompt_id

    os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
    fps = max(int(os.getenv("ASSETS_SOURCE_FPS", "25")), 1)
    duration_sec = float(os.getenv("ASSETS_SOURCE_DURATION_SEC", "4.0"))
    duration_sec = min(max(duration_sec, 3.0), 6.0)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop",
        "1",
        "-i",
        portrait_path,
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        f"fps={fps},scale=720:720:force_original_aspect_ratio=increase,crop=720:720,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_mp4_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_mp4_path, None
    except Exception:
        allow_cv_fallback = os.getenv("ASSETS_ALLOW_OPENCV_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not allow_cv_fallback:
            _raise_stage_validation_error(
                f"assets idle video generation failed for character '{character_id}' and OpenCV fallback is disabled"
            )
        # Small OpenCV fallback for local-only environments.
        img = cv2.imread(portrait_path)
        if img is None:
            raise RuntimeError(f"ensure_character_idle_video: failed to decode portrait image: {portrait_path}")
        frame = cv2.resize(img, (720, 720))
        writer = cv2.VideoWriter(out_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (720, 720))
        if not writer.isOpened():
            raise RuntimeError(f"ensure_character_idle_video: failed to create video writer: {out_mp4_path}")
        try:
            frame_count = max(1, int(round(duration_sec * fps)))
            for _ in range(frame_count):
                writer.write(frame)
        finally:
            writer.release()
        return out_mp4_path, None


def ensure_scene_background(scene: Dict[str, Any], episode_id: str, comfy_cfg: ComfyUIConfig) -> Tuple[str, Optional[str]]:
    scene_id = str(scene.get("scene_id") or "").strip()
    if not scene_id:
        raise RuntimeError("ensure_scene_background: missing scene_id")
    existing = str(scene.get("background_path") or "").strip()
    if existing and os.path.exists(existing):
        return existing, None

    data_root = os.getenv("DATA_ROOT", "data")
    out_dir = os.path.join(data_root, "runs", episode_id, "assets", "backgrounds")
    os.makedirs(out_dir, exist_ok=True)
    if bool(getattr(comfy_cfg, "backgrounds_enabled", True)):
        prompt_text = str(scene.get("background_prompt") or "").strip() or (
            f"cinematic wide background for {scene_id}, detailed environment, no people"
        )
        negative_text = getattr(
            comfy_cfg,
            "negative_text_default",
            "blurry, low quality, deformed, extra limbs, watermark, text, people",
        )
        seed = _deterministic_seed(f"{episode_id}:{scene_id}:background")
        client = ComfyUIClient(config=comfy_cfg)
        try:
            return client.run_background(
                episode_id=episode_id,
                scene_id=scene_id,
                prompt_text=prompt_text,
                negative_text=negative_text,
                seed=seed,
            )
        except Exception as err:
            _raise_stage_validation_error(
                f"assets background generation failed for scene '{scene_id}': {err}"
            )

    _raise_stage_validation_error(
        f"assets background generation disabled or workflow missing for scene '{scene_id}'"
    )


def build_assets(
    episode_id: str,
    cast_plan: Dict[str, Any],
    screenplay: Dict[str, Any],
    scene_plan: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    del screenplay, scene_plan
    comfy_cfg = _coerce_comfy_cfg(cfg or {})
    data_root = os.getenv("DATA_ROOT", "data")
    store = ArtifactStore()

    cast_out = deepcopy(cast_plan if isinstance(cast_plan, dict) else {})
    scene_out = deepcopy(cfg.get("scene_assets", {}) if isinstance(cfg.get("scene_assets"), dict) else {})
    comfy_prompts: Dict[str, Dict[str, str]] = {"portraits": {}, "backgrounds": {}, "idle": {}}
    roles = cast_out.get("roles", []) if isinstance(cast_out, dict) else []
    for role in roles:
        if not isinstance(role, dict):
            continue
        character_id = str(role.get("character_id") or "").strip()
        if not character_id:
            raise RuntimeError("build_assets: role missing character_id")
        portrait_path, portrait_prompt_id = ensure_character_portrait(role, episode_id, comfy_cfg)
        role["avatar_image_path"] = portrait_path
        with open(portrait_path, "rb") as f:
            role["avatar_image_artifact_id"] = store.put(
                data=f.read(),
                content_type="image/png",
                tags=["avatar", "portrait", f"character:{character_id}"],
            )
        if portrait_prompt_id:
            comfy_prompts["portraits"][character_id] = portrait_prompt_id
        source_path = os.path.join(
            data_root,
            "runs",
            episode_id,
            "assets",
            "characters",
            character_id,
            "source.mp4",
        )
        source_video_path, idle_prompt_id = ensure_character_idle_video(
            character_id,
            portrait_path,
            source_path,
            episode_id=episode_id,
            comfy_cfg=comfy_cfg,
        )
        role["avatar_source_video_path"] = source_video_path
        with open(source_video_path, "rb") as f:
            role["avatar_source_video_artifact_id"] = store.put(
                data=f.read(),
                content_type="video/mp4",
                tags=["avatar", "source_video", f"character:{character_id}"],
            )
        if idle_prompt_id:
            comfy_prompts["idle"][character_id] = idle_prompt_id

    scenes = scene_out.get("scenes", []) if isinstance(scene_out, dict) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or "").strip()
        background_path, background_prompt_id = ensure_scene_background(scene, episode_id, comfy_cfg)
        scene["background_path"] = background_path
        with open(background_path, "rb") as f:
            background_artifact_id = store.put(
                data=f.read(),
                content_type="image/png",
                tags=["background", f"scene:{scene_id or 'unknown'}"],
            )
        scene["background_artifact_id"] = background_artifact_id
        scene["background_asset_id"] = background_artifact_id
        if scene_id and background_prompt_id:
            comfy_prompts["backgrounds"][scene_id] = background_prompt_id

    return {
        "cast_plan": cast_out,
        "scene_assets": scene_out,
        "comfyui": {"prompts": comfy_prompts},
    }


def _deterministic_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _raise_stage_validation_error(message: str) -> None:
    try:
        from orchestrator.pipeline import StageValidationError  # local import to avoid module cycle at import time
    except Exception:
        raise RuntimeError(message)
    raise StageValidationError(
        {
            "stage": "assets",
            "line_id": None,
            "artifact_path": None,
            "reason": "BACKGROUND_GENERATION_FAILED",
            "message": message,
            "extra": {},
        }
    )


def _coerce_comfy_cfg(cfg: Dict[str, Any]) -> ComfyUIConfig:
    section = cfg.get("comfyui") if isinstance(cfg, dict) else None
    section = section if isinstance(section, dict) else {}
    obj = ComfyUIConfig(
        enabled=bool(section.get("enabled", True)),
        host=str(section.get("host", "192.168.0.51")),
        port=int(section.get("port", 8188)),
        max_wait_sec=int(section.get("max_wait_sec", 300)),
        poll_interval_sec=float(section.get("poll_interval_sec", 1)),
        request_timeout_sec=float(section.get("request_timeout_sec", 10)),
        workflow_portrait_json_path=str(
            section.get("workflow_portrait_json_path", "workflows/comfy/portrait_sdxl_workflow.json")
        ),
        workflow_idle_avatar_json_path=str(
            section.get("workflow_idle_avatar_json_path", "workflows/comfy/idle_avatar_video.json")
        ),
        workflow_background_json_path=str(
            section.get("workflow_background_json_path", "workflows/comfy/background_sdxl.json")
        ),
        idle_avatar_enabled=bool(section.get("idle_avatar_enabled", False)),
        idle_avatar_seconds=int(section.get("idle_avatar_seconds", 4)),
        idle_avatar_fps=int(section.get("idle_avatar_fps", 25)),
        backgrounds_enabled=bool(section.get("backgrounds_enabled", True)),
        background_width=int(section.get("background_width", 1280)),
        background_height=int(section.get("background_height", 720)),
        output_fetch_mode=str(section.get("output_fetch_mode", "http")),
        shared_fs_output_root=str(section.get("shared_fs_output_root", "")),
    )
    setattr(
        obj,
        "negative_text_default",
        str(section.get("negative_text_default", "blurry, low quality, deformed, extra limbs, watermark, text")),
    )
    return obj
