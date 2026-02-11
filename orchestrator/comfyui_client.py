"""Minimal ComfyUI HTTP client."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ComfyUIConfig:
    enabled: bool = True
    host: str = "192.168.0.51"
    port: int = 8188
    max_wait_sec: int = 300
    poll_interval_sec: float = 1.0
    request_timeout_sec: float = 10.0
    workflow_portrait_json_path: str = "workflows/comfy/portrait_sdxl_workflow.json"
    workflow_idle_avatar_json_path: str = "workflows/comfy/idle_avatar_video.json"
    workflow_background_json_path: str = "workflows/comfy/background_sdxl.json"
    idle_avatar_enabled: bool = False
    idle_avatar_seconds: int = 4
    idle_avatar_fps: int = 25
    backgrounds_enabled: bool = True
    background_width: int = 1280
    background_height: int = 720
    output_fetch_mode: str = "http"
    shared_fs_output_root: str = ""


def _load_default_config() -> ComfyUIConfig:
    # Keep this client self-contained. If YAML parsing is unavailable, use defaults.
    cfg = ComfyUIConfig()
    path = os.path.join("config", "defaults.yaml")
    if not os.path.exists(path):
        return cfg
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        section = payload.get("comfyui", {}) if isinstance(payload, dict) else {}
        if isinstance(section, dict):
            cfg.enabled = bool(section.get("enabled", cfg.enabled))
            cfg.host = str(section.get("host", cfg.host))
            cfg.port = int(section.get("port", cfg.port))
            cfg.max_wait_sec = int(section.get("max_wait_sec", cfg.max_wait_sec))
            cfg.poll_interval_sec = float(section.get("poll_interval_sec", cfg.poll_interval_sec))
            cfg.request_timeout_sec = float(section.get("request_timeout_sec", cfg.request_timeout_sec))
            cfg.workflow_portrait_json_path = str(
                section.get("workflow_portrait_json_path", cfg.workflow_portrait_json_path)
            )
            cfg.workflow_idle_avatar_json_path = str(
                section.get("workflow_idle_avatar_json_path", cfg.workflow_idle_avatar_json_path)
            )
            cfg.workflow_background_json_path = str(
                section.get("workflow_background_json_path", cfg.workflow_background_json_path)
            )
            cfg.idle_avatar_enabled = bool(section.get("idle_avatar_enabled", cfg.idle_avatar_enabled))
            cfg.idle_avatar_seconds = int(section.get("idle_avatar_seconds", cfg.idle_avatar_seconds))
            cfg.idle_avatar_fps = int(section.get("idle_avatar_fps", cfg.idle_avatar_fps))
            cfg.backgrounds_enabled = bool(section.get("backgrounds_enabled", cfg.backgrounds_enabled))
            cfg.background_width = int(section.get("background_width", cfg.background_width))
            cfg.background_height = int(section.get("background_height", cfg.background_height))
            cfg.output_fetch_mode = str(section.get("output_fetch_mode", cfg.output_fetch_mode))
            cfg.shared_fs_output_root = str(section.get("shared_fs_output_root", cfg.shared_fs_output_root))
    except Exception:
        return cfg
    return cfg


class ComfyUIClient:
    def __init__(
        self,
        config: Optional[ComfyUIConfig] = None,
        *,
        episode_id: Optional[str] = None,
        character_id: Optional[str] = None,
        data_root: Optional[str] = None,
    ) -> None:
        self.config = config or _load_default_config()
        self.episode_id = (episode_id or "").strip()
        self.character_id = (character_id or "").strip()
        self.data_root = (data_root or os.getenv("DATA_ROOT", "data")).strip()
        self.base_url = f"http://{self.config.host}:{self.config.port}"

    def submit_workflow(self, workflow_dict: Dict[str, Any]) -> str:
        """POST /prompt and return prompt_id."""
        url = f"{self.base_url}/prompt"
        body = json.dumps({"prompt": workflow_dict}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        prompt_id = str(payload.get("prompt_id") or "").strip()
        if not prompt_id:
            raise RuntimeError("ComfyUI submit failed: missing prompt_id")
        return prompt_id

    def wait_for_history(self, prompt_id: str) -> Dict[str, Any]:
        """Poll GET /history/{prompt_id} until outputs exist or timeout."""
        if not prompt_id:
            raise RuntimeError("ComfyUI wait failed: empty prompt_id")
        url = f"{self.base_url}/history/{urllib.parse.quote(prompt_id)}"
        deadline = time.time() + float(self.config.max_wait_sec)
        last_payload: Dict[str, Any] = {}
        while time.time() < deadline:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                last_payload = payload
                history = payload.get(prompt_id, payload)
                if isinstance(history, dict):
                    outputs = history.get("outputs")
                    if isinstance(outputs, dict) and outputs:
                        return history
            time.sleep(float(self.config.poll_interval_sec))
        raise TimeoutError(
            f"ComfyUI history timeout after {self.config.max_wait_sec}s for prompt_id={prompt_id}; "
            f"last_payload_keys={list(last_payload.keys()) if isinstance(last_payload, dict) else []}"
        )

    def download_output_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """GET /view and return raw bytes."""
        if not filename:
            raise RuntimeError("ComfyUI download failed: empty filename")
        query = urllib.parse.urlencode(
            {
                "filename": filename,
                "subfolder": subfolder or "",
                "type": folder_type or "",
            }
        )
        url = f"{self.base_url}/view?{query}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError(f"ComfyUI download failed: empty response for filename={filename}")
        return data

    def run_portrait(
        self,
        prompt_text: str,
        negative_text: str,
        seed: int,
        filename_prefix: str,
    ) -> tuple[str, str]:
        """
        Run portrait workflow and write result to:
        data/runs/<episode_id>/assets/characters/<character_id>/portrait.png
        """
        workflow_path = self.config.workflow_portrait_json_path
        if not os.path.exists(workflow_path):
            raise RuntimeError(f"ComfyUI workflow template missing: {workflow_path}")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise RuntimeError("ComfyUI workflow template must be a JSON object")

        # Patch required nodes.
        pos_node = _find_first_node_by_class_type(workflow, "CLIPTextEncode", occurrence=1)
        neg_node = _find_first_node_by_class_type(workflow, "CLIPTextEncode", occurrence=2)
        ks_node = _find_first_node_by_class_type(workflow, "KSampler", occurrence=1)
        save_node = _find_first_node_by_class_type(workflow, "SaveImage", occurrence=1)
        if pos_node is None or neg_node is None or ks_node is None or save_node is None:
            raise RuntimeError(
                "ComfyUI workflow patch failed: required nodes not found "
                "(CLIPTextEncode x2, KSampler, SaveImage)"
            )

        _patch_node_input(pos_node, "text", prompt_text)
        _patch_node_input(neg_node, "text", negative_text)
        _patch_node_input(ks_node, "seed", int(seed))
        _patch_node_input(save_node, "filename_prefix", filename_prefix)

        prompt_id = self.submit_workflow(workflow)
        history = self.wait_for_history(prompt_id)
        image_meta = _extract_first_image_output(history)
        if image_meta is None:
            raise RuntimeError(f"ComfyUI history has no image outputs for prompt_id={prompt_id}")

        content = self.download_output_file(
            filename=str(image_meta.get("filename") or ""),
            subfolder=str(image_meta.get("subfolder") or ""),
            folder_type=str(image_meta.get("type") or ""),
        )

        if not self.episode_id or not self.character_id:
            raise RuntimeError(
                "ComfyUI run_portrait requires episode_id and character_id set on client "
                "to write portrait.png path"
            )
        out_dir = os.path.join(
            self.data_root,
            "runs",
            self.episode_id,
            "assets",
            "characters",
            self.character_id,
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "portrait.png")
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path, prompt_id

    def run_background(
        self,
        episode_id: str,
        scene_id: str,
        prompt_text: str,
        negative_text: str,
        seed: int,
    ) -> tuple[str, str]:
        workflow_path = self.config.workflow_background_json_path
        if not os.path.exists(workflow_path):
            raise RuntimeError(f"ComfyUI background workflow template missing: {workflow_path}")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise RuntimeError("ComfyUI background workflow template must be a JSON object")

        pos_node = _find_first_node_by_class_type(workflow, "CLIPTextEncode", occurrence=1)
        neg_node = _find_first_node_by_class_type(workflow, "CLIPTextEncode", occurrence=2)
        latent_node = _find_first_node_by_class_type(workflow, "EmptyLatentImage", occurrence=1)
        ks_node = _find_first_node_by_class_type(workflow, "KSampler", occurrence=1)
        save_node = _find_first_node_by_class_type(workflow, "SaveImage", occurrence=1)
        if pos_node is None or neg_node is None or latent_node is None or ks_node is None or save_node is None:
            raise RuntimeError(
                "ComfyUI background workflow patch failed: required nodes not found "
                "(CLIPTextEncode x2, EmptyLatentImage, KSampler, SaveImage)"
            )
        _patch_node_input(pos_node, "text", prompt_text)
        _patch_node_input(neg_node, "text", negative_text)
        _patch_node_input(latent_node, "width", int(self.config.background_width))
        _patch_node_input(latent_node, "height", int(self.config.background_height))
        _patch_node_input(ks_node, "seed", int(seed))
        _patch_node_input(save_node, "filename_prefix", f"backgrounds/{scene_id}")

        prompt_id = self.submit_workflow(workflow)
        history = self.wait_for_history(prompt_id)
        image_meta = _extract_first_image_output(history)
        if image_meta is None:
            raise RuntimeError(f"ComfyUI background history has no image outputs for prompt_id={prompt_id}")
        content = self.download_output_file(
            filename=str(image_meta.get("filename") or ""),
            subfolder=str(image_meta.get("subfolder") or ""),
            folder_type=str(image_meta.get("type") or ""),
        )
        out_dir = os.path.join(self.data_root, "runs", episode_id, "assets", "backgrounds")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{scene_id}.png")
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path, prompt_id

    def run_idle_avatar_video(
        self,
        episode_id: str,
        character_id: str,
        portrait_png_path: str,
        seed: int,
    ) -> tuple[str, str]:
        if not os.path.exists(portrait_png_path):
            raise RuntimeError(f"ComfyUI idle avatar input portrait missing: {portrait_png_path}")
        workflow_path = self.config.workflow_idle_avatar_json_path
        if not os.path.exists(workflow_path):
            raise RuntimeError(f"ComfyUI idle avatar workflow template missing: {workflow_path}")

        upload_url = f"{self.base_url}/upload/image"
        with open(portrait_png_path, "rb") as f:
            image_bytes = f.read()
        boundary = f"----codex-{int(time.time() * 1000)}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(portrait_png_path)}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            upload_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
            upload_payload = json.loads(resp.read().decode("utf-8"))
        uploaded_name = str(upload_payload.get("name") or upload_payload.get("filename") or "").strip()
        if not uploaded_name:
            raise RuntimeError("ComfyUI idle avatar upload failed: missing uploaded filename")

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise RuntimeError("ComfyUI idle avatar workflow template must be a JSON object")

        image_loader = _find_node_with_input(workflow, "image")
        save_video = _find_first_node_by_class_type(workflow, "SaveVideo", occurrence=1)
        save_image = _find_first_node_by_class_type(workflow, "SaveImage", occurrence=1)
        sampler = _find_first_node_by_class_type(workflow, "KSampler", occurrence=1)
        if image_loader is None:
            raise RuntimeError("ComfyUI idle avatar workflow patch failed: image loader node not found")
        _patch_node_input(image_loader, "image", uploaded_name)
        if sampler is not None:
            _patch_node_input(sampler, "seed", int(seed))
        if save_video is not None:
            _patch_node_input(save_video, "filename_prefix", f"idle/{character_id}")
            _patch_node_input(save_video, "fps", int(self.config.idle_avatar_fps))
            _patch_node_input(save_video, "seconds", int(self.config.idle_avatar_seconds))
        elif save_image is not None:
            _patch_node_input(save_image, "filename_prefix", f"idle_frames/{character_id}")
        else:
            raise RuntimeError("ComfyUI idle avatar workflow patch failed: no SaveVideo/SaveImage node found")

        prompt_id = self.submit_workflow(workflow)
        history = self.wait_for_history(prompt_id)
        out_dir = os.path.join(self.data_root, "runs", episode_id, "assets", "characters", character_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "source.mp4")

        video_meta = _extract_first_video_output(history)
        if video_meta is not None:
            content = self.download_output_file(
                filename=str(video_meta.get("filename") or ""),
                subfolder=str(video_meta.get("subfolder") or ""),
                folder_type=str(video_meta.get("type") or ""),
            )
            with open(out_path, "wb") as f:
                f.write(content)
            return out_path, prompt_id

        images_meta = _extract_image_outputs(history)
        if not images_meta:
            raise RuntimeError(f"ComfyUI idle avatar history has no video/images outputs for prompt_id={prompt_id}")
        frames_dir = os.path.join(out_dir, "idle_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for idx, item in enumerate(images_meta):
            content = self.download_output_file(
                filename=str(item.get("filename") or ""),
                subfolder=str(item.get("subfolder") or ""),
                folder_type=str(item.get("type") or ""),
            )
            with open(os.path.join(frames_dir, f"{idx:06d}.png"), "wb") as f:
                f.write(content)
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(int(self.config.idle_avatar_fps)),
            "-i",
            os.path.join(frames_dir, "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path, prompt_id


def _find_first_node_by_class_type(
    workflow: Dict[str, Any],
    class_type: str,
    *,
    occurrence: int,
) -> Optional[Dict[str, Any]]:
    count = 0
    for _, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") == class_type:
            count += 1
            if count == occurrence:
                return node
    return None


def _patch_node_input(node: Dict[str, Any], key: str, value: Any) -> None:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"ComfyUI workflow patch failed: node missing inputs for key={key}")
    inputs[key] = value


def _extract_first_image_output(history: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None
    for _, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images")
        if not isinstance(images, list) or not images:
            continue
        first = images[0]
        if isinstance(first, dict) and first.get("filename"):
            return first
    return None


def _extract_image_outputs(history: Dict[str, Any]) -> list[Dict[str, Any]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return []
    items: list[Dict[str, Any]] = []
    for _, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if isinstance(image, dict) and image.get("filename"):
                items.append(image)
    return items


def _extract_first_video_output(history: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None
    for _, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        videos = node_out.get("videos")
        if isinstance(videos, list) and videos:
            first = videos[0]
            if isinstance(first, dict) and first.get("filename"):
                return first
    return None


def _find_node_with_input(workflow: Dict[str, Any], input_key: str) -> Optional[Dict[str, Any]]:
    for _, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and input_key in inputs:
            return node
    return None
