"""
FFmpeg timeline renderer.
"""
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcp_servers.assets.artifact_store import ArtifactStore


@dataclass
class RenderSettings:
    width: int
    height: int
    fps: int


class RenderService:
    def __init__(self, artifact_root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=artifact_root)

    def render_preview(self, timeline_uri: str) -> Dict[str, Any]:
        return self._render(timeline_uri=timeline_uri, preset="preview")

    def render_final(self, timeline_uri: str) -> Dict[str, Any]:
        return self._render(timeline_uri=timeline_uri, preset="final")

    def _render(self, timeline_uri: str, preset: str) -> Dict[str, Any]:
        timeline_path = self._resolve(timeline_uri)
        with open(timeline_path, "r", encoding="utf-8") as f:
            timeline = json.load(f)

        settings = self._settings(preset)
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_outputs: List[str] = []
            for idx, scene in enumerate(timeline.get("scenes", [])):
                scene_path = os.path.join(tmpdir, f"scene_{idx:03d}.mp4")
                self._render_scene(scene=scene, out_path=scene_path, settings=settings)
                scene_outputs.append(scene_path)

            if not scene_outputs:
                raise ValueError("timeline has no scenes")

            out_path = os.path.join(tmpdir, "episode.mp4")
            self._concat_scenes(scene_outputs, out_path)
            with open(out_path, "rb") as f:
                data = f.read()

        artifact_id = self.store.put(data=data, content_type="video/mp4", tags=["render", preset])
        return {"artifact_id": artifact_id, "preset": preset}

    def _render_scene(self, scene: Dict[str, Any], out_path: str, settings: RenderSettings) -> None:
        start = float(scene.get("start_sec", 0.0))
        end = float(scene.get("end_sec", 0.0))
        duration = max(0.0, end - start)
        if duration <= 0:
            raise ValueError("scene duration must be > 0")

        layers = scene.get("layers", [])
        audio = scene.get("audio", [])

        background = [l for l in layers if l.get("type") == "background"]
        if not background:
            raise ValueError("scene missing background layer")
        bg_layer = background[0]

        inputs: List[str] = []
        input_kinds: List[str] = []
        filters: List[str] = []

        bg_path = self._resolve(bg_layer["asset_id"])
        bg_is_image = _is_image(bg_path)
        if bg_is_image:
            inputs += ["-loop", "1", "-t", _fmt_time(duration), "-i", bg_path]
        else:
            inputs += ["-i", bg_path]
        input_kinds.append("bg")

        overlay_layers = [l for l in layers if l.get("type") in {"actor", "props"}]
        subtitle_layers = [l for l in layers if l.get("type") == "caption"]

        for layer in overlay_layers:
            asset_path = self._resolve(layer["asset_id"])
            if _is_image(asset_path):
                inputs += ["-loop", "1", "-t", _fmt_time(duration), "-i", asset_path]
            else:
                inputs += ["-i", asset_path]
            input_kinds.append("overlay")

        for layer in audio:
            asset_path = self._resolve(layer["asset_id"])
            inputs += ["-i", asset_path]
            input_kinds.append("audio")

        # Base background
        filters.append(
            f"[0:v]scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease,"
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2,"
            "format=rgba[base]"
        )

        current = "base"
        video_input_index = 1
        for layer in overlay_layers:
            local_start = max(0.0, float(layer.get("start_sec", start)) - start)
            local_end = min(duration, float(layer.get("end_sec", end)) - start)
            if local_end <= 0 or local_start >= duration or local_end <= local_start:
                video_input_index += 1
                continue

            position = layer.get("position") or {}
            x = float(position.get("x", 0.0)) * settings.width
            y = float(position.get("y", 0.0)) * settings.height
            scale = float(layer.get("scale", 1.0))

            filters.append(
                f"[{video_input_index}:v]trim=0:{_fmt_time(duration)},setpts=PTS-STARTPTS,"
                f"scale=iw*{scale}:ih*{scale}:flags=bicubic,format=rgba[ov{video_input_index}]"
            )
            filters.append(
                f"[{current}][ov{video_input_index}]overlay={_fmt_number(x)}:{_fmt_number(y)}:"
                f"enable='between(t,{_fmt_time(local_start)},{_fmt_time(local_end)})'[v{video_input_index}]"
            )
            current = f"v{video_input_index}"
            video_input_index += 1

        subtitle_filter = ""
        for layer in subtitle_layers:
            sub_path = self._resolve(layer["asset_id"])
            if _is_subtitle(sub_path):
                subtitle_filter = f"subtitles='{sub_path}'"
                break

        if subtitle_filter:
            filters.append(f"[{current}]{subtitle_filter}[vout]")
        else:
            filters.append(f"[{current}]format=yuv420p[vout]")

        audio_filters: List[str] = []
        audio_maps: List[str] = []
        audio_index = 1 + len(overlay_layers)
        for layer in audio:
            local_start = max(0.0, float(layer.get("start_sec", start)) - start)
            local_end = min(duration, float(layer.get("end_sec", end)) - start)
            if local_end <= 0 or local_start >= duration or local_end <= local_start:
                audio_index += 1
                continue
            delay_ms = int(math.floor(local_start * 1000))
            audio_filters.append(
                f"[{audio_index}:a]atrim=0:{_fmt_time(local_end - local_start)},"
                f"asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms}[a{audio_index}]"
            )
            audio_maps.append(f"[a{audio_index}]")
            audio_index += 1

        if audio_maps:
            audio_filters.append(
                f"{''.join(audio_maps)}amix=inputs={len(audio_maps)}:duration=longest[aout]"
            )
        else:
            audio_filters.append("anullsrc=channel_layout=stereo:sample_rate=48000[aout]")

        filter_complex = ";".join(filters + audio_filters)

        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-r",
            str(settings.fps),
            "-t",
            _fmt_time(duration),
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, check=True)

    def _concat_scenes(self, scene_paths: List[str], out_path: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in scene_paths:
                f.write(f"file '{path}'\n")
            list_path = f.name
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        cmd = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
        subprocess.run(cmd, check=True)

    def _resolve(self, uri: str) -> str:
        if os.path.exists(uri):
            return uri
        return self.store.get_path(uri)

    def _settings(self, preset: str) -> RenderSettings:
        width = int(os.getenv("RENDER_WIDTH", "1280"))
        height = int(os.getenv("RENDER_HEIGHT", "720"))
        fps = int(os.getenv("RENDER_FPS", "30"))
        if preset == "preview":
            width = int(os.getenv("RENDER_PREVIEW_WIDTH", str(width // 2)))
            height = int(os.getenv("RENDER_PREVIEW_HEIGHT", str(height // 2)))
        return RenderSettings(width=width, height=height, fps=fps)


def _is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp"}


def _is_subtitle(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".srt", ".ass", ".vtt"}


def _fmt_time(value: float) -> str:
    return f"{value:.3f}"


def _fmt_number(value: float) -> str:
    return f"{value:.2f}"
