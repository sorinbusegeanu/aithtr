"""
FFmpeg timeline renderer with optional ASR subtitle burn-in.
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
        self.subtitles_mode = os.getenv("SUBTITLES_MODE", "off")
        self.subtitles_lang = os.getenv("SUBTITLES_LANG", "en")
        self.subtitles_style = os.getenv("SUBTITLES_STYLE", "")
        self.asr_model = None
        if self.subtitles_mode == "asr":
            self.asr_model = _load_asr_model()

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

            base_video = os.path.join(tmpdir, "episode_base.mp4")
            self._concat_scenes(scene_outputs, base_video)

            final_video = base_video
            if self.subtitles_mode == "asr" and self.asr_model is not None:
                ass_path = os.path.join(tmpdir, "subtitles.ass")
                _generate_ass_from_asr(self.asr_model, base_video, ass_path, lang=self.subtitles_lang, style=self.subtitles_style)
                final_video = os.path.join(tmpdir, "episode_subtitled.mp4")
                _burn_subtitles(base_video, ass_path, final_video)

            with open(final_video, "rb") as f:
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
        filters: List[str] = []

        bg_path = self._resolve(bg_layer["asset_id"])
        bg_is_image = _is_image(bg_path)
        if bg_is_image:
            inputs += ["-loop", "1", "-t", _fmt_time(duration), "-i", bg_path]
        else:
            inputs += ["-i", bg_path]

        overlay_layers = [l for l in layers if l.get("type") in {"actor", "props"}]
        subtitle_layers = [l for l in layers if l.get("type") == "caption"]

        for layer in overlay_layers:
            asset_path = self._resolve(layer["asset_id"])
            if _is_image(asset_path):
                inputs += ["-loop", "1", "-t", _fmt_time(duration), "-i", asset_path]
            else:
                inputs += ["-i", asset_path]

        for layer in audio:
            asset_path = self._resolve(layer["asset_id"])
            inputs += ["-i", asset_path]

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
                f"[{video_input_index}:v]trim={_fmt_time(local_start)}:{_fmt_time(local_end)},"
                "setpts=PTS-STARTPTS,"
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


def _load_asr_model():
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _generate_ass_from_asr(model, video_path: str, ass_path: str, lang: str, style: str) -> None:
    # Extract audio to wav for ASR
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    segments, _info = model.transcribe(wav_path, language=lang)
    _write_ass(ass_path, list(segments), style=style)


def _write_ass(path: str, segments: List[Any], style: str = "") -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    if style:
        header = header.replace("Style: Default,Arial,42", f"Style: Default,{style}")

    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    lines = [header]
    for seg in segments:
        start = fmt_time(seg.start)
        end = fmt_time(seg.end)
        text = seg.text.strip().replace("\n", " ")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _burn_subtitles(video_path: str, ass_path: str, out_path: str) -> None:
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"subtitles='{ass_path}'",
        "-c:a",
        "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)


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
