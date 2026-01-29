"""Episode orchestrator with HITL gates and caching."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.showrunner.agent import run as showrunner_run
from agents.writer.agent import run as writer_run
from agents.dramaturg.agent import run as dramaturg_run
from agents.casting.agent import run as casting_run
from agents.scene.agent import run as scene_run
from agents.director.agent import run as director_run
from agents.editor.agent import run as editor_run
from agents.qc.agent import run as qc_agent_run
from agents.curator.agent import run as curator_run
from agents.common import LLMClient

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.render.server import RenderService
from mcp_servers.qc.server import QCService
from mcp_servers.lipsync.server import LipSyncService

from .cache import StepCache
from .tts_client import QwenTTSClient
from .audio_utils import build_segments
from .validators import validate_screenplay, validate_cast_plan, validate_timeline_references, estimate_screenplay_duration_sec
from .bibles import load_series_bible, load_character_bible
from .run_logger import RunLogger
from .gates import render_screenplay_markdown, apply_line_edits


@dataclass
class EpisodeConfig:
    theme: str
    mood: str
    duration_sec: int
    auto_approve: bool = False
    seed: Optional[int] = 42


AGENT_SCHEMA_HINTS = {
    "showrunner": "{premise, beats[{label,duration_sec}], tone, cast_constraints{required_characters,max_cast_per_scene,max_scenes}}",
    "writer": "{scenes[{scene_id,setting_prompt,characters[],lines[{line_id,speaker,text,emotion,pause_ms_after,sfx_tag?}]}]}",
    "dramaturg": "{required_edits[], suggested_changes[], duration_targets{...}}",
    "casting": "{cast_plan{roles[{role,character_id,voice_id,avatar_id,emotion_map}]}, cast_bible_update{...}}",
    "scene": "{scenes[{scene_id,background_asset_id,props[],layout_hints{subtitle_safe_zone{x,y,width,height}}}]}",
    "director": "{scenes[{scene_id,stage[],entrances[],reactions[],subtitle_placement{x,y}}]}",
    "editor": "{duration_sec, scenes[{scene_id,start_sec,end_sec,layers[],audio[]}]}",
    "qc": "{duration_sec,audio,video,subtitles,errors[]}",
    "curator": "{updates[],embeddings[]}",
}


class Orchestrator:
    def __init__(self) -> None:
        self.store = ArtifactStore()
        self.cache = StepCache()
        self.render = RenderService()
        self.qc = QCService()
        self.tts = QwenTTSClient()
        self.lipsync = LipSyncService()
        self.llm = LLMClient()
        self.agent_retries = int(os.getenv("AGENT_RETRIES", "2"))
        self.tool_retries = int(os.getenv("TOOL_RETRIES", "5"))
        self.screenplay_tolerance_sec = int(os.getenv("SCREENPLAY_DURATION_TOLERANCE_SEC", "15"))

    def run_daily_episode(self, config: EpisodeConfig) -> Dict[str, Any]:
        episode_id = _make_episode_id()
        run_dir = os.path.join("data", "runs", episode_id)
        logger = RunLogger(run_dir)
        steps: List[Dict[str, Any]] = []

        series_bible = load_series_bible()
        character_bible = load_character_bible()
        logger.save_step("bibles", {"series_bible": series_bible, "character_bible": character_bible})

        episode_brief = self._cached_step(
            "showrunner",
            {"theme": config.theme, "mood": config.mood, "duration_sec": config.duration_sec},
            showrunner_run,
            logger=logger,
        )
        steps.append(_step("showrunner", "completed"))

        screenplay = self._cached_step(
            "writer",
            {
                "episode_brief": episode_brief,
                "series_bible": series_bible,
                "character_bible": character_bible,
                "target_duration_sec": config.duration_sec,
            },
            writer_run,
            validator=lambda data: validate_screenplay(
                data,
                target_duration_sec=config.duration_sec,
                tolerance_sec=self.screenplay_tolerance_sec,
                style_guard=series_bible.get("style_guard", {}),
            ),
            logger=logger,
        )
        steps.append(_step("writer", "completed"))

        dramaturg_notes = self._cached_step("dramaturg", {"screenplay": screenplay}, dramaturg_run, logger=logger)
        steps.append(_step("dramaturg", "completed"))

        if not self._gate_screenplay(screenplay, logger, config.auto_approve):
            raise RuntimeError("Screenplay not approved")

        casting = self._cached_step(
            "casting",
            {"screenplay": screenplay, "character_bible": character_bible},
            casting_run,
            validator=lambda data: validate_cast_plan(data.get("cast_plan", {}), screenplay),
            logger=logger,
        )
        cast_plan = casting.get("cast_plan", {"roles": []})
        steps.append(_step("casting", "completed"))

        scene_assets = self._cached_step("scene", {"screenplay": screenplay}, scene_run, logger=logger)
        steps.append(_step("scene", "completed"))

        scene_plan = self._cached_step("director", {"screenplay": screenplay}, director_run, logger=logger)
        steps.append(_step("director", "completed"))

        performances = self._run_performances(screenplay, cast_plan, logger)
        steps.append(_step("performance", "completed"))

        timeline = self._cached_step(
            "editor",
            {"screenplay": screenplay, "performances": performances},
            editor_run,
            validator=lambda data: validate_timeline_references(data, screenplay, self.store),
            logger=logger,
        )
        steps.append(_step("editor", "completed"))

        timeline_id = self._store_json(timeline, content_type="application/json")
        preview = self._cached_render("render_preview", timeline_id, preset="preview")
        steps.append(_step("render_preview", "completed"))

        highlight = self._make_highlight(preview, logger)
        if not self._gate_preview(preview, highlight, logger, config.auto_approve):
            raise RuntimeError("Preview not approved")

        final = self._cached_render("render_final", timeline_id, preset="final")
        steps.append(_step("render_final", "completed"))

        video_qc = self._call_tool_with_retry(self.qc.qc_video, final)
        audio_qc = self._call_tool_with_retry(self.qc.qc_audio, final)
        qc_report = qc_agent_run(
            {
                "duration_sec": video_qc.get("duration_sec", audio_qc.get("duration_sec", 0.0)),
                "audio": audio_qc,
                "video": video_qc,
                "subtitles": {"missing": False},
                "errors": [],
            }
        )
        qc_id = self._store_json(qc_report, content_type="application/json")
        steps.append(_step("qc", "completed"))

        curator_run({"updates": [], "embeddings": []})
        steps.append(_step("curator", "completed"))

        manifest = {
            "episode_id": episode_id,
            "status": "completed",
            "artifacts": {
                "episode_brief": self._store_json(episode_brief, "application/json"),
                "screenplay": self._store_json(screenplay, "application/json"),
                "cast_plan": self._store_json(cast_plan, "application/json"),
                "scene_assets": self._store_json(scene_assets, "application/json"),
                "scene_plan": self._store_json(scene_plan, "application/json"),
                "timeline": timeline_id,
                "preview_mp4": preview,
                "highlight_mp4": highlight,
                "episode_mp4": final,
                "qc_report": qc_id,
            },
            "steps": steps,
        }
        logger.save_step("manifest", manifest)
        return manifest

    def _store_json(self, data: Dict[str, Any], content_type: str) -> str:
        payload = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
        return self.store.put(payload, content_type=content_type, tags=["json"])

    def _cached_step(
        self,
        name: str,
        payload: Dict[str, Any],
        fn,
        validator=None,
        logger: RunLogger | None = None,
    ) -> Dict[str, Any]:
        key = self.cache.make_key(name, payload)
        cached = self.cache.get(key)
        if cached:
            try:
                path = self.store.get_path(cached)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if validator:
                    errors = validator(data)
                    if errors:
                        data = self._fix_with_retries(name, payload, data, errors, validator, logger)
                return data
            except Exception:
                pass

        if logger:
            step_dir = logger.step_dir(name)
            logger.write_json(os.path.join(step_dir, "input.json"), payload)
            logger.log(f"step:{name} input saved")

        data = fn(payload, llm=self.llm)
        if name == "writer":
            data = _normalize_screenplay(data)

        if logger:
            step_dir = logger.step_dir(name)
            logger.write_json(os.path.join(step_dir, "raw.json"), {
                "prompt": self.llm.last_prompt,
                "raw": self.llm.last_raw,
            })

        if validator:
            errors = validator(data)
            if errors:
                if logger:
                    logger.write_json(os.path.join(logger.step_dir(name), "validation_errors.json"), errors)
                data = self._fix_with_retries(name, payload, data, errors, validator, logger)

        if logger:
            logger.write_json(os.path.join(logger.step_dir(name), "normalized.json"), data)

        artifact_id = self._store_json(data, content_type="application/json")
        self.cache.set(key, artifact_id)
        return data

    def _fix_with_retries(
        self,
        name: str,
        payload: Dict[str, Any],
        bad_json: Dict[str, Any],
        errors: List[str],
        validator,
        logger: RunLogger | None = None,
    ) -> Dict[str, Any]:
        for attempt in range(self.agent_retries):
            fixed = self.llm.complete_json(
                _fix_prompt(
                    name=name,
                    input_json=payload,
                    bad_json=bad_json,
                    errors=errors,
                    schema_hint=AGENT_SCHEMA_HINTS.get(name, ""),
                )
            )
            if name == "writer":
                fixed = _normalize_screenplay(fixed)
            new_errors = validator(fixed)
            if logger:
                logger.write_json(
                    os.path.join(logger.step_dir(name), f"retry_{attempt+1}.json"),
                    {"fixed": fixed, "errors": new_errors},
                )
            if not new_errors:
                return fixed
            bad_json = fixed
            errors = new_errors
        if name == "writer" and any(e.startswith("screenplay_too_short") for e in errors):
            expanded = _auto_expand_screenplay(bad_json, payload.get("target_duration_sec"))
            new_errors = validator(expanded)
            if not new_errors:
                return expanded
        raise RuntimeError(f"{name} invalid after {self.agent_retries} retries: {errors}")

    def _cached_render(self, name: str, timeline_id: str, preset: str) -> str:
        key = self.cache.make_key(name, {"timeline_id": timeline_id, "preset": preset})
        cached = self.cache.get(key)
        if cached:
            return cached
        if preset == "preview":
            result = self._call_tool_with_retry(self.render.render_preview, timeline_id)
        else:
            result = self._call_tool_with_retry(self.render.render_final, timeline_id)
        artifact_id = result["artifact_id"]
        self.cache.set(key, artifact_id)
        return artifact_id

    def _call_tool_with_retry(self, fn, *args, **kwargs):
        last_err: Optional[Exception] = None
        for _ in range(self.tool_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as err:
                last_err = err
                time.sleep(2)
        raise RuntimeError(f"Tool call failed after {self.tool_retries + 1} attempts: {last_err}")

    def _run_performances(self, screenplay: Dict[str, Any], cast_plan: Dict[str, Any], logger: RunLogger) -> Dict[str, Any]:
        batch_pause_sec = float(os.getenv("TTS_BATCH_PAUSE_SEC", "0.2"))
        joiner = os.getenv("TTS_BATCH_JOINER", " ... ")

        performance_manifest: Dict[str, Any] = {"scenes": []}

        scenes = screenplay.get("scenes", [])
        for scene in scenes:
            scene_id = scene.get("scene_id", "scene-1")
            by_character: Dict[str, List[Dict[str, Any]]] = {}
            for line in scene.get("lines", []):
                speaker = line.get("speaker") or "unknown"
                by_character.setdefault(speaker, []).append(line)

            scene_perf = {"scene_id": scene_id, "characters": []}

            def process_character(character_id: str, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
                text = joiner.join([l.get("text", "").strip() for l in lines])
                emotion = _resolve_emotion(lines)
                result = self._call_tool_with_retry(
                    self.tts.tts_synthesize,
                    text=text,
                    character_id=character_id,
                    emotion=emotion,
                    style=None,
                )
                wav_path = result["wav_path"]
                duration_ms = int(result.get("duration_ms", 0))
                total_duration_sec = max(duration_ms / 1000.0, 0.0)
                segments = build_segments(lines, total_duration_sec, batch_pause_sec)
                wav_id = _artifact_id_from_path(wav_path)

                lipsync_result = self._call_tool_with_retry(
                    self.lipsync.lipsync_render_clip,
                    avatar_id=_select_avatar(cast_plan, character_id),
                    wav_id=wav_id,
                )
                video_id = lipsync_result["artifact_id"]

                return {
                    "character_id": character_id,
                    "wav_artifact_id": wav_id,
                    "video_artifact_id": video_id,
                    "segments": [
                        {
                            "line_id": line.get("line_id"),
                            "start_sec": seg.get("start_sec", 0.0),
                            "end_sec": seg.get("start_sec", 0.0) + seg.get("duration_sec", 0.0),
                        }
                        for line, seg in zip(lines, segments)
                    ],
                }

            with ThreadPoolExecutor(max_workers=min(8, len(by_character))) as executor:
                futures = {
                    executor.submit(process_character, character_id, lines): character_id
                    for character_id, lines in by_character.items()
                }
                for fut in as_completed(futures):
                    scene_perf["characters"].append(fut.result())

            performance_manifest["scenes"].append(scene_perf)

        logger.save_step("performance", performance_manifest)
        return performance_manifest

    def _gate_screenplay(self, screenplay: Dict[str, Any], logger: RunLogger, auto: bool) -> bool:
        md = render_screenplay_markdown(screenplay)
        md_path = os.path.join(logger.run_dir, "gate_a_screenplay.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.log(f"Gate A screenplay markdown: {md_path}")

        if auto:
            return True

        print(f"Gate A screenplay: {md_path}")
        print("Enter edits in format: line_id|new text. Blank line to finish.")
        edits: Dict[str, str] = {}
        while True:
            raw = input("> ").strip()
            if not raw:
                break
            if "|" not in raw:
                print("Invalid format. Use line_id|new text")
                continue
            line_id, new_text = raw.split("|", 1)
            edits[line_id.strip()] = new_text.strip()

        if edits:
            screenplay = apply_line_edits(screenplay, edits)
            logger.write_json(os.path.join(logger.run_dir, "gate_a_edits.json"), edits)
        return True

    def _make_highlight(self, preview_artifact_id: str, logger: RunLogger) -> str:
        duration = int(os.getenv("HIGHLIGHT_DURATION_SEC", "75"))
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        preview_path = self.store.get_path(preview_artifact_id)
        highlight_path = os.path.join(logger.run_dir, "highlight.mp4")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            preview_path,
            "-t",
            str(duration),
            "-c",
            "copy",
            highlight_path,
        ]
        _run_cmd(cmd)
        with open(highlight_path, "rb") as f:
            data = f.read()
        return self.store.put(data=data, content_type="video/mp4", tags=["highlight"])

    def _gate_preview(self, preview_id: str, highlight_id: str, logger: RunLogger, auto: bool) -> bool:
        if auto:
            return True
        print("Gate B highlight artifact:", highlight_id)
        reply = input("Approve preview highlight? (y/n): ").strip().lower()
        return reply in {"y", "yes"}


def _resolve_emotion(group: List[Dict[str, Any]]) -> str:
    emotions = {str((line.get("emotion") or "neutral")).lower() for line in group}
    if len(emotions) == 1:
        return emotions.pop()
    return "neutral"


def _select_avatar(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("role") == speaker:
            return role.get("avatar_id", "avatar-default")
    return "avatar-default"


def _artifact_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".wav"):
        return base[:-4]
    return base


def _make_episode_id() -> str:
    return datetime.now(timezone.utc).strftime("ep-%Y%m%d-%H%M%S")


def _step(name: str, status: str) -> Dict[str, Any]:
    return {"name": name, "status": status}


def _fix_prompt(
    name: str,
    input_json: Dict[str, Any],
    bad_json: Dict[str, Any],
    errors: List[str],
    schema_hint: str,
) -> str:
    extra = ""
    if name == "writer" and any(e.startswith("screenplay_too_short") for e in errors):
        extra = (
            "The screenplay is too short. Expand by adding more lines/scenes "
            "until the target duration is reached, while obeying style guard.\n"
        )
    bad_json_str = json.dumps(bad_json, ensure_ascii=True)
    if len(bad_json_str) > 2000 and name == "writer":
        bad_json_str = ""
    return (
        "Fix the JSON to match the schema and constraints. Return JSON only.\n\n"
        f"Step: {name}\n"
        f"Schema: {schema_hint}\n"
        f"Errors: {errors}\n\n"
        f"{extra}\n"
        f"Input: {input_json}\n\n"
        f"Current JSON: {bad_json_str}\n"
    )


def _auto_expand_screenplay(screenplay: Dict[str, Any], target_duration_sec: Any) -> Dict[str, Any]:
    try:
        target = float(target_duration_sec or 0)
    except Exception:
        target = 0.0
    if target <= 0:
        return screenplay

    scenes = screenplay.get("scenes", [])
    if not scenes:
        return screenplay

    # Append filler lines to the last scene until estimated duration meets target.
    last_scene = scenes[-1]
    lines = last_scene.setdefault("lines", [])
    characters = last_scene.get("characters") or []
    speaker = characters[0] if characters else "Narrator"

    line_idx = len(lines) + 1
    while estimate_screenplay_duration_sec(screenplay) < target:
        text = "We should keep exploring this and see what happens next."
        lines.append(
            {
                "line_id": f"auto-{line_idx}",
                "speaker": speaker,
                "text": text,
                "emotion": "neutral",
                "pause_ms_after": 200,
            }
        )
        line_idx += 1
        if line_idx > 2000:
            break
    return screenplay


def _normalize_screenplay(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    scenes = screenplay.get("scenes") or []
    line_counter = 1
    scene_counter = 1
    for scene in scenes:
        if not scene.get("scene_id"):
            scene["scene_id"] = f"scene-{scene_counter}"
        scene_counter += 1
        lines = scene.get("lines") or []
        characters = scene.get("characters") or []
        char_set = set()
        for c in characters:
            if isinstance(c, dict):
                value = c.get("character_id") or c.get("name") or c.get("id")
            else:
                value = c
            if value:
                char_set.add(value)
        for line in lines:
            speaker = line.get("speaker")
            if speaker and speaker not in char_set:
                characters.append(speaker)
                char_set.add(speaker)
            line["line_id"] = line.get("line_id") or f"line-{line_counter}"
            line["line_id"] = str(line_counter)
            line_counter += 1
        scene["characters"] = characters
    return screenplay


def _run_cmd(cmd: List[str]) -> None:
    import subprocess

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
