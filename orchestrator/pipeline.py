"""Episode orchestrator with HITL gates and caching."""
import json
import os
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

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.render.server import RenderService
from mcp_servers.qc.server import QCService
from mcp_servers.tts.server import TTSService
from mcp_servers.lipsync.server import LipSyncService

from .cache import StepCache


@dataclass
class EpisodeConfig:
    theme: str
    mood: str
    duration_sec: int
    auto_approve: bool = False
    seed: Optional[int] = 42


class Orchestrator:
    def __init__(self) -> None:
        self.store = ArtifactStore()
        self.cache = StepCache()
        self.render = RenderService()
        self.qc = QCService()
        self.tts = TTSService()
        self.lipsync = LipSyncService()

    def run_daily_episode(self, config: EpisodeConfig) -> Dict[str, Any]:
        episode_id = _make_episode_id()
        steps: List[Dict[str, Any]] = []

        episode_brief = self._cached_step(
            "showrunner",
            {"theme": config.theme, "mood": config.mood, "duration_sec": config.duration_sec},
            showrunner_run,
        )
        steps.append(_step("showrunner", "completed"))

        screenplay = self._cached_step("writer", {"episode_brief": episode_brief}, writer_run)
        steps.append(_step("writer", "completed"))

        dramaturg_notes = self._cached_step("dramaturg", {"screenplay": screenplay}, dramaturg_run)
        steps.append(_step("dramaturg", "completed"))

        if not self._gate("Gate A: Approve screenplay?", config.auto_approve):
            raise RuntimeError("Screenplay not approved")

        casting = self._cached_step("casting", {"screenplay": screenplay}, casting_run)
        cast_plan = casting.get("cast_plan", {"roles": []})
        steps.append(_step("casting", "completed"))

        scene_assets = self._cached_step("scene", {"screenplay": screenplay}, scene_run)
        steps.append(_step("scene", "completed"))

        scene_plan = self._cached_step("director", {"screenplay": screenplay}, director_run)
        steps.append(_step("director", "completed"))

        timeline = self._cached_step("editor", {"screenplay": screenplay}, editor_run)
        steps.append(_step("editor", "completed"))

        self._run_performances(screenplay, cast_plan)
        steps.append(_step("performance", "completed"))

        timeline_id = self._store_json(timeline, content_type="application/json")
        preview = self._cached_render("render_preview", timeline_id, preset="preview")
        steps.append(_step("render_preview", "completed"))

        if not self._gate("Gate B: Approve preview?", config.auto_approve):
            raise RuntimeError("Preview not approved")

        final = self._cached_render("render_final", timeline_id, preset="final")
        steps.append(_step("render_final", "completed"))

        video_qc = self.qc.qc_video(final)
        audio_qc = self.qc.qc_audio(final)
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
                "episode_mp4": final,
                "qc_report": qc_id,
            },
            "steps": steps,
        }
        return manifest

    def _store_json(self, data: Dict[str, Any], content_type: str) -> str:
        payload = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
        return self.store.put(payload, content_type=content_type, tags=["json"])

    def _cached_step(self, name: str, payload: Dict[str, Any], fn) -> Dict[str, Any]:
        key = self.cache.make_key(name, payload)
        cached = self.cache.get(key)
        if cached:
            path = self.store.get_path(cached)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        result = fn(payload)
        artifact_id = self._store_json(result, content_type="application/json")
        self.cache.set(key, artifact_id)
        return result

    def _cached_render(self, name: str, timeline_id: str, preset: str) -> str:
        key = self.cache.make_key(name, {"timeline_id": timeline_id, "preset": preset})
        cached = self.cache.get(key)
        if cached:
            return cached
        if preset == "preview":
            result = self.render.render_preview(timeline_id)
        else:
            result = self.render.render_final(timeline_id)
        artifact_id = result["artifact_id"]
        self.cache.set(key, artifact_id)
        return artifact_id

    def _run_performances(self, screenplay: Dict[str, Any], cast_plan: Dict[str, Any]) -> None:
        lines = []
        for scene in screenplay.get("scenes", []):
            for line in scene.get("lines", []):
                lines.append(line)

        if not lines:
            return

        def synthesize(line: Dict[str, Any]) -> str:
            text = line.get("text", "")
            voice_id = _select_voice(cast_plan, line.get("speaker"))
            try:
                result = self.tts.tts_synthesize(text=text, voice_id=voice_id)
                return result["artifact_id"]
            except Exception:
                # Fallback: store empty wav
                return self.store.put(b"", content_type="audio/wav", tags=["silence"])

        def render(line: Dict[str, Any], wav_id: str) -> None:
            avatar_id = _select_avatar(cast_plan, line.get("speaker"))
            try:
                self.lipsync.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_id)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=min(8, len(lines))) as executor:
            futures = {executor.submit(synthesize, line): line for line in lines}
            for fut in as_completed(futures):
                line = futures[fut]
                wav_id = fut.result()
                executor.submit(render, line, wav_id)

    def _gate(self, prompt: str, auto: bool) -> bool:
        if auto:
            return True
        reply = input(f"{prompt} (y/n): ").strip().lower()
        return reply in {"y", "yes"}


def _select_voice(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("role") == speaker:
            return role.get("voice_id", "voice-default")
    return "voice-default"


def _select_avatar(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("role") == speaker:
            return role.get("avatar_id", "avatar-default")
    return "avatar-default"


def _make_episode_id() -> str:
    return datetime.now(timezone.utc).strftime("ep-%Y%m%d-%H%M%S")


def _step(name: str, status: str) -> Dict[str, Any]:
    return {"name": name, "status": status}
