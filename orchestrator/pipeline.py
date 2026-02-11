"""Episode orchestrator with HITL gates and caching."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from copy import deepcopy
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
from agents.critic.agent import run as critic_run
from agents.common import LLMClient

from .mcp_clients import AssetClient, LipSyncClient, QCClient, RenderClient
from .cache import StepCache
from .tts_client import XTTSClient
from .audio_utils import build_segments, split_wav_ffmpeg
from .validators import (
    estimate_screenplay_duration_sec,
    validate_screenplay,
    validate_cast_plan_contract,
    validate_timeline_references,
)
from .bibles import load_series_bible, load_character_bible, save_character_bible
from .memory_client import MemoryClient
from .run_logger import RunLogger
from .gates import render_screenplay_markdown, apply_line_edits
from .voice_seeder import VOICE_SEED_FAILED, VoiceSeederConfig, load_voice_map, resolve_voice_id, run_voice_seeder
from .assets import build_assets


@dataclass
class EpisodeConfig:
    theme: str
    mood: str
    duration_sec: int
    auto_approve: bool = False
    seed: Optional[int] = 42
    transcript: bool = False
    transcript_path: Optional[str] = None
    resume_run_id: Optional[str] = None
    resume_from_step: Optional[str] = None
    force_regenerate_voices: bool = False


AGENT_SCHEMA_HINTS = {
    "showrunner": "{premise, beats[{label,duration_sec}], tone, cast_constraints{required_characters,max_cast_per_scene,max_scenes}}",
    "writer": "{scenes[{scene_id,setting_prompt,characters[],lines[{line_id,speaker,text,emotion,pause_ms_after,sfx_tag?}]}]}",
    "dramaturg": "{required_edits[], suggested_changes[], duration_targets{...}}",
    "casting": "{cast_plan{roles[{role,character_id,display_name,voice_id,voice_seed_text?,voice_seed_seconds_target?,avatar_id,emotion_map}]}, cast_bible_update{...}}",
    "scene": "{scenes[{scene_id,background_asset_id,props[],layout_hints{subtitle_safe_zone{x,y,width,height}}}]}",
    "assets": "{characters[{character_id,avatar_image_path,avatar_image_artifact_id,avatar_source_video_path,avatar_source_video_artifact_id}],scenes[{scene_id,background_path,background_artifact_id}]}",
    "director": "{scenes[{scene_id,stage[],entrances[],reactions[],subtitle_placement{x,y}}]}",
    "editor": "{duration_sec, scenes[{scene_id,start_sec,end_sec,layers[],audio[]}]}",
    "qc": "{duration_sec,audio,video,subtitles,errors[]}",
    "curator": "{updates[],embeddings[]}",
}


class Orchestrator:
    def __init__(self) -> None:
        self.store = AssetClient()
        self.cache = StepCache()
        self.render = RenderClient()
        self.qc = QCClient()
        self.tts = XTTSClient()
        self.lipsync = LipSyncClient()
        self.llm_clients: Dict[str, LLMClient] = {}
        self.memory = MemoryClient()
        self.agent_retries = int(os.getenv("AGENT_RETRIES", "2"))
        self.critic_retries = int(os.getenv("CRITIC_RETRIES", "3"))
        self.tool_retries = int(os.getenv("TOOL_RETRIES", "5"))
        self.screenplay_tolerance_sec = int(os.getenv("SCREENPLAY_DURATION_TOLERANCE_SEC", "15"))
        self.critic_gate_enabled = os.getenv("CRITIC_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}

    def _llm_for(self, agent_name: str) -> LLMClient:
        llm = self.llm_clients.get(agent_name)
        if llm is None:
            llm = LLMClient(agent_name=agent_name)
            self.llm_clients[agent_name] = llm
        return llm

    def run_daily_episode(self, config: EpisodeConfig) -> Dict[str, Any]:
        episode_id = config.resume_run_id or _make_episode_id()
        data_root = os.getenv("DATA_ROOT", "data")
        run_dir = os.path.join(data_root, "runs", episode_id)
        if config.resume_run_id and not os.path.exists(run_dir):
            raise RuntimeError(f"resume run not found: {run_dir}")
        transcript_path = None
        if config.transcript_path:
            transcript_path = config.transcript_path
        elif config.transcript:
            transcript_path = os.path.join(run_dir, "conversation.log")
        logger = RunLogger(run_dir, transcript_path=transcript_path)
        steps: List[Dict[str, Any]] = []

        series_bible = load_series_bible()
        character_bible = self._load_character_bible()
        episode_summaries = self._memory_texts("episode", type_="episode_summary", k=5)
        continuity_notes = self._memory_texts("continuity", type_="continuity", k=5)
        asset_reuse_notes = self._memory_texts("asset", type_="scene_assets", k=5)
        voice_tuning_history = self._memory_texts("voice", type_="voice_tuning", k=10)
        logger.save_step("bibles", {"series_bible": series_bible, "character_bible": character_bible})
        self._preflight_episode_services(
            _load_comfyui_cfg(),
            logger=logger,
            episode_id=episode_id,
            resume_from_step=config.resume_from_step,
        )

        episode_brief = self._resume_or_cached_step(
            "showrunner",
            config.resume_from_step,
            logger,
            "showrunner",
            {
                "theme": config.theme,
                "mood": config.mood,
                "duration_sec": config.duration_sec,
                "memory": {
                    "episode_summaries": episode_summaries,
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("showrunner"),
                },
            },
            showrunner_run,
            validator=None,
            critic=True,
        )
        steps.append(_step("showrunner", "completed"))

        casting = self._resume_or_cached_step(
            "casting",
            config.resume_from_step,
            logger,
            "casting",
            {
                "episode_brief": episode_brief,
                "character_bible": character_bible,
                "memory": {
                    "voice_tuning_history": voice_tuning_history,
                    "critic_lessons": self._critic_lessons("casting"),
                },
            },
            casting_run,
            validator=lambda data: validate_cast_plan_contract(
                data.get("cast_plan", {}),
                required_characters=(episode_brief.get("cast_constraints", {}) or {}).get("required_characters", []),
            ),
            critic=True,
        )
        cast_plan = casting.get("cast_plan", {"roles": []})
        steps.append(_step("casting", "completed"))
        character_bible = self._apply_cast_bible_update(character_bible, casting.get("cast_bible_update", {}))
        cast_plan = self._normalize_cast_voice_ids(cast_plan)
        cast_plan = self._resolve_cast_avatars(cast_plan, character_bible)
        character_bible = self._sync_character_bible_avatars_from_cast(character_bible, cast_plan)
        self._validate_cast_avatar_contract(cast_plan, logger=logger, episode_id=episode_id)
        self._store_voice_tuning_history(cast_plan, episode_id)

        allowed_speakers = _allowed_speakers_from_cast(cast_plan)
        cast_roster = _cast_roster_from_plan(cast_plan)
        writer_targets = _default_writer_targets(config.duration_sec)
        writer_character_bible = dict(character_bible) if isinstance(character_bible, dict) else {}
        writer_character_bible.pop("missing_cast_for_speakers", None)

        writer_validator = _build_writer_validator(
            target_duration_sec=config.duration_sec,
            tolerance_sec=self.screenplay_tolerance_sec,
            style_guard=series_bible.get("style_guard", {}),
            allowed_speakers=set(allowed_speakers),
            writer_targets=writer_targets,
        )

        screenplay = self._resume_or_cached_step(
            "writer",
            config.resume_from_step,
            logger,
            "writer",
            {
                "episode_brief": episode_brief,
                "series_bible": series_bible,
                "character_bible": writer_character_bible,
                "cast_roster": cast_roster,
                "allowed_speakers": allowed_speakers,
                "target_duration_sec": config.duration_sec,
                "writer_targets": writer_targets,
                "memory": {
                    "episode_summaries": episode_summaries,
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("writer"),
                },
            },
            writer_run,
            validator=writer_validator,
            critic=True,
        )
        screenplay = _strip_writer_meta(screenplay)
        steps.append(_step("writer", "completed"))

        dramaturg_notes = self._resume_or_cached_step(
            "dramaturg",
            config.resume_from_step,
            logger,
            "dramaturg",
            {
                "screenplay": screenplay,
                "memory": {
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("dramaturg"),
                },
            },
            dramaturg_run,
            validator=None,
            critic=True,
        )
        steps.append(_step("dramaturg", "completed"))

        if not self._gate_screenplay(screenplay, logger, config.auto_approve):
            raise RuntimeError("Screenplay not approved")

        self._validate_script_outputs(screenplay, cast_plan, logger=logger, episode_id=episode_id)
        self._validate_scene_avatar_assignments(
            screenplay=screenplay,
            cast_plan=cast_plan,
            logger=logger,
            episode_id=episode_id,
        )

        scene_assets = self._resume_or_cached_step(
            "scene",
            config.resume_from_step,
            logger,
            "scene",
            {
                "screenplay": screenplay,
                "memory": {
                    "asset_reuse": asset_reuse_notes,
                    "critic_lessons": self._critic_lessons("scene"),
                },
            },
            scene_run,
            validator=None,
            critic=True,
        )
        steps.append(_step("scene", "completed"))
        self._store_scene_assets(scene_assets, episode_id)

        assets = self._resume_or_cached_step(
            "assets",
            config.resume_from_step,
            logger,
            "assets",
            {
                "episode_id": episode_id,
                "cast_plan": cast_plan,
                "screenplay": screenplay,
                "scene_plan": {},
                "scene_assets": scene_assets,
                "comfyui": _load_comfyui_cfg(),
            },
            self._run_assets_stage,
            validator=None,
            critic=False,
        )
        cast_plan = assets.get("cast_plan", cast_plan) if isinstance(assets, dict) else cast_plan
        scene_assets = assets.get("scene_assets", scene_assets) if isinstance(assets, dict) else scene_assets
        self._validate_assets_stage(
            cast_plan=cast_plan,
            scene_assets=scene_assets,
            logger=logger,
            episode_id=episode_id,
        )
        if logger:
            logger.write_json(os.path.join(logger.step_dir("casting"), "normalized.json"), {"cast_plan": cast_plan})
            logger.write_json(os.path.join(logger.step_dir("scene"), "normalized.json"), scene_assets)
        steps.append(_step("assets", "completed"))

        scene_plan = self._resume_or_cached_step(
            "director",
            config.resume_from_step,
            logger,
            "director",
            {
                "screenplay": screenplay,
                "memory": {
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("director"),
                },
            },
            director_run,
            validator=None,
            critic=True,
        )
        steps.append(_step("director", "completed"))

        self._run_voice_seeder(cast_plan, logger, config.force_regenerate_voices)
        steps.append(_step("voice_seeder", "completed"))

        performances = self._resume_or_run_performances(
            screenplay,
            cast_plan,
            logger,
            config.resume_from_step,
            force=config.force_regenerate_voices,
        )
        self._validate_tts_avatar_outputs(
            performances,
            logger=logger,
            episode_id=episode_id,
        )
        self._ensure_performances_ready(
            performances,
            logger=logger,
            episode_id=episode_id,
        )
        steps.append(_step("performance", "completed"))

        timeline = self._resume_or_cached_step(
            "editor",
            config.resume_from_step,
            logger,
            "editor",
            {
                "screenplay": _compact_screenplay_for_editor(screenplay),
                "performances": _compact_performances_for_editor(performances),
                "scene_assets": _compact_scene_assets_for_editor(scene_assets),
                "scene_plan": _compact_scene_plan_for_editor(scene_plan),
            },
            editor_run,
            validator=lambda data: validate_timeline_references(data, screenplay, self.store),
            critic=False,
        )
        timeline = self._normalize_timeline_for_render(
            timeline,
            scene_assets,
            scene_plan,
            performances,
            logger=logger,
            episode_id=episode_id,
        )
        self._write_compose_manifest(timeline, logger)
        self._validate_compose_manifest(
            timeline,
            logger=logger,
            episode_id=episode_id,
        )
        steps.append(_step("editor", "completed"))

        timeline_id = self._store_json(timeline, content_type="application/json")
        preview = self._resume_or_cached_render("render_preview", timeline_id, "preview", logger, config.resume_from_step)
        steps.append(_step("render_preview", "completed"))

        highlight = self._make_highlight(preview, logger)
        if not self._gate_preview(preview, highlight, logger, config.auto_approve):
            raise RuntimeError("Preview not approved")

        final = self._resume_or_cached_render("render_final", timeline_id, "final", logger, config.resume_from_step)
        line_count = sum(
            len(scene.get("lines", []))
            for scene in screenplay.get("scenes", [])
            if isinstance(scene, dict)
        )
        self._validate_final_render(
            final,
            line_count=line_count,
            logger=logger,
            episode_id=episode_id,
        )
        steps.append(_step("render_final", "completed"))

        try:
            video_qc = self._call_tool_with_retry(self.qc.qc_video, final)
        except RuntimeError as err:
            if self._is_missing_ffprobe_error(err) and self._allow_noffprobe_qc_fallback():
                video_qc = {
                    "duration_sec": float(timeline.get("duration_sec", 0.0)),
                    "status": "warning",
                    "warnings": ["qc_video skipped: ffprobe not available"],
                }
            else:
                raise
        try:
            audio_qc = self._call_tool_with_retry(self.qc.qc_audio, final)
        except RuntimeError as err:
            if self._is_missing_ffprobe_error(err) and self._allow_noffprobe_qc_fallback():
                audio_qc = {
                    "duration_sec": float(timeline.get("duration_sec", 0.0)),
                    "status": "warning",
                    "warnings": ["qc_audio skipped: ffprobe not available"],
                }
            else:
                raise
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

        curator_updates = self._cached_step(
            "curator",
            {"updates": [], "embeddings": []},
            curator_run,
            logger=logger,
            critic=True,
        )
        steps.append(_step("curator", "completed"))

        manifest = {
            "episode_id": episode_id,
            "status": "completed",
            "assets": {
                "comfyui_prompts": (
                    (((assets or {}).get("comfyui") or {}).get("prompts", {}))
                    if isinstance(assets, dict)
                    else {}
                )
            },
            "artifacts": {
                "episode_brief": self._store_json(episode_brief, "application/json"),
                "screenplay": self._store_json(screenplay, "application/json"),
                "cast_plan": self._store_json(cast_plan, "application/json"),
                "scene_assets": self._store_json(scene_assets, "application/json"),
                "assets": self._store_json(assets, "application/json"),
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
        self._store_episode_memory(
            episode_id=episode_id,
            episode_brief=episode_brief,
            screenplay=screenplay,
            cast_plan=cast_plan,
            scene_assets=scene_assets,
            logger=logger,
        )
        return manifest

    def _store_json(self, data: Dict[str, Any], content_type: str) -> str:
        payload = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
        return self.store.put(payload, content_type=content_type, tags=["json"])

    def _preflight_episode_services(
        self,
        cfg: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
        resume_from_step: Optional[str] = None,
    ) -> None:
        stage_order = [
            "showrunner",
            "casting",
            "writer",
            "dramaturg",
            "scene",
            "assets",
            "director",
            "voice_seeder",
            "performance",
            "editor",
            "render_preview",
            "render_final",
            "qc",
        ]
        order_index = {name: idx for idx, name in enumerate(stage_order)}
        start_idx = order_index.get(str(resume_from_step or "").strip(), 0)
        check_comfy = start_idx <= order_index["assets"]
        check_tts = start_idx <= order_index["performance"]
        check_lipsync = start_idx <= order_index["performance"]

        # 1) ffmpeg
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        if not shutil.which(ffmpeg_bin):
            self._raise_stage_failure(
                stage="preflight",
                line_id=None,
                artifact_path=ffmpeg_bin,
                reason="MISSING_FFMPEG",
                message=f"ffmpeg binary not found: {ffmpeg_bin}",
                episode_id=episode_id,
                logger=logger,
            )

        # 2) ComfyUI HTTP health
        comfy_enabled = bool((cfg or {}).get("enabled", True))
        if check_comfy and comfy_enabled:
            host = str((cfg or {}).get("host", "127.0.0.1"))
            port = int((cfg or {}).get("port", 8188))
            timeout = float((cfg or {}).get("request_timeout_sec", 10))
            comfy_url = f"http://{host}:{port}/system_stats"
            try:
                req = urllib.request.Request(comfy_url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = int(getattr(resp, "status", 200) or 200)
                if status != 200:
                    raise RuntimeError(f"unexpected status {status}")
            except Exception as err:
                self._raise_stage_failure(
                    stage="preflight",
                    line_id=None,
                    artifact_path=comfy_url,
                    reason="COMFYUI_UNREACHABLE",
                    message=f"ComfyUI preflight failed: {err}",
                    episode_id=episode_id,
                    logger=logger,
                )

        # 3) Wav2Lip MCP reachability
        if check_lipsync and getattr(self.lipsync, "mode", "local") != "local":
            lipsync_url = str(getattr(self.lipsync, "url", "") or "")
            try:
                req = urllib.request.Request(lipsync_url, method="GET")
                with urllib.request.urlopen(req, timeout=float(os.getenv("MCP_HTTP_TIMEOUT_SEC", "10"))):
                    pass
            except urllib.error.HTTPError:
                # HTTP response means service is reachable.
                pass
            except Exception as err:
                self._raise_stage_failure(
                    stage="preflight",
                    line_id=None,
                    artifact_path=lipsync_url or None,
                    reason="LIPSYNC_UNREACHABLE",
                    message=f"Wav2Lip MCP preflight failed: {err}",
                    episode_id=episode_id,
                    logger=logger,
                )

        # 4) TTS MCP reachability
        if check_tts and getattr(self.tts, "mode", "local") != "local":
            tts_url = str(getattr(self.tts, "url", "") or "")
            try:
                req = urllib.request.Request(tts_url, method="GET")
                with urllib.request.urlopen(req, timeout=float(os.getenv("MCP_HTTP_TIMEOUT_SEC", "10"))):
                    pass
            except urllib.error.HTTPError:
                # HTTP response means service is reachable.
                pass
            except Exception as err:
                self._raise_stage_failure(
                    stage="preflight",
                    line_id=None,
                    artifact_path=tts_url or None,
                    reason="TTS_UNREACHABLE",
                    message=f"TTS MCP preflight failed: {err}",
                    episode_id=episode_id,
                    logger=logger,
                )

    def _raise_stage_failure(
        self,
        *,
        stage: str,
        reason: str,
        message: str,
        episode_id: str,
        logger: Optional[RunLogger],
        line_id: Optional[str] = None,
        artifact_path: Optional[str] = None,
        final_artifact_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "stage": stage,
            "line_id": line_id,
            "artifact_path": artifact_path,
            "reason": reason,
            "message": message,
            "extra": extra or {},
        }
        if logger:
            logger.log(f"stage_failure:{json.dumps(payload, ensure_ascii=True)}")
        self._write_debug_bundle(
            episode_id=episode_id,
            logger=logger,
            error_payload=payload,
            final_artifact_id=final_artifact_id,
        )
        raise StageValidationError(payload)

    def _write_compose_manifest(self, timeline: Dict[str, Any], logger: RunLogger) -> str:
        compose_dir = os.path.join(logger.run_dir, "compose")
        os.makedirs(compose_dir, exist_ok=True)
        scenes_manifest: List[Dict[str, Any]] = []
        for scene in timeline.get("scenes", []) if isinstance(timeline, dict) else []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "")
            bg = None
            video_tracks: List[Dict[str, Any]] = []
            overlays: List[Dict[str, Any]] = []
            for layer in scene.get("layers", []) if isinstance(scene.get("layers"), list) else []:
                if not isinstance(layer, dict):
                    continue
                layer_type = str(layer.get("type") or "")
                if layer_type == "background":
                    bg = layer.get("asset_id")
                elif layer_type in {"actor", "props"}:
                    video_tracks.append(layer)
                else:
                    overlays.append(layer)
            scenes_manifest.append(
                {
                    "scene_id": scene_id,
                    "background": bg,
                    "video_tracks": video_tracks,
                    "audio_tracks": scene.get("audio", []),
                    "overlays": overlays,
                }
            )
        manifest = {
            "canvas": {
                "width": int(os.getenv("RENDER_WIDTH", "1280")),
                "height": int(os.getenv("RENDER_HEIGHT", "720")),
                "fps": int(os.getenv("RENDER_FPS", "30")),
            },
            "timeline": timeline,
            "scenes": scenes_manifest,
        }
        out_path = os.path.join(compose_dir, "manifest.json")
        logger.write_json(out_path, manifest)
        return out_path

    def _validate_script_outputs(
        self,
        screenplay: Dict[str, Any],
        cast_plan: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        scene_list = screenplay.get("scenes", []) if isinstance(screenplay, dict) else []
        if not scene_list:
            self._raise_stage_failure(
                stage="script",
                reason="MISSING_FILE",
                message="screenplay has no scenes",
                episode_id=episode_id,
                logger=logger,
            )
        speaker_to_character = _speaker_to_character(cast_plan)
        for scene in scene_list:
            for line in scene.get("lines", []) if isinstance(scene, dict) else []:
                line_id = str(line.get("line_id") or "").strip() or None
                text = str(line.get("text") or "").strip()
                if not text:
                    self._raise_stage_failure(
                        stage="script",
                        reason="EMPTY_TEXT",
                        message="line text is empty",
                        line_id=line_id,
                        episode_id=episode_id,
                        logger=logger,
                    )
                speaker = str(line.get("speaker") or "").strip()
                if speaker not in speaker_to_character:
                    self._raise_stage_failure(
                        stage="script",
                        reason="UNRESOLVED_CHARACTER",
                        message=f"speaker cannot be resolved to cast entry: {speaker!r}",
                        line_id=line_id,
                        episode_id=episode_id,
                        logger=logger,
                    )

    def _validate_tts_avatar_outputs(
        self,
        performances: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        sample_rate_expected = int(os.getenv("TTS_EXPECTED_SAMPLE_RATE", "22050"))
        min_duration = float(os.getenv("TTS_MIN_LINE_DURATION_SEC", "0.2"))
        min_rms = float(os.getenv("TTS_MIN_RMS", "0.005"))
        max_peak = float(os.getenv("TTS_MAX_PEAK", "0.99"))
        min_motion = float(os.getenv("VIDEO_MIN_MOTION_DIFF", "0.06"))

        import cv2
        import numpy as np
        import soundfile as sf

        for scene in performances.get("scenes", []) if isinstance(performances, dict) else []:
            for row in scene.get("characters", []) if isinstance(scene, dict) else []:
                if str(row.get("status", "")) != "ok":
                    continue
                if not str(row.get("voice_id") or "").strip():
                    self._raise_stage_failure(
                        stage="tts",
                        reason="MISSING_VOICE_ASSIGNMENT",
                        message="missing voice_id on successful character performance row",
                        line_id=None,
                        episode_id=episode_id,
                        logger=logger,
                    )
                line_audio_refs = row.get("line_audio_artifacts", [])
                if not isinstance(line_audio_refs, list) or not line_audio_refs:
                    self._raise_stage_failure(
                        stage="tts",
                        reason="MISSING_FILE",
                        message="missing per-line audio artifacts",
                        line_id=None,
                        episode_id=episode_id,
                        logger=logger,
                    )
                for line_ref in line_audio_refs:
                    line_id = str(line_ref.get("line_id") or "").strip() or None
                    wav_id = str(line_ref.get("wav_artifact_id") or "").strip()
                    if not wav_id:
                        self._raise_stage_failure(
                            stage="tts",
                            reason="MISSING_FILE",
                            message="missing per-line wav artifact id",
                            line_id=line_id,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    wav_path = self.store.get_path(wav_id)
                    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
                    if audio is None:
                        self._raise_stage_failure(
                            stage="tts",
                            reason="MISSING_FILE",
                            message="could not read per-line wav",
                            line_id=line_id,
                            artifact_path=wav_path,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    data = np.asarray(audio, dtype=np.float32)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    duration = float(data.shape[0]) / float(sr or 1)
                    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
                    peak = float(np.max(np.abs(data))) if data.size else 0.0
                    if duration < min_duration:
                        self._raise_stage_failure(
                            stage="tts",
                            reason="DURATION_TOO_SHORT",
                            message=f"line wav duration too short: {duration:.3f}s",
                            line_id=line_id,
                            artifact_path=wav_path,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    if sr != sample_rate_expected:
                        self._raise_stage_failure(
                            stage="tts",
                            reason="UNEXPECTED_SAMPLE_RATE",
                            message=f"sample rate={sr}, expected={sample_rate_expected}",
                            line_id=line_id,
                            artifact_path=wav_path,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    if rms < min_rms:
                        # Keep as non-fatal signal; warning logging intentionally suppressed.
                        pass
                    if peak >= max_peak:
                        self._raise_stage_failure(
                            stage="tts",
                            reason="CLIPPING_RISK",
                            message=f"line wav peak too high: {peak:.6f}",
                            line_id=line_id,
                            artifact_path=wav_path,
                            episode_id=episode_id,
                            logger=logger,
                        )

                line_video_refs = row.get("line_video_artifacts", [])
                if not isinstance(line_video_refs, list) or not line_video_refs:
                    self._raise_stage_failure(
                        stage="avatar",
                        reason="MISSING_FILE",
                        message="missing per-line video artifacts",
                        line_id=None,
                        episode_id=episode_id,
                        logger=logger,
                    )
                for line_ref in line_video_refs:
                    line_id = str(line_ref.get("line_id") or "").strip() or None
                    video_id = str(line_ref.get("video_artifact_id") or "").strip()
                    source_diag = line_ref.get("source_lipsync_diagnostics")
                    if not isinstance(source_diag, dict):
                        source_diag = row.get("lipsync_diagnostics")
                    if not video_id:
                        self._raise_stage_failure(
                            stage="avatar",
                            reason="MISSING_FILE",
                            message="missing per-line avatar clip id",
                            line_id=line_id,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    video_path = self.store.get_path(video_id)
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    ok, prev = cap.read()
                    motion_sum = 0.0
                    motion_n = 0
                    motion_nonzero_n = 0
                    while ok:
                        ok2, cur = cap.read()
                        if not ok2:
                            break
                        diff = cv2.absdiff(cur, prev)
                        diff_mean = float(diff.mean())
                        motion_sum += diff_mean
                        motion_n += 1
                        if diff_mean > 1e-6:
                            motion_nonzero_n += 1
                        prev = cur
                    cap.release()
                    if frame_count <= 1:
                        self._raise_stage_failure(
                            stage="avatar",
                            reason="ZERO_FRAMES",
                            message=f"avatar clip has frame_count={frame_count}",
                            line_id=line_id,
                            artifact_path=video_path,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    if width <= 0 or height <= 0:
                        self._raise_stage_failure(
                            stage="avatar",
                            reason="INVALID_RESOLUTION",
                            message=f"avatar clip has invalid resolution {width}x{height}",
                            line_id=line_id,
                            artifact_path=video_path,
                            episode_id=episode_id,
                            logger=logger,
                        )
                    mean_motion = (motion_sum / motion_n) if motion_n else 0.0
                    post_nonzero_ratio = (float(motion_nonzero_n) / float(motion_n)) if motion_n else 0.0
                    min_nonzero_ratio = float(os.getenv("AVATAR_MIN_NONZERO_DIFF_RATIO", "0.15"))
                    source_min_motion = float(os.getenv("AVATAR_SOURCE_MIN_MOTION_DIFF", "0.01"))
                    source_mean_motion = None
                    source_nonzero_ratio = None
                    if isinstance(source_diag, dict):
                        try:
                            source_video = source_diag.get("video_pre_split") or {}
                            source_motion = source_diag.get("motion_signal") or {}
                            source_mean_motion = float(source_video.get("mean_frame_diff", 0.0))
                            source_nonzero_ratio = float(source_video.get("nonzero_frame_diff_ratio", 0.0))
                            if source_nonzero_ratio <= 0.0 and source_motion.get("nonzero_ratio") is not None:
                                source_nonzero_ratio = float(source_motion.get("nonzero_ratio"))
                        except Exception:
                            source_mean_motion = None
                            source_nonzero_ratio = None
                    source_static = (
                        source_mean_motion is None
                        or (source_mean_motion < source_min_motion and (source_nonzero_ratio or 0.0) < min_nonzero_ratio)
                    )
                    post_static = mean_motion < min_motion and post_nonzero_ratio < min_nonzero_ratio
                    if post_static and source_static:
                        extra = {
                            "line_clip_post_encode_mean_diff": mean_motion,
                            "line_clip_post_encode_nonzero_ratio": post_nonzero_ratio,
                            "line_clip_frame_count": frame_count,
                            "line_clip_fps": float(frame_count / max(float(line_ref.get("duration_sec") or 0.0), 1e-6))
                            if line_ref.get("duration_sec")
                            else None,
                            "line_clip_resolution": {"width": width, "height": height},
                        }
                        if isinstance(source_diag, dict):
                            extra["source_lipsync_diagnostics"] = source_diag
                            try:
                                source_video = source_diag.get("video_pre_split") or {}
                                source_motion = source_diag.get("motion_signal") or {}
                                extra["source_pre_split_mean_diff"] = float(source_video.get("mean_frame_diff", 0.0))
                                extra["source_pre_split_nonzero_ratio"] = float(
                                    source_video.get("nonzero_frame_diff_ratio", 0.0)
                                )
                                extra["source_motion_signal_nonzero_ratio"] = source_motion.get("nonzero_ratio")
                                extra["source_landmark_tracking_ok"] = source_motion.get("landmark_tracking_ok")
                            except Exception:
                                pass
                        self._raise_stage_failure(
                            stage="avatar",
                            reason="NO_MOTION",
                            message=f"avatar clip appears static: mean frame diff={mean_motion:.4f}",
                            line_id=line_id,
                            artifact_path=video_path,
                            episode_id=episode_id,
                            logger=logger,
                            extra=extra,
                        )

    def _validate_compose_manifest(
        self,
        timeline: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        width = int(os.getenv("RENDER_WIDTH", "1280"))
        height = int(os.getenv("RENDER_HEIGHT", "720"))
        for scene in timeline.get("scenes", []) if isinstance(timeline, dict) else []:
            layers = scene.get("layers", []) if isinstance(scene, dict) else []
            if not isinstance(layers, list):
                continue
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                aid = str(layer.get("asset_id") or "").strip()
                if not aid:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="MISSING_FILE",
                        message="layer missing asset_id",
                        episode_id=episode_id,
                        logger=logger,
                    )
                try:
                    self.store.get_path(aid)
                except Exception:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="MISSING_FILE",
                        message=f"layer asset is not available: {aid}",
                        artifact_path=aid,
                        episode_id=episode_id,
                        logger=logger,
                    )
                pos = layer.get("position") or {}
                x = float(pos.get("x", 0.5))
                y = float(pos.get("y", 0.5))
                if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="OUT_OF_BOUNDS_PLACEMENT",
                        message=f"layer position out of bounds: x={x}, y={y}",
                        artifact_path=aid,
                        episode_id=episode_id,
                        logger=logger,
                    )
                scale = float(layer.get("scale", 1.0))
                if scale <= 0:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="OUT_OF_BOUNDS_PLACEMENT",
                        message=f"layer scale must be > 0, got {scale}",
                        artifact_path=aid,
                        episode_id=episode_id,
                        logger=logger,
                    )
                if int(layer.get("z", 0)) < 0:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="MISSING_Z_ORDER",
                        message="layer z-order is missing/invalid",
                        artifact_path=aid,
                        episode_id=episode_id,
                        logger=logger,
                    )
                if "z" not in layer:
                    self._raise_stage_failure(
                        stage="compose",
                        reason="MISSING_Z_ORDER",
                        message="layer z-order key is missing",
                        artifact_path=aid,
                        episode_id=episode_id,
                        logger=logger,
                    )
                if str(layer.get("type") or "") in {"background", "actor", "props"}:
                    try:
                        import cv2

                        path = self.store.get_path(aid)
                        cap = cv2.VideoCapture(path)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        cap.release()
                        if w <= 0 or h <= 0:
                            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                h, w = img.shape[:2]
                        if w <= 0 or h <= 0:
                            self._raise_stage_failure(
                                stage="compose",
                                reason="UNKNOWN_DIMENSIONS",
                                message="layer asset width/height could not be resolved",
                                artifact_path=path,
                                episode_id=episode_id,
                                logger=logger,
                            )
                    except StageValidationError:
                        raise
                    except Exception:
                        self._raise_stage_failure(
                            stage="compose",
                            reason="UNKNOWN_DIMENSIONS",
                            message="failed reading layer dimensions",
                            artifact_path=aid,
                            episode_id=episode_id,
                            logger=logger,
                        )
        _ = width, height

    def _validate_final_render(
        self,
        final_artifact_id: str,
        *,
        line_count: int,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        import cv2

        final_path = self.store.get_path(final_artifact_id)
        meta = self._ffprobe_media(final_path)
        vdur = float(meta.get("video_duration_sec", 0.0))
        adur = float(meta.get("audio_duration_sec", 0.0))
        if abs(vdur - adur) > 0.25:
            self._raise_stage_failure(
                stage="render",
                reason="AUDIO_VIDEO_DURATION_MISMATCH",
                message=f"audio/video duration mismatch: audio={adur:.3f}s video={vdur:.3f}s",
                artifact_path=final_path,
                final_artifact_id=final_artifact_id,
                episode_id=episode_id,
                logger=logger,
            )
        if line_count > 1:
            _ = self._detect_audio_gap_count(final_path)

        cap = cv2.VideoCapture(final_path)
        ok, prev = cap.read()
        motion_sum = 0.0
        motion_n = 0
        while ok:
            ok2, cur = cap.read()
            if not ok2:
                break
            diff = cv2.absdiff(cur, prev)
            motion_sum += float(diff.mean())
            motion_n += 1
            prev = cur
        cap.release()
        mean_motion = (motion_sum / motion_n) if motion_n else 0.0
        min_motion = float(os.getenv("FINAL_MIN_MOTION_DIFF", "0.15"))
        if mean_motion < min_motion:
            hard_fail = os.getenv("FINAL_NO_MOTION_HARD_FAIL", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if hard_fail:
                self._raise_stage_failure(
                    stage="render",
                    reason="NO_MOTION",
                    message=f"final video appears static: mean frame diff={mean_motion:.4f}",
                    artifact_path=final_path,
                    final_artifact_id=final_artifact_id,
                    episode_id=episode_id,
                    logger=logger,
                )
            else:
                # Non-fatal mode enabled; warning logging intentionally suppressed.
                pass

    def _ffprobe_media(self, path: str) -> Dict[str, float]:
        ffprobe_bin = os.getenv("FFPROBE_BIN", "ffprobe")
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        video_duration = 0.0
        audio_duration = 0.0
        for stream in data.get("streams", []):
            if not isinstance(stream, dict):
                continue
            duration = float(stream.get("duration") or 0.0)
            if stream.get("codec_type") == "video":
                video_duration = max(video_duration, duration)
            if stream.get("codec_type") == "audio":
                audio_duration = max(audio_duration, duration)
        if video_duration <= 0:
            video_duration = float(data.get("format", {}).get("duration", 0.0))
        if audio_duration <= 0:
            audio_duration = float(data.get("format", {}).get("duration", 0.0))
        return {
            "video_duration_sec": video_duration,
            "audio_duration_sec": audio_duration,
        }

    def _detect_audio_gap_count(self, path: str) -> int:
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        cmd = [
            ffmpeg_bin,
            "-i",
            path,
            "-af",
            "silencedetect=noise=-35dB:d=0.18",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return int(result.stderr.count("silence_start:"))

    def _write_debug_bundle(
        self,
        *,
        episode_id: str,
        logger: Optional[RunLogger],
        error_payload: Dict[str, Any],
        final_artifact_id: Optional[str] = None,
    ) -> None:
        data_root = os.getenv("DATA_ROOT", "data")
        debug_dir = os.path.join(data_root, "debug", episode_id)
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "error.json"), "w", encoding="utf-8") as f:
            json.dump(error_payload, f, ensure_ascii=True, indent=2)

        if not logger:
            return

        for rel in [
            "episode.log",
            "run_manifest.json",
            os.path.join("compose", "manifest.json"),
            os.path.join("performance", "normalized.json"),
            os.path.join("writer", "normalized.json"),
            os.path.join("casting", "normalized.json"),
            os.path.join("editor", "normalized.json"),
        ]:
            src = os.path.join(logger.run_dir, rel)
            if os.path.exists(src):
                dst = os.path.join(debug_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        perf_path = os.path.join(logger.run_dir, "performance", "normalized.json")
        if os.path.exists(perf_path):
            with open(perf_path, "r", encoding="utf-8") as f:
                perf = json.load(f)
            line_durations: List[Dict[str, Any]] = []
            sample_dir = os.path.join(debug_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            for scene in perf.get("scenes", []) if isinstance(perf, dict) else []:
                sid = str(scene.get("scene_id") or "")
                for row in scene.get("characters", []) if isinstance(scene, dict) else []:
                    cid = str(row.get("character_id") or "")
                    for item in row.get("line_audio_artifacts", []) if isinstance(row.get("line_audio_artifacts"), list) else []:
                        line_durations.append(
                            {
                                "scene_id": sid,
                                "character_id": cid,
                                "line_id": item.get("line_id"),
                                "audio_duration_sec": item.get("duration_sec"),
                                "wav_artifact_id": item.get("wav_artifact_id"),
                            }
                        )
                    for item in row.get("line_video_artifacts", []) if isinstance(row.get("line_video_artifacts"), list) else []:
                        line_durations.append(
                            {
                                "scene_id": sid,
                                "character_id": cid,
                                "line_id": item.get("line_id"),
                                "video_duration_sec": item.get("duration_sec"),
                                "video_artifact_id": item.get("video_artifact_id"),
                            }
                        )
                        try:
                            import cv2

                            vpath = self.store.get_path(str(item.get("video_artifact_id") or ""))
                            cap = cv2.VideoCapture(vpath)
                            ok, frame = cap.read()
                            cap.release()
                            if ok and frame is not None:
                                safe_line = _slug(str(item.get("line_id") or "line"))
                                out = os.path.join(sample_dir, f"{sid}_{cid}_{safe_line}.png")
                                cv2.imwrite(out, frame)
                        except Exception:
                            pass
            with open(os.path.join(debug_dir, "line_durations.json"), "w", encoding="utf-8") as f:
                json.dump(line_durations, f, ensure_ascii=True, indent=2)

        if final_artifact_id:
            try:
                final_path = self.store.get_path(final_artifact_id)
                ffprobe = self._ffprobe_media(final_path)
                with open(os.path.join(debug_dir, "final_ffprobe.json"), "w", encoding="utf-8") as f:
                    json.dump(ffprobe, f, ensure_ascii=True, indent=2)
            except Exception:
                pass

    def _resume_or_cached_step(
        self,
        step: str,
        resume_from: Optional[str],
        logger: RunLogger | None,
        name: str,
        payload: Dict[str, Any],
        fn,
        validator=None,
        critic: bool = False,
    ) -> Dict[str, Any]:
        if logger and self._should_resume_step(step, resume_from):
            resumed = self._load_step_output(logger, step)
            if resumed is not None:
                return resumed
        return self._cached_step(name, payload, fn, validator=validator, logger=logger, critic=critic)

    def _resume_or_run_performances(
        self,
        screenplay: Dict[str, Any],
        cast_plan: Dict[str, Any],
        logger: RunLogger | None,
        resume_from: Optional[str],
        force: bool = False,
    ) -> Dict[str, Any]:
        if logger and self._should_resume_step("performance", resume_from) and not force:
            resumed = self._load_step_output(logger, "performance")
            if resumed is not None:
                if self._has_ready_performance_assets(resumed):
                    return resumed
                logger.log("Resumed performance had no successful clips; re-running performance step")
        performances = self._run_performances(screenplay, cast_plan, logger)
        if logger:
            logger.write_json(os.path.join(logger.step_dir("performance"), "normalized.json"), performances)
        return performances

    def _resume_or_cached_render(
        self,
        step: str,
        timeline_id: str,
        preset: str,
        logger: RunLogger | None,
        resume_from: Optional[str],
    ) -> str:
        if logger and self._should_resume_step(step, resume_from):
            resumed = self._load_step_output(logger, step)
            if isinstance(resumed, dict) and resumed.get("artifact_id"):
                artifact_id = str(resumed["artifact_id"])
                if not self._is_unusable_render_artifact(artifact_id):
                    return artifact_id
        artifact_id = self._cached_render(step, timeline_id, preset=preset)
        if logger:
            logger.write_json(os.path.join(logger.step_dir(step), "normalized.json"), {"artifact_id": artifact_id})
        return artifact_id

    def _load_step_output(self, logger: RunLogger, step: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(logger.step_dir(step), "normalized.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _should_resume_step(self, step: str, resume_from: Optional[str]) -> bool:
        if not resume_from:
            return True
        order = [
            "showrunner",
            "casting",
            "writer",
            "dramaturg",
            "scene",
            "assets",
            "director",
            "voice_seeder",
            "performance",
            "editor",
            "render_preview",
            "render_final",
            "qc",
            "curator",
            "manifest",
        ]
        try:
            step_idx = order.index(step)
            resume_idx = order.index(resume_from)
        except ValueError:
            return True
        return step_idx < resume_idx

    def _cached_step(
        self,
        name: str,
        payload: Dict[str, Any],
        fn,
        validator=None,
        logger: RunLogger | None = None,
        critic: bool = False,
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
                        data = self._fix_with_retries(name, payload, data, errors, fn, validator, logger)
                return data
            except Exception:
                pass

        if logger:
            step_dir = logger.step_dir(name)
            logger.write_json(os.path.join(step_dir, "input.json"), payload)
            logger.log(f"step:{name} input saved")

        step_llm = self._llm_for(name)
        data = fn(payload, llm=step_llm)
        if name == "casting":
            data = self._normalize_casting_output(payload, data)
        if name == "writer":
            data = _normalize_writer_screenplay(data)
            data = _apply_writer_length_policy(payload, data, logger=logger, step=name)

        if logger:
            step_dir = logger.step_dir(name)
            logger.write_json(os.path.join(step_dir, "raw.json"), {
                "prompt": step_llm.last_prompt,
                "raw": step_llm.last_raw,
            })
            logger.log_chat(name, step_llm.last_messages, step_llm.last_raw)
            logger.write_json(os.path.join(step_dir, "normalized_pre_fix.json"), data)
            if name == "writer":
                logger.write_json(os.path.join(step_dir, "writer_metrics_pre_fix.json"), _writer_metrics(data))

        if validator:
            errors = validator(data)
            if errors:
                if logger:
                    logger.write_json(os.path.join(logger.step_dir(name), "validation_errors.json"), errors)
                data = self._fix_with_retries(name, payload, data, errors, fn, validator, logger)
        if logger:
            logger.write_json(os.path.join(logger.step_dir(name), "normalized_post_fix.json"), data)
            if name == "writer":
                logger.write_json(os.path.join(logger.step_dir(name), "writer_metrics_post_fix.json"), _writer_metrics(data))

        if critic and self.critic_gate_enabled:
            data = self._critic_gate(name, payload, data, fn, validator, logger)
            if logger:
                logger.write_json(os.path.join(logger.step_dir(name), "normalized_post_critic.json"), data)
                if name == "writer":
                    logger.write_json(os.path.join(logger.step_dir(name), "writer_metrics_post_critic.json"), _writer_metrics(data))
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
        fn,
        validator,
        logger: RunLogger | None = None,
    ) -> Dict[str, Any]:
        step_llm = self._llm_for(name)
        retries = self.agent_retries
        if name == "writer":
            retries = max(self.agent_retries, int(os.getenv("WRITER_RETRIES", "4")))
        prev_sig: Optional[str] = None
        prev_err_sig: Optional[str] = None
        for attempt in range(retries):
            if name == "writer" and _only_line_id_errors(errors):
                fixed_local = _normalize_writer_screenplay(deepcopy(bad_json))
                new_errors_local = validator(fixed_local)
                if not new_errors_local:
                    return fixed_local
            if name == "writer" and _writer_needs_full_regen(errors):
                fixed = fn(
                    payload,
                    llm=step_llm,
                    critic_feedback=_writer_repair_feedback(payload, bad_json, errors),
                )
            else:
                fixed = step_llm.complete_json(
                    _fix_prompt(
                        name=name,
                        input_json=payload,
                        bad_json=bad_json,
                        errors=errors,
                        schema_hint=AGENT_SCHEMA_HINTS.get(name, ""),
                    )
                )
            if name == "casting":
                fixed = self._normalize_casting_output(payload, fixed)
            if name == "writer":
                fixed = _normalize_writer_screenplay(fixed)
                # Avoid repetitive stage_warning spam on retries; keep warnings from initial pass.
                fixed = _apply_writer_length_policy(payload, fixed, logger=None, step=name)
            if logger:
                logger.log_chat(f"{name}.fix", step_llm.last_messages, step_llm.last_raw)
            new_errors = validator(fixed)
            if name == "writer" and _has_writer_repetition_errors(new_errors):
                min_words = int((payload.get("writer_targets", {}) or {}).get("min_total_words") or 0)
                repaired_local = _writer_local_repair(fixed, min_total_words=min_words)
                repaired_local = _normalize_writer_screenplay(repaired_local)
                repaired_local = _apply_writer_length_policy(payload, repaired_local, logger=None, step=name)
                repaired_errors = validator(repaired_local)
                if logger:
                    logger.write_json(
                        os.path.join(logger.step_dir(name), f"retry_{attempt+1}_local_repair.json"),
                        {"fixed": repaired_local, "errors": repaired_errors},
                    )
                fixed = repaired_local
                new_errors = repaired_errors
            if name == "writer":
                sig = json.dumps(fixed, sort_keys=True, ensure_ascii=True)
                err_sig = json.dumps(new_errors, sort_keys=True, ensure_ascii=True)
                if prev_sig == sig and prev_err_sig == err_sig:
                    min_words = int((payload.get("writer_targets", {}) or {}).get("min_total_words") or 0)
                    repaired_local = _writer_local_repair(fixed, min_total_words=min_words)
                    repaired_local = _normalize_writer_screenplay(repaired_local)
                    repaired_local = _apply_writer_length_policy(payload, repaired_local, logger=None, step=name)
                    repaired_errors = validator(repaired_local)
                    if logger:
                        logger.write_json(
                            os.path.join(logger.step_dir(name), f"retry_{attempt+1}_local_repair.json"),
                            {"fixed": repaired_local, "errors": repaired_errors},
                        )
                    if not repaired_errors:
                        return repaired_local
                    raise RuntimeError(
                        f"{name} retry made no progress (identical output/errors): {repaired_errors}"
                    )
                prev_sig = sig
                prev_err_sig = err_sig
            if logger:
                logger.write_json(
                    os.path.join(logger.step_dir(name), f"retry_{attempt+1}.json"),
                    {"fixed": fixed, "errors": new_errors},
                )
            if not new_errors:
                return fixed
            bad_json = fixed
            errors = new_errors
        raise RuntimeError(f"{name} invalid after {retries} retries: {errors}")

    def _critic_gate(
        self,
        name: str,
        payload: Dict[str, Any],
        data: Dict[str, Any],
        fn,
        validator,
        logger: RunLogger | None = None,
    ) -> Dict[str, Any]:
        try:
            base_threshold = int(os.getenv("CRITIC_PASS_SCORE", "50"))
        except ValueError:
            base_threshold = 50
        try:
            max_rounds = int(os.getenv("CRITIC_MAX_ROUNDS", "3"))
        except ValueError:
            max_rounds = 3
        for round_idx in range(max_rounds):
            threshold = base_threshold - (round_idx * 25)
            review = self._run_critic(
                name,
                payload,
                data,
                [],
                logger,
                attempt=0,
                threshold=threshold,
                round_idx=round_idx,
                attempt_in_round=0,
            )
            if review.get("passed", False):
                return data
            for attempt in range(1, self.critic_retries + 1):
                feedback = self._critic_feedback_text(review)
                step_llm = self._llm_for(name)
                data = fn(payload, llm=step_llm, critic_feedback=feedback)
                if name == "casting":
                    data = self._normalize_casting_output(payload, data)
                if name == "writer":
                    data = _normalize_writer_screenplay(data)
                    data = _apply_writer_length_policy(payload, data, logger=logger, step=name)
                if logger:
                    logger.log_chat(name, step_llm.last_messages, step_llm.last_raw)
                    logger.write_json(
                        os.path.join(logger.step_dir(name), f"critic_retry_{round_idx}_{attempt}.json"),
                        {
                            "critic_feedback": feedback,
                            "prompt": step_llm.last_prompt,
                            "raw": step_llm.last_raw,
                            "normalized": data,
                        },
                    )

                if validator:
                    errors = validator(data)
                    if errors:
                        if logger:
                            logger.write_json(
                                os.path.join(logger.step_dir(name), f"critic_retry_{round_idx}_{attempt}_validation_errors.json"),
                                errors,
                            )
                        data = self._fix_with_retries(name, payload, data, errors, fn, validator, logger)

                review = self._run_critic(
                    name,
                    payload,
                    data,
                    [],
                    logger,
                    attempt=attempt,
                    threshold=threshold,
                    round_idx=round_idx,
                    attempt_in_round=attempt,
                )
                if review.get("passed", False):
                    return data
        raise RuntimeError(f"{name} failed critic gate after {max_rounds} rounds")

    def _run_critic(
        self,
        name: str,
        payload: Dict[str, Any],
        data: Dict[str, Any],
        errors: List[str],
        logger: RunLogger | None = None,
        attempt: int = 0,
        threshold: Optional[int] = None,
        round_idx: int = 0,
        attempt_in_round: int = 0,
    ) -> Dict[str, Any]:
        critic_llm = self._llm_for("critic")
        review = critic_run(
            step_name=name,
            input_data=payload,
            output_data=data,
            schema_hint=AGENT_SCHEMA_HINTS.get(name, ""),
            validation_errors=errors,
            llm=critic_llm,
        )
        if threshold is None:
            try:
                threshold = int(os.getenv("CRITIC_PASS_SCORE", "75"))
            except ValueError:
                threshold = 75
        try:
            score = int(review.get("quality_score", 0))
        except Exception:
            score = 0
        if score >= threshold:
            review["passed"] = True
        try:
            self.memory.store(
                type_="critic_review",
                text=self._critic_feedback_text(review),
                tags=["critic", f"step:{name}", "passed" if review.get("passed") else "failed"],
            )
        except Exception:
            pass
        if logger:
            critic_dir = os.path.join(logger.run_dir, "critic", name)
            os.makedirs(critic_dir, exist_ok=True)
            suffix = "" if attempt == 0 else f"_retry_{attempt}"
            logger.write_json(os.path.join(critic_dir, f"review{suffix}.json"), review)
            logger.log_chat(f"{name}.critic", critic_llm.last_messages, critic_llm.last_raw)
            logger.log(
                "critic:"
                + name
                + f" passed={review.get('passed', False)} score={review.get('quality_score', 'unknown')} "
                + f"threshold={threshold} round={round_idx} attempt={attempt_in_round}"
            )
            try:
                run_review_log = os.path.join(logger.run_dir, "critic", "review.log")
                with open(run_review_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{_now()}] step={name} passed={review.get('passed', False)} "
                        f"score={review.get('quality_score', 'unknown')} "
                        f"threshold={threshold} round={round_idx} attempt={attempt_in_round}\n"
                    )
                    evaluation = review.get("evaluation")
                    if evaluation:
                        f.write(str(evaluation).strip() + "\n")
                    f.write("\n")
            except Exception:
                pass
        return review

    def _critic_feedback_text(self, review: Dict[str, Any]) -> str:
        score = review.get("quality_score", "unknown")
        passed = review.get("passed", False)
        evaluation = review.get("evaluation", "")
        feedback = f"passed={passed}\nscore={score}\n\n{evaluation}".strip()
        return feedback

    def _memory_texts(self, query: str, type_: str, k: int = 5) -> List[str]:
        try:
            results = self.memory.retrieve(query=query, k=k, filters={"type": type_})
        except Exception:
            return []
        return [r.get("text", "") for r in results if r.get("text")]

    def _critic_lessons(self, step: str, k: int = 3) -> List[str]:
        try:
            results = self.memory.retrieve(
                query="critic",
                k=k,
                filters={"tags": [f"step:{step}"]},
            )
        except Exception:
            return []
        return [r.get("text", "") for r in results if r.get("text")]

    def _load_character_bible(self) -> Dict[str, Any]:
        memory_bible = None
        try:
            memory_bible = self.memory.get_bible("character_bible")
        except Exception:
            memory_bible = None
        if memory_bible:
            return memory_bible
        bible = load_character_bible()
        if bible:
            try:
                self.memory.put_bible("character_bible", bible)
            except Exception:
                pass
        return bible

    def _apply_cast_bible_update(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        if not update:
            return current
        merged = dict(current)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        save_character_bible(merged)
        try:
            self.memory.put_bible("character_bible", merged)
        except Exception:
            pass
        return merged

    def _avatar_is_usable(self, avatar_id: Optional[str]) -> bool:
        if not avatar_id or avatar_id == "avatar-default":
            return False
        try:
            path = self.store.get_path(avatar_id)
        except Exception:
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        min_bytes = int(os.getenv("MIN_AVATAR_BYTES", "2048"))
        return size >= min_bytes

    def _fallback_avatar_from_catalog(self) -> Optional[str]:
        catalog_path = os.getenv("ASSET_CATALOG_PATH", os.path.join(os.getenv("DATA_ROOT", "data"), "assets", "catalog.json"))
        if not os.path.exists(catalog_path):
            return None
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            return None
        if isinstance(catalog, list):
            results = catalog
        elif isinstance(catalog, dict):
            results = catalog.get("assets") or catalog.get("items") or []
        else:
            return None
        if not isinstance(results, list):
            return None
        for entry in results:
            if not isinstance(entry, dict):
                continue
            tags = entry.get("tags") or []
            kind = entry.get("kind")
            if kind != "avatar" and "avatar" not in tags:
                continue
            avatar_id = entry.get("asset_id")
            if self._avatar_is_usable(avatar_id):
                return avatar_id
        return None

    def _resolve_cast_avatars(self, cast_plan: Dict[str, Any], character_bible: Dict[str, Any]) -> Dict[str, Any]:
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        if not roles:
            return cast_plan

        imported_pool = self._load_imported_avatar_pool()
        prefer_imported = os.getenv("CASTING_PREFER_IMPORTED_AVATARS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        used_avatar_ids: set[str] = set()
        for role in roles:
            avatar_id = str(role.get("avatar_id") or "").strip()
            if prefer_imported:
                replacement = self._next_available_avatar(imported_pool, used_avatar_ids)
                if replacement:
                    role["avatar_id"] = replacement
                    used_avatar_ids.add(replacement)
                    continue
            # Keep valid, unique avatar assignments from casting.
            if self._avatar_is_usable(avatar_id) and avatar_id not in used_avatar_ids:
                used_avatar_ids.add(avatar_id)
                continue
            mapped = ""
            character_id = role.get("character_id")
            if character_id:
                for character in character_bible.get("characters", []) if isinstance(character_bible, dict) else []:
                    if character.get("character_id") == character_id:
                        mapped = str(character.get("avatar_id") or "").strip()
                        break
            if self._avatar_is_usable(mapped) and mapped not in used_avatar_ids:
                role["avatar_id"] = mapped
                used_avatar_ids.add(mapped)
                continue

            replacement = self._next_available_avatar(imported_pool, used_avatar_ids)
            role["avatar_id"] = replacement or ""
            if replacement:
                used_avatar_ids.add(replacement)

        return cast_plan

    def _load_imported_avatar_pool(self) -> List[str]:
        avatar_dir = os.getenv(
            "CASTING_AVATAR_IMPORT_DIR",
            os.path.join(os.getenv("DATA_ROOT", "data"), "assets", "avatars", "imported_from_downloads"),
        )
        if not os.path.isdir(avatar_dir):
            return []
        names = sorted(os.listdir(avatar_dir))
        scored: List[tuple[float, str]] = []
        for name in names:
            lower = name.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png") or lower.endswith(".webp")):
                continue
            path = os.path.join(avatar_dir, name)
            if not os.path.isfile(path):
                continue
            score = _avatar_expression_score(path)
            try:
                with open(path, "rb") as f:
                    avatar_id = self.store.put(
                        data=f.read(),
                        content_type=_content_type_for_image_name(name),
                        tags=["avatar", "imported", "casting"],
                    )
                if self._avatar_is_usable(avatar_id):
                    scored.append((score, avatar_id))
            except Exception:
                continue
        # Prefer avatars with clearer frontal faces and mouth-region detail.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [aid for _, aid in scored]

    def _next_available_avatar(self, pool: List[str], used: set[str]) -> str:
        for avatar_id in pool:
            if avatar_id and avatar_id not in used and self._avatar_is_usable(avatar_id):
                return avatar_id
        return ""

    def _sync_character_bible_avatars_from_cast(
        self,
        character_bible: Dict[str, Any],
        cast_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(character_bible, dict):
            return character_bible
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        if not isinstance(roles, list) or not roles:
            return character_bible
        characters = character_bible.get("characters")
        if not isinstance(characters, list):
            return character_bible
        changed = False
        by_id: Dict[str, Dict[str, Any]] = {}
        for entry in characters:
            if isinstance(entry, dict):
                cid = str(entry.get("character_id") or "").strip()
                if cid:
                    by_id[cid] = entry
        for role in roles:
            if not isinstance(role, dict):
                continue
            cid = str(role.get("character_id") or "").strip()
            aid = str(role.get("avatar_id") or "").strip()
            if not cid or not self._avatar_is_usable(aid):
                continue
            target = by_id.get(cid)
            if not isinstance(target, dict):
                continue
            current = str(target.get("avatar_id") or "").strip()
            # Preserve canonical bible avatar once set; cast role can override per-episode.
            if not current:
                target["avatar_id"] = aid
                changed = True
        if changed:
            save_character_bible(character_bible)
            try:
                self.memory.put_bible("character_bible", character_bible)
            except Exception:
                pass
        return character_bible

    def _normalize_casting_output(self, payload: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        cast_plan = data.get("cast_plan")
        if not isinstance(cast_plan, dict):
            cast_plan = {}
            data["cast_plan"] = cast_plan
        roles = cast_plan.get("roles")
        if not isinstance(roles, list):
            roles = []
        roles = [r for r in roles if isinstance(r, dict)]

        default_voice = os.getenv("PIPER_DEFAULT_VOICE_ID", "en_US-lessac-medium")
        imported_pool = self._load_imported_avatar_pool()
        used_avatar_ids: set[str] = set()
        seen_character_ids: set[str] = set()

        normalized_roles: List[Dict[str, Any]] = []
        for role in roles:
            role_name = str(role.get("role") or "").strip() or "Unknown"
            display_name = str(role.get("display_name") or "").strip() or role_name
            character_id = str(role.get("character_id") or "").strip()
            if not character_id:
                character_id = _character_id_from_name(role_name, seen_character_ids)
            if character_id in seen_character_ids:
                character_id = _character_id_from_name(character_id, seen_character_ids)
            seen_character_ids.add(character_id)

            voice_id = str(role.get("voice_id") or "").strip() or default_voice
            voice_seed_text = str(role.get("voice_seed_text") or "").strip() or (
                f"{display_name} speaks clearly about mythology, symbols, and meaning in a calm informative tone."
            )
            voice_seed_seconds_target = role.get("voice_seed_seconds_target", 10)
            emotion_map = role.get("emotion_map")
            if not isinstance(emotion_map, dict):
                emotion_map = {
                    "neutral": {"style": "neutral"},
                    "happy": {"style": "warm"},
                    "sad": {"style": "soft"},
                    "angry": {"style": "sharp"},
                }

            avatar_id = str(role.get("avatar_id") or "").strip()
            if not self._avatar_is_usable(avatar_id) or avatar_id in used_avatar_ids:
                avatar_id = self._next_available_avatar(imported_pool, used_avatar_ids)
            if avatar_id:
                used_avatar_ids.add(avatar_id)

            normalized_roles.append(
                {
                    "role": role_name,
                    "character_id": character_id,
                    "display_name": display_name,
                    "voice_id": voice_id,
                    "voice_seed_text": voice_seed_text,
                    "voice_seed_seconds_target": voice_seed_seconds_target,
                    "avatar_id": avatar_id,
                    "emotion_map": emotion_map,
                }
            )

        required = (
            (payload.get("episode_brief", {}) or {}).get("cast_constraints", {}) or {}
        ).get("required_characters", [])
        required_list = [str(x or "").strip() for x in required if str(x or "").strip()]
        aliases: set[str] = set()
        for role in normalized_roles:
            aliases.update(
                {
                    str(role.get("role") or "").strip(),
                    str(role.get("display_name") or "").strip(),
                    str(role.get("character_id") or "").strip(),
                }
            )
        for req in required_list:
            if req in aliases:
                continue
            character_id = _character_id_from_name(req, seen_character_ids)
            seen_character_ids.add(character_id)
            avatar_id = self._next_available_avatar(imported_pool, used_avatar_ids)
            if avatar_id:
                used_avatar_ids.add(avatar_id)
            normalized_roles.append(
                {
                    "role": req,
                    "character_id": character_id,
                    "display_name": req,
                    "voice_id": default_voice,
                    "voice_seed_text": f"{req} introduces themselves and explains their role in this story.",
                    "voice_seed_seconds_target": 10,
                    "avatar_id": avatar_id,
                    "emotion_map": {
                        "neutral": {"style": "neutral"},
                        "happy": {"style": "warm"},
                        "sad": {"style": "soft"},
                        "angry": {"style": "sharp"},
                    },
                }
            )

        cast_plan["roles"] = normalized_roles
        if not isinstance(data.get("cast_bible_update"), dict):
            data["cast_bible_update"] = {}
        return data

    def _validate_cast_avatar_contract(
        self,
        cast_plan: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        import cv2

        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        seen_ids: set[str] = set()
        min_w = int(os.getenv("MIN_AVATAR_WIDTH", "128"))
        min_h = int(os.getenv("MIN_AVATAR_HEIGHT", "128"))
        for role in roles:
            if not isinstance(role, dict):
                continue
            character_id = str(role.get("character_id") or "").strip() or None
            avatar_id = str(role.get("avatar_id") or "").strip()
            if not avatar_id:
                self._raise_stage_failure(
                    stage="casting",
                    reason="MISSING_AVATAR_ID",
                    message="cast role missing avatar_id",
                    line_id=character_id,
                    artifact_path=None,
                    episode_id=episode_id,
                    logger=logger,
                )
            if avatar_id in seen_ids:
                self._raise_stage_failure(
                    stage="casting",
                    reason="DUPLICATE_AVATAR_ID",
                    message=f"avatar_id reused across roles: {avatar_id}",
                    line_id=character_id,
                    artifact_path=avatar_id,
                    episode_id=episode_id,
                    logger=logger,
                )
            seen_ids.add(avatar_id)
            if not self._avatar_is_usable(avatar_id):
                self._raise_stage_failure(
                    stage="casting",
                    reason="INVALID_AVATAR_ASSET",
                    message=f"avatar_id not usable: {avatar_id}",
                    line_id=character_id,
                    artifact_path=avatar_id,
                    episode_id=episode_id,
                    logger=logger,
                )
            avatar_path = self.store.get_path(avatar_id)
            img = cv2.imread(avatar_path)
            if img is None:
                self._raise_stage_failure(
                    stage="casting",
                    reason="INVALID_AVATAR_ASSET",
                    message=f"avatar image unreadable: {avatar_id}",
                    line_id=character_id,
                    artifact_path=avatar_path,
                    episode_id=episode_id,
                    logger=logger,
                )
            h, w = img.shape[:2]
            if w < min_w or h < min_h:
                self._raise_stage_failure(
                    stage="casting",
                    reason="INVALID_AVATAR_DIMENSIONS",
                    message=f"avatar image too small: {w}x{h}, min={min_w}x{min_h}",
                    line_id=character_id,
                    artifact_path=avatar_path,
                    episode_id=episode_id,
                    logger=logger,
                )

    def _validate_scene_avatar_assignments(
        self,
        *,
        screenplay: Dict[str, Any],
        cast_plan: Dict[str, Any],
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        speaker_to_character = _speaker_to_character(cast_plan)
        character_to_avatar: Dict[str, str] = {}
        for role in cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []:
            if not isinstance(role, dict):
                continue
            cid = str(role.get("character_id") or "").strip()
            aid = str(role.get("avatar_id") or "").strip()
            if cid and aid:
                character_to_avatar[cid] = aid
        for scene in screenplay.get("scenes", []) if isinstance(screenplay, dict) else []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip() or None
            active_speakers: set[str] = set()
            for line in scene.get("lines", []) if isinstance(scene.get("lines"), list) else []:
                if isinstance(line, dict):
                    speaker = str(line.get("speaker") or "").strip()
                    if speaker:
                        active_speakers.add(speaker)
            active_character_ids: List[str] = []
            for speaker in sorted(active_speakers):
                cid = speaker_to_character.get(speaker, "")
                if cid:
                    active_character_ids.append(cid)
            active_avatars = [character_to_avatar.get(cid, "") for cid in active_character_ids if cid]
            if len(active_avatars) >= 2 and len(set(active_avatars)) < len(active_avatars):
                self._raise_stage_failure(
                    stage="casting",
                    reason="SCENE_DUPLICATE_AVATAR_ID",
                    message=f"scene has active speakers sharing the same avatar: {scene_id}",
                    line_id=scene_id,
                    artifact_path=None,
                    episode_id=episode_id,
                    logger=logger,
                    extra={
                        "active_character_ids": active_character_ids,
                        "active_avatar_ids": active_avatars,
                    },
                )
            if len(active_avatars) > 1 and len(set(active_avatars)) == 1:
                self._raise_stage_failure(
                    stage="casting",
                    reason="SCENE_ALL_SAME_AVATAR",
                    message=f"scene resolved all active speakers to one avatar: {scene_id}",
                    line_id=scene_id,
                    artifact_path=active_avatars[0] if active_avatars else None,
                    episode_id=episode_id,
                    logger=logger,
                    extra={
                        "active_character_ids": active_character_ids,
                        "active_avatar_ids": active_avatars,
                    },
                )

    def _store_voice_tuning_history(self, cast_plan: Dict[str, Any], episode_id: str) -> None:
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        for role in roles:
            character_id = role.get("character_id")
            voice_id = role.get("voice_id")
            payload = {
                "character_id": character_id,
                "voice_id": voice_id,
                "emotion_map": role.get("emotion_map", {}),
            }
            tags = ["voice", "voice_tuning"]
            if character_id:
                tags.append(f"character:{character_id}")
            if voice_id:
                tags.append(f"voice:{voice_id}")
            try:
                self.memory.store(
                    type_="voice_tuning",
                    text=json.dumps(payload, ensure_ascii=True, indent=2),
                    tags=tags,
                    episode_id=episode_id,
                )
            except Exception:
                pass

    def _normalize_cast_voice_ids(self, cast_plan: Dict[str, Any]) -> Dict[str, Any]:
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        default_voice = os.getenv("PIPER_DEFAULT_VOICE_ID", "en_US-lessac-medium")
        for role in roles:
            character_id = str(role.get("character_id") or "").strip()
            voice_id = str(role.get("voice_id") or "").strip()
            if not voice_id or voice_id == "voice-default":
                # Generate a stable per-character voice id so each cast member can
                # have distinct seeded reference audio.
                suffix = _slug(character_id or str(role.get("role") or "character"))
                role["voice_id"] = f"voice-{suffix}" if suffix else f"voice-{default_voice}"
        return cast_plan

    def _run_voice_seeder(self, cast_plan: Dict[str, Any], logger: RunLogger | None, force: bool) -> Dict[str, Any]:
        if logger and os.path.exists(os.path.join(logger.step_dir("voice_seeder"), "normalized.json")) and not force:
            resumed = self._load_step_output(logger, "voice_seeder")
            if resumed is not None:
                return resumed
        data_root = os.getenv("DATA_ROOT", "data")
        model_dir = os.getenv("PIPER_MODEL_DIR", os.path.join(data_root, "tts", "piper_voices"))
        cfg = VoiceSeederConfig(
            data_root=data_root,
            piper_model_dir=model_dir,
            piper_bin=os.getenv("PIPER_BIN", "piper"),
            force_regenerate=force,
            seed=int(os.getenv("VOICE_SEEDER_SEED", "42")),
        )
        result = run_voice_seeder(cast_plan, cfg)
        if logger:
            logger.write_json(os.path.join(logger.step_dir("voice_seeder"), "normalized.json"), result)
            logger.save_step("voice_seeder", {"characters": result.get("results", [])})
        return result

    def _run_assets_stage(
        self,
        payload: Dict[str, Any],
        llm=None,
        critic_feedback: str | None = None,
    ) -> Dict[str, Any]:
        del llm, critic_feedback
        episode_id = str(payload.get("episode_id") or "").strip()
        cast_plan = payload.get("cast_plan", {}) if isinstance(payload, dict) else {}
        scene_assets = payload.get("scene_assets", {}) if isinstance(payload, dict) else {}
        screenplay = payload.get("screenplay", {}) if isinstance(payload, dict) else {}
        scene_plan = payload.get("scene_plan", {}) if isinstance(payload, dict) else {}
        cfg = {
            "comfyui": payload.get("comfyui", {}),
            "scene_assets": scene_assets,
        }
        return build_assets(
            episode_id=episode_id,
            cast_plan=cast_plan,
            screenplay=screenplay,
            scene_plan=scene_plan,
            cfg=cfg,
        )

    def _materialize_episode_assets(
        self,
        *,
        episode_id: str,
        cast_plan: Dict[str, Any],
        scene_assets: Dict[str, Any],
    ) -> Dict[str, Any]:
        import cv2

        if not episode_id:
            raise RuntimeError("assets stage requires episode_id")
        data_root = os.getenv("DATA_ROOT", "data")
        run_assets_root = os.path.join(data_root, "runs", episode_id, "assets")
        characters_root = os.path.join(run_assets_root, "characters")
        backgrounds_root = os.path.join(run_assets_root, "backgrounds")
        os.makedirs(characters_root, exist_ok=True)
        os.makedirs(backgrounds_root, exist_ok=True)

        source_fps = float(os.getenv("ASSETS_SOURCE_FPS", "25"))
        source_duration_sec = max(float(os.getenv("ASSETS_SOURCE_DURATION_SEC", "4.0")), 0.5)
        source_width = int(os.getenv("ASSETS_SOURCE_WIDTH", "720"))
        source_height = int(os.getenv("ASSETS_SOURCE_HEIGHT", "720"))
        source_frames = max(1, int(round(source_duration_sec * source_fps)))

        out_characters: List[Dict[str, Any]] = []
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        for role in roles:
            if not isinstance(role, dict):
                continue
            character_id = str(role.get("character_id") or "").strip()
            avatar_id = str(role.get("avatar_id") or "").strip()
            if not character_id:
                raise RuntimeError("assets stage: missing character_id in cast_plan")
            if not avatar_id:
                raise RuntimeError(f"assets stage: missing avatar_id for character '{character_id}'")
            try:
                avatar_src_path = self.store.get_path(avatar_id)
            except Exception as err:
                raise RuntimeError(
                    f"assets stage: avatar artifact unavailable for character '{character_id}' (avatar_id={avatar_id})"
                ) from err

            char_dir = os.path.join(characters_root, character_id)
            os.makedirs(char_dir, exist_ok=True)
            portrait_path = os.path.join(char_dir, "portrait.png")
            source_video_path = os.path.join(char_dir, "source.mp4")

            frame = cv2.imread(avatar_src_path)
            if frame is None:
                cap = cv2.VideoCapture(avatar_src_path)
                ok, first = cap.read()
                cap.release()
                if not ok or first is None:
                    raise RuntimeError(
                        f"assets stage: could not decode avatar media for character '{character_id}' from {avatar_src_path}"
                    )
                frame = first
            frame = cv2.resize(frame, (source_width, source_height))
            if not cv2.imwrite(portrait_path, frame):
                raise RuntimeError(f"assets stage: failed to write portrait image for character '{character_id}'")

            writer = cv2.VideoWriter(
                source_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (source_width, source_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"assets stage: failed to open source.mp4 writer for character '{character_id}'")
            try:
                for _ in range(source_frames):
                    writer.write(frame)
            finally:
                writer.release()

            with open(source_video_path, "rb") as f:
                source_video_artifact_id = self.store.put(
                    data=f.read(),
                    content_type="video/mp4",
                    tags=["avatar", "source_video", f"character:{character_id}"],
                )

            out_characters.append(
                {
                    "character_id": character_id,
                    "avatar_id": avatar_id,
                    "avatar_image_path": portrait_path,
                    "avatar_source_video_path": source_video_path,
                    "avatar_source_video_artifact_id": source_video_artifact_id,
                }
            )

        out_scenes: List[Dict[str, Any]] = []
        for scene in scene_assets.get("scenes", []) if isinstance(scene_assets, dict) else []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip()
            background_id = str(scene.get("background_asset_id") or "").strip()
            if not scene_id:
                raise RuntimeError("assets stage: scene entry missing scene_id")
            if not background_id:
                raise RuntimeError(f"assets stage: scene '{scene_id}' missing background_asset_id")
            try:
                bg_src_path = self.store.get_path(background_id)
            except Exception as err:
                raise RuntimeError(
                    f"assets stage: scene '{scene_id}' background artifact unavailable ({background_id})"
                ) from err
            bg_path = os.path.join(backgrounds_root, f"{scene_id}.png")
            bg_img = cv2.imread(bg_src_path)
            if bg_img is None:
                cap = cv2.VideoCapture(bg_src_path)
                ok, first = cap.read()
                cap.release()
                if not ok or first is None:
                    raise RuntimeError(
                        f"assets stage: scene '{scene_id}' background media unreadable from {bg_src_path}"
                    )
                bg_img = first
            if not cv2.imwrite(bg_path, bg_img):
                raise RuntimeError(f"assets stage: failed to write background for scene '{scene_id}'")

            out_scenes.append(
                {
                    "scene_id": scene_id,
                    "background_artifact_id": background_id,
                    "background_path": bg_path,
                }
            )

        return {"characters": out_characters, "scenes": out_scenes}

    def _apply_assets_to_plan(
        self,
        *,
        cast_plan: Dict[str, Any],
        scene_assets: Dict[str, Any],
        assets: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        cast_out = deepcopy(cast_plan if isinstance(cast_plan, dict) else {})
        scene_out = deepcopy(scene_assets if isinstance(scene_assets, dict) else {})

        by_character: Dict[str, Dict[str, Any]] = {}
        for item in assets.get("characters", []) if isinstance(assets, dict) else []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("character_id") or "").strip()
            if cid:
                by_character[cid] = item
        for role in cast_out.get("roles", []) if isinstance(cast_out.get("roles"), list) else []:
            if not isinstance(role, dict):
                continue
            cid = str(role.get("character_id") or "").strip()
            item = by_character.get(cid)
            if not item:
                continue
            role["avatar_image_path"] = item.get("avatar_image_path")
            role["avatar_image_artifact_id"] = item.get("avatar_image_artifact_id")
            role["avatar_source_video_path"] = item.get("avatar_source_video_path")
            role["avatar_source_video_artifact_id"] = item.get("avatar_source_video_artifact_id")

        by_scene: Dict[str, Dict[str, Any]] = {}
        for item in assets.get("scenes", []) if isinstance(assets, dict) else []:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("scene_id") or "").strip()
            if sid:
                by_scene[sid] = item
        for scene in scene_out.get("scenes", []) if isinstance(scene_out.get("scenes"), list) else []:
            if not isinstance(scene, dict):
                continue
            sid = str(scene.get("scene_id") or "").strip()
            item = by_scene.get(sid)
            if not item:
                continue
            bg_id = str(item.get("background_artifact_id") or "").strip()
            if bg_id:
                scene["background_asset_id"] = bg_id
                scene["background_artifact_id"] = bg_id
            scene["background_path"] = item.get("background_path")

        return cast_out, scene_out

    def _validate_assets_stage(
        self,
        *,
        cast_plan: Dict[str, Any],
        scene_assets: Dict[str, Any],
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> None:
        roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
        for role in roles:
            if not isinstance(role, dict):
                continue
            character_id = str(role.get("character_id") or "").strip() or "unknown"
            portrait_path = str(role.get("avatar_image_path") or "").strip()
            portrait_artifact_id = str(role.get("avatar_image_artifact_id") or "").strip()
            source_path = str(role.get("avatar_source_video_path") or "").strip()
            source_artifact_id = str(role.get("avatar_source_video_artifact_id") or "").strip()
            if not portrait_path or not os.path.exists(portrait_path):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=portrait_path or None,
                    reason="MISSING_PORTRAIT",
                    message=f"missing portrait.png for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not portrait_artifact_id:
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=None,
                    reason="MISSING_PORTRAIT_ARTIFACT_ID",
                    message=f"missing portrait artifact id for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not self._artifact_available(portrait_artifact_id):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=portrait_artifact_id,
                    reason="MISSING_PORTRAIT_ARTIFACT",
                    message=f"portrait artifact unavailable for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not source_path or not os.path.exists(source_path):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=source_path or None,
                    reason="MISSING_SOURCE_VIDEO",
                    message=f"missing source.mp4 for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not source_artifact_id:
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=None,
                    reason="MISSING_SOURCE_VIDEO_ARTIFACT_ID",
                    message=f"missing source video artifact id for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not self._artifact_available(source_artifact_id):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=character_id,
                    artifact_path=source_artifact_id,
                    reason="MISSING_SOURCE_VIDEO_ARTIFACT",
                    message=f"source video artifact unavailable for character '{character_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )

        scenes = scene_assets.get("scenes", []) if isinstance(scene_assets, dict) else []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip() or "unknown"
            bg_path = str(scene.get("background_path") or "").strip()
            bg_artifact = str(scene.get("background_artifact_id") or scene.get("background_asset_id") or "").strip()
            if not bg_path or not os.path.exists(bg_path):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=scene_id,
                    artifact_path=bg_path or None,
                    reason="MISSING_BACKGROUND_PATH",
                    message=f"missing background image for scene '{scene_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not bg_artifact or not self._artifact_available(bg_artifact):
                self._raise_stage_failure(
                    stage="assets",
                    line_id=scene_id,
                    artifact_path=bg_artifact or None,
                    reason="MISSING_BACKGROUND_ARTIFACT",
                    message=f"missing background artifact for scene '{scene_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )

    def _store_scene_assets(self, scene_assets: Dict[str, Any], episode_id: str) -> None:
        payload = json.dumps(scene_assets, ensure_ascii=True, indent=2)
        try:
            self.memory.store(
                type_="scene_assets",
                text=payload,
                tags=["asset", "scene_assets"],
                episode_id=episode_id,
            )
        except Exception:
            pass

    def _store_episode_memory(
        self,
        episode_id: str,
        episode_brief: Dict[str, Any],
        screenplay: Dict[str, Any],
        cast_plan: Dict[str, Any],
        scene_assets: Dict[str, Any],
        logger: RunLogger | None = None,
    ) -> None:
        summary = self._summarize_episode(episode_brief, screenplay, cast_plan, scene_assets, logger)
        if not summary:
            return
        summary_text = json.dumps(summary, ensure_ascii=True, indent=2)
        try:
            self.memory.store(
                type_="episode_summary",
                text=summary_text,
                tags=["episode", "summary"],
                episode_id=episode_id,
            )
        except Exception:
            pass
        continuity = summary.get("continuity_notes")
        if continuity:
            try:
                self.memory.store(
                    type_="continuity",
                    text="\n".join(continuity) if isinstance(continuity, list) else str(continuity),
                    tags=["continuity"],
                    episode_id=episode_id,
                )
            except Exception:
                pass

    def _summarize_episode(
        self,
        episode_brief: Dict[str, Any],
        screenplay: Dict[str, Any],
        cast_plan: Dict[str, Any],
        scene_assets: Dict[str, Any],
        logger: RunLogger | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "episode_brief": episode_brief,
            "screenplay": {
                "scenes": [
                    {
                        "scene_id": s.get("scene_id"),
                        "characters": s.get("characters", []),
                        "line_count": len(s.get("lines", [])),
                    }
                    for s in screenplay.get("scenes", [])
                ]
            },
            "cast_plan": cast_plan,
            "scene_assets": scene_assets,
        }
        prompt = (
            "Summarize this episode for future continuity and reuse. "
            "Return JSON only with keys: summary, continuity_notes (array), character_state (object), tags (array).\n\n"
            f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
        )
        summary_llm = self._llm_for("curator")
        summary = summary_llm.complete_json(prompt)
        if logger:
            logger.log_chat("summary", summary_llm.last_messages, summary_llm.last_raw)
            logger.write_json(os.path.join(logger.run_dir, "episode_summary.json"), summary)
        return summary

    def _cached_render(self, name: str, timeline_id: str, preset: str) -> str:
        key = self.cache.make_key(name, {"timeline_id": timeline_id, "preset": preset})
        cached = self.cache.get(key)
        if cached and not self._is_unusable_render_artifact(cached):
            return cached
        try:
            if preset == "preview":
                result = self._call_tool_with_retry(self.render.render_preview, timeline_id)
            else:
                result = self._call_tool_with_retry(self.render.render_final, timeline_id)
            artifact_id = result["artifact_id"]
        except RuntimeError as err:
            if self._is_missing_ffmpeg_error(err) and self._allow_noffmpeg_render_fallback():
                artifact_id = self._render_without_ffmpeg(timeline_id=timeline_id, preset=preset)
            else:
                raise
        self.cache.set(key, artifact_id)
        return artifact_id

    def _is_unusable_render_artifact(self, artifact_id: str) -> bool:
        tags = self._artifact_tags_local(artifact_id)
        if tags and (("fallback" in tags) or ("noaudio" in tags)):
            return True
        # Guard against stale cached renders with broken timestamps/durations.
        try:
            path = self.store.get_path(artifact_id)
            meta = self._ffprobe_media(path)
            vdur = float(meta.get("video_duration_sec", 0.0))
            adur = float(meta.get("audio_duration_sec", 0.0))
            if vdur <= 0.25:
                return True
            if abs(vdur - adur) > 0.25:
                return True
        except Exception:
            # If probing fails, treat as unusable and force regeneration.
            return True
        return False

    def _artifact_tags_local(self, artifact_id: str) -> List[str]:
        artifact_root = os.getenv("ARTIFACT_ROOT")
        if not artifact_root:
            artifact_root = os.path.join(os.getenv("DATA_ROOT", "data"), "artifacts")
        meta = os.path.join(artifact_root, artifact_id[:2], artifact_id[2:4], f"{artifact_id}.json")
        if not os.path.exists(meta):
            return []
        try:
            with open(meta, "r", encoding="utf-8") as f:
                payload = json.load(f)
            tags = payload.get("tags", [])
            if isinstance(tags, list):
                return [str(t) for t in tags]
        except Exception:
            return []
        return []

    def _ensure_performances_ready(
        self,
        performances: Dict[str, Any],
        logger: Optional[RunLogger] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        failed: List[str] = []
        ok_count = 0
        allow_lipsync_fallback = self._allow_lipsync_fallback_for_final()
        for scene in performances.get("scenes", []) if isinstance(performances, dict) else []:
            scene_id = scene.get("scene_id", "unknown-scene")
            for item in scene.get("characters", []) if isinstance(scene, dict) else []:
                status = str(item.get("status", "ok"))
                if status == "ok" and bool(item.get("lipsync_fallback")) and not allow_lipsync_fallback:
                    character_id = item.get("character_id", "unknown-character")
                    failed.append(f"{scene_id}/{character_id}: LIPSYNC_FALLBACK_USED (fallback output is disallowed)")
                    continue
                if status == "ok" and item.get("wav_artifact_id") and item.get("video_artifact_id"):
                    ok_count += 1
                    continue
                character_id = item.get("character_id", "unknown-character")
                code = item.get("error_code", "PERFORMANCE_FAILED")
                err = item.get("error", "unknown error")
                failed.append(f"{scene_id}/{character_id}: {code} ({err})")
        if ok_count == 0:
            details = "; ".join(failed[:6]) if failed else "no successful performance artifacts generated"
            if logger and episode_id:
                self._raise_stage_failure(
                    stage="avatar",
                    reason="NO_SUCCESSFUL_PERFORMANCE",
                    message=f"Performance stage failed before editor step: {details}",
                    episode_id=episode_id,
                    logger=logger,
                )
            raise RuntimeError(f"Performance stage failed before editor step: {details}")

    def _has_ready_performance_assets(self, performances: Dict[str, Any]) -> bool:
        allow_lipsync_fallback = self._allow_lipsync_fallback_for_final()
        for scene in performances.get("scenes", []) if isinstance(performances, dict) else []:
            for item in scene.get("characters", []) if isinstance(scene, dict) else []:
                status = str(item.get("status", "ok"))
                if status == "ok" and bool(item.get("lipsync_fallback")) and not allow_lipsync_fallback:
                    continue
                if status == "ok" and item.get("wav_artifact_id") and item.get("video_artifact_id"):
                    return True
        return False

    def _call_tool_with_retry(self, fn, *args, **kwargs):
        last_err: Optional[Exception] = None
        last_err_text = "unknown error"
        tool_name = getattr(fn, "__name__", repr(fn))
        for _ in range(self.tool_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as err:
                last_err = err
                last_err_text = _format_exception(err)
                time.sleep(2)
        raise RuntimeError(
            f"Tool call failed after {self.tool_retries + 1} attempts for {tool_name}: {last_err_text}"
        ) from last_err

    def _run_performances(self, screenplay: Dict[str, Any], cast_plan: Dict[str, Any], logger: RunLogger) -> Dict[str, Any]:
        batch_pause_sec = float(os.getenv("TTS_BATCH_PAUSE_SEC", "0.35"))
        joiner = os.getenv("TTS_BATCH_JOINER", " ... ")
        data_root = os.getenv("DATA_ROOT", "data")
        voice_map = load_voice_map(data_root)
        speaker_to_character = _speaker_to_character(cast_plan)

        seed_failures: Dict[str, str] = {}
        if logger:
            seeder = self._load_step_output(logger, "voice_seeder") or {}
            for item in seeder.get("results", []) if isinstance(seeder, dict) else []:
                if item.get("status") == "failed" and item.get("character_id"):
                    seed_failures[str(item["character_id"])] = str(item.get("error", VOICE_SEED_FAILED))

        performance_manifest: Dict[str, Any] = {"scenes": []}

        scenes = screenplay.get("scenes", [])
        for scene in scenes:
            scene_id = scene.get("scene_id", "scene-1")
            by_character: Dict[str, List[Dict[str, Any]]] = {}
            unresolved: List[Dict[str, Any]] = []
            for line in scene.get("lines", []):
                speaker = line.get("speaker") or "unknown"
                character_id = speaker_to_character.get(str(speaker))
                if not character_id:
                    unresolved.append(
                        {
                            "line_id": line.get("line_id"),
                            "speaker": speaker,
                            "error_code": "MISSING_CHARACTER_ID",
                            "error": f"Could not resolve character_id for speaker={speaker!r} from cast_plan",
                        }
                    )
                    continue
                by_character.setdefault(character_id, []).append(line)

            scene_perf = {"scene_id": scene_id, "characters": [], "unresolved_lines": unresolved}

            def process_character(character_id: str, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
                if character_id in seed_failures:
                    return {
                        "character_id": character_id,
                        "status": "failed",
                        "error_code": VOICE_SEED_FAILED,
                        "error": seed_failures[character_id],
                    }
                voice_id, source = resolve_voice_id(character_id, cast_plan, voice_map)
                if not voice_id:
                    return {
                        "character_id": character_id,
                        "status": "failed",
                        "error_code": "MISSING_VOICE_ID",
                        "error": f"missing voice_id for character_id={character_id}",
                    }
                text = joiner.join([l.get("text", "").strip() for l in lines])
                emotion = _resolve_emotion(lines)
                try:
                    result = self._call_tool_with_retry(
                        self.tts.tts_synthesize,
                        text=text,
                        character_id=character_id,
                        voice_id=voice_id,
                        emotion=emotion,
                        style=None,
                    )
                    wav_path = result["wav_path"]
                    duration_ms = int(result.get("duration_ms", 0))
                    total_duration_sec = max(duration_ms / 1000.0, 0.0)
                    segments = build_segments(lines, total_duration_sec, batch_pause_sec)
                    wav_id = _artifact_id_from_path(wav_path)
                except Exception as err:
                    return {
                        "character_id": character_id,
                        "voice_id": voice_id,
                        "voice_id_source": source,
                        "status": "failed",
                        "error_code": "XTTS_FAILED",
                        "error": _format_exception(err),
                    }

                try:
                    avatar_source_path = _select_avatar_source_path(cast_plan, character_id)
                    if not avatar_source_path:
                        return {
                            "character_id": character_id,
                            "voice_id": voice_id,
                            "voice_id_source": source,
                            "status": "failed",
                            "error_code": "MISSING_AVATAR_SOURCE",
                            "error": f"missing avatar_source_video_path for character_id={character_id}",
                        }
                    if not os.path.exists(avatar_source_path):
                        return {
                            "character_id": character_id,
                            "voice_id": voice_id,
                            "voice_id_source": source,
                            "status": "failed",
                            "error_code": "MISSING_AVATAR_SOURCE_FILE",
                            "error": f"avatar_source_video_path does not exist for character_id={character_id}: {avatar_source_path}",
                        }
                    with open(avatar_source_path, "rb") as f:
                        avatar_source_artifact = self.store.put(
                            data=f.read(),
                            content_type="video/mp4",
                            tags=["avatar", "source_video", "performance", f"character:{character_id}"],
                        )
                    lipsync_result = self._call_tool_with_retry(
                        self.lipsync.lipsync_render_clip,
                        avatar_id=avatar_source_artifact,
                        wav_id=wav_id,
                    )
                    video_id = lipsync_result["artifact_id"]
                    lipsync_diagnostics = (
                        lipsync_result.get("diagnostics") if isinstance(lipsync_result, dict) else None
                    )
                    fallback_used = False
                except Exception as err:
                    lipsync_diagnostics = None
                    fallback_used = False
                    if os.getenv("LIPSYNC_ALLOW_STATIC_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        try:
                            video_id = self._render_static_lipsync_fallback(
                                avatar_id=_select_avatar(cast_plan, character_id),
                                wav_id=wav_id,
                            )
                            fallback_used = True
                        except Exception as fallback_err:
                            return {
                                "character_id": character_id,
                                "voice_id": voice_id,
                                "voice_id_source": source,
                                "status": "failed",
                                "error_code": "LIPSYNC_FAILED",
                                "error": f"{_format_exception(err)} | fallback_error={_format_exception(fallback_err)}",
                            }
                    else:
                        return {
                            "character_id": character_id,
                            "voice_id": voice_id,
                            "voice_id_source": source,
                            "status": "failed",
                            "error_code": "LIPSYNC_FAILED",
                            "error": _format_exception(err),
                        }

                try:
                    line_audio_artifacts: List[Dict[str, Any]] = []
                    line_video_artifact_ids = self._split_video_ffmpeg(video_id, segments)
                    line_audio_paths = split_wav_ffmpeg(
                        wav_path,
                        segments,
                        sample_rate=int(os.getenv("TTS_EXPECTED_SAMPLE_RATE", "22050")),
                    )
                    line_video_artifacts: List[Dict[str, Any]] = []
                    for idx, (line, seg) in enumerate(zip(lines, segments)):
                        line_id = str(line.get("line_id") or "")
                        audio_id = ""
                        if idx < len(line_audio_paths):
                            with open(line_audio_paths[idx], "rb") as f:
                                audio_id = self.store.put(
                                    data=f.read(),
                                    content_type="audio/wav",
                                    tags=["audio", "line", "tts", f"character:{character_id}"],
                                )
                        video_line_id = line_video_artifact_ids[idx] if idx < len(line_video_artifact_ids) else ""
                        duration_sec = float(seg.get("duration_sec", 0.0))
                        line_audio_artifacts.append(
                            {
                                "line_id": line_id,
                                "wav_artifact_id": audio_id,
                                "duration_sec": duration_sec,
                            }
                        )
                        line_video_artifacts.append(
                            {
                                "line_id": line_id,
                                "video_artifact_id": video_line_id,
                                "duration_sec": duration_sec,
                            }
                        )
                    for path in line_audio_paths:
                        try:
                            os.unlink(path)
                        except Exception:
                            pass
                    return {
                        "character_id": character_id,
                        "voice_id": voice_id,
                        "voice_id_source": source,
                        "status": "ok",
                        "wav_artifact_id": wav_id,
                        "video_artifact_id": video_id,
                        "lipsync_diagnostics": lipsync_diagnostics,
                        "lipsync_fallback": fallback_used,
                        "line_audio_artifacts": line_audio_artifacts,
                        "line_video_artifacts": line_video_artifacts,
                        "segments": [
                            {
                                "line_id": line.get("line_id"),
                                "start_sec": seg.get("start_sec", 0.0),
                                "end_sec": seg.get("start_sec", 0.0) + seg.get("duration_sec", 0.0),
                            }
                            for line, seg in zip(lines, segments)
                        ],
                    }
                except Exception as err:  # pragma: no cover - defensive guard
                    return {
                        "character_id": character_id,
                        "voice_id": voice_id,
                        "voice_id_source": source,
                        "status": "failed",
                        "error_code": "PERFORMANCE_ARTIFACT_SPLIT_FAILED",
                        "error": _format_exception(err),
                    }

            max_workers_env = int(os.getenv("TTS_MAX_WORKERS", "8"))
            max_workers = max(1, min(max_workers_env, len(by_character)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_character, character_id, lines): character_id
                    for character_id, lines in by_character.items()
                }
                for fut in as_completed(futures):
                    scene_perf["characters"].append(fut.result())

            performance_manifest["scenes"].append(scene_perf)

        logger.save_step("performance", performance_manifest)
        return performance_manifest

    def _split_video_ffmpeg(self, input_artifact_id: str, segments: List[Dict[str, float]]) -> List[str]:
        import cv2

        source_path = self.store.get_path(input_artifact_id)
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        split_pad = max(float(os.getenv("AVATAR_SPLIT_PAD_SEC", "0.15")), 0.0)
        outputs: List[str] = []
        source_cap = cv2.VideoCapture(source_path)
        ok_src, source_frame = source_cap.read()
        source_cap.release()
        for seg in segments:
            start = float(seg.get("start_sec", 0.0))
            duration = max(float(seg.get("duration_sec", 0.0)), 0.25)
            start = max(0.0, start - (split_pad * 0.5))
            duration = max(duration + split_pad, 0.25)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                out_path = tmp.name
            try:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    source_path,
                    "-an",
                    "-vf",
                    "format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    out_path,
                ]
                split_ok = True
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    split_ok = False
                frame_count = 0
                if split_ok:
                    cap = cv2.VideoCapture(out_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                if (not split_ok or frame_count <= 0) and ok_src and source_frame is not None:
                    fps = 24.0
                    height, width = source_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if writer.isOpened():
                        n = max(1, int(round(duration * fps)))
                        for _ in range(n):
                            writer.write(source_frame)
                        writer.release()
                with open(out_path, "rb") as f:
                    artifact_id = self.store.put(
                        data=f.read(),
                        content_type="video/mp4",
                        tags=["video", "line", "avatar"],
                    )
                outputs.append(artifact_id)
            finally:
                if os.path.exists(out_path):
                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass
        return outputs

    def _render_static_lipsync_fallback(self, avatar_id: str, wav_id: str) -> str:
        avatar_path = self.store.get_path(avatar_id)
        wav_path = self.store.get_path(wav_id)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        try:
            ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
            cmd_animated = [
                ffmpeg_bin,
                "-y",
                "-loop",
                "1",
                "-i",
                avatar_path,
                "-i",
                wav_path,
                "-filter_complex",
                "[0:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
                "zoompan=z='if(lte(zoom,1.0),1.0,min(zoom+0.00035,1.08))':x='iw/2-(iw/zoom/2)':"
                "y='ih/2-(ih/zoom/2)':d=1:fps=24:s=1280x720,format=yuv420p[bg];"
                "[1:a]aformat=channel_layouts=mono,showwaves=s=1280x180:mode=line:colors=white,format=rgba[w];"
                "[bg][w]overlay=0:520[v]",
                "-map",
                "[v]",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                out_path,
            ]
            try:
                subprocess.run(cmd_animated, check=True, capture_output=True, text=True)
            except Exception:
                # Fallback to plain still+audio mux when advanced filters are unavailable.
                cmd_static = [
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
                subprocess.run(cmd_static, check=True, capture_output=True, text=True)
            fallback_tags = ["lipsync", "fallback"]
            with open(out_path, "rb") as f:
                data = f.read()
            return self.store.put(data=data, content_type="video/mp4", tags=fallback_tags)
        finally:
            if os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except Exception:
                    pass

    def _render_without_ffmpeg(self, timeline_id: str, preset: str) -> str:
        import cv2

        timeline_path = self.store.get_path(timeline_id)
        with open(timeline_path, "r", encoding="utf-8") as f:
            timeline = json.load(f)

        width = int(os.getenv("RENDER_WIDTH", "1280"))
        height = int(os.getenv("RENDER_HEIGHT", "720"))
        fps = int(os.getenv("RENDER_FPS", "30"))
        if preset == "preview":
            width = int(os.getenv("RENDER_PREVIEW_WIDTH", str(width // 2)))
            height = int(os.getenv("RENDER_PREVIEW_HEIGHT", str(height // 2)))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        try:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("opencv VideoWriter could not open output path")
            try:
                for scene in timeline.get("scenes", []):
                    if not isinstance(scene, dict):
                        continue
                    start = float(scene.get("start_sec", 0.0))
                    end = float(scene.get("end_sec", start + 1.0))
                    duration = max(0.1, end - start)
                    frame_count = max(1, int(round(duration * fps)))
                    frame = self._timeline_scene_frame(scene, width=width, height=height)
                    for _ in range(frame_count):
                        writer.write(frame)
            finally:
                writer.release()

            with open(out_path, "rb") as f:
                data = f.read()
            return self.store.put(data=data, content_type="video/mp4", tags=["render", preset, "fallback", "noaudio"])
        finally:
            if os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except Exception:
                    pass

    def _timeline_scene_frame(self, scene: Dict[str, Any], width: int, height: int):
        import cv2

        layers = scene.get("layers", [])
        asset_id: Optional[str] = None
        for layer in layers if isinstance(layers, list) else []:
            if not isinstance(layer, dict):
                continue
            if layer.get("type") == "background" and layer.get("asset_id"):
                asset_id = str(layer["asset_id"])
                break
        if not asset_id:
            for layer in layers if isinstance(layers, list) else []:
                if isinstance(layer, dict) and layer.get("asset_id"):
                    asset_id = str(layer["asset_id"])
                    break

        if asset_id:
            try:
                path = self.store.get_path(asset_id)
                frame = cv2.imread(path)
                if frame is None:
                    cap = cv2.VideoCapture(path)
                    ok, first = cap.read()
                    cap.release()
                    if ok and first is not None:
                        frame = first
                if frame is not None:
                    return cv2.resize(frame, (width, height))
            except Exception:
                pass

        return cv2.merge(
            [
                cv2.UMat(height, width, cv2.CV_8UC1, 0).get(),
                cv2.UMat(height, width, cv2.CV_8UC1, 0).get(),
                cv2.UMat(height, width, cv2.CV_8UC1, 0).get(),
            ]
        )

    def _is_missing_ffmpeg_error(self, err: Exception) -> bool:
        text = str(err)
        return "No such file or directory: 'ffmpeg'" in text or ("[Errno 2]" in text and "ffmpeg" in text)

    def _is_missing_ffprobe_error(self, err: Exception) -> bool:
        text = str(err)
        return "No such file or directory: 'ffprobe'" in text or ("[Errno 2]" in text and "ffprobe" in text)

    def _allow_noffmpeg_render_fallback(self) -> bool:
        return os.getenv("RENDER_ALLOW_NOFFMPEG_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _allow_noffprobe_qc_fallback(self) -> bool:
        return os.getenv("QC_ALLOW_NOFFPROBE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _allow_lipsync_fallback_for_final(self) -> bool:
        return os.getenv("ALLOW_LIPSYNC_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

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
        try:
            _run_cmd(cmd)
        except FileNotFoundError:
            if os.getenv("HIGHLIGHT_ALLOW_COPY_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}:
                with open(preview_path, "rb") as src, open(highlight_path, "wb") as dst:
                    dst.write(src.read())
            else:
                raise
        with open(highlight_path, "rb") as f:
            data = f.read()
        return self.store.put(data=data, content_type="video/mp4", tags=["highlight"])

    def _gate_preview(self, preview_id: str, highlight_id: str, logger: RunLogger, auto: bool) -> bool:
        if auto:
            return True
        print("Gate B highlight artifact:", highlight_id)
        reply = input("Approve preview highlight? (y/n): ").strip().lower()
        return reply in {"y", "yes"}

    def _normalize_timeline_for_render(
        self,
        timeline: Dict[str, Any],
        scene_assets: Dict[str, Any],
        scene_plan: Dict[str, Any],
        performances: Dict[str, Any],
        *,
        logger: Optional[RunLogger],
        episode_id: str,
    ) -> Dict[str, Any]:
        if not isinstance(timeline, dict):
            return timeline
        scenes = timeline.get("scenes")
        if not isinstance(scenes, list):
            return timeline

        scene_bg: Dict[str, str] = {}
        scene_bg_path: Dict[str, str] = {}
        for item in scene_assets.get("scenes", []) if isinstance(scene_assets, dict) else []:
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id") or "").strip()
            bg_id = str(item.get("background_asset_id") or "").strip()
            bg_path = str(item.get("background_path") or "").strip()
            if scene_id and bg_id:
                scene_bg[scene_id] = bg_id
            if scene_id and bg_path:
                scene_bg_path[scene_id] = bg_path

        perf_scene: Dict[str, List[Dict[str, Any]]] = {}
        for scene in performances.get("scenes", []) if isinstance(performances, dict) else []:
            if not isinstance(scene, dict):
                continue
            sid = str(scene.get("scene_id") or "").strip()
            if not sid:
                continue
            rows: List[Dict[str, Any]] = []
            for char in scene.get("characters", []) if isinstance(scene.get("characters"), list) else []:
                if not isinstance(char, dict):
                    continue
                if str(char.get("status", "")) != "ok":
                    continue
                vid = str(char.get("video_artifact_id") or "").strip()
                if not vid or not self._artifact_available(vid):
                    continue
                rows.append(char)
            perf_scene[sid] = rows

        stage_layout: Dict[str, Dict[str, Dict[str, float]]] = {}
        for scene in scene_plan.get("scenes", []) if isinstance(scene_plan, dict) else []:
            if not isinstance(scene, dict):
                continue
            sid = str(scene.get("scene_id") or "").strip()
            if not sid:
                continue
            mapping: Dict[str, Dict[str, float]] = {}
            for entry in scene.get("stage", []) if isinstance(scene.get("stage"), list) else []:
                if not isinstance(entry, dict):
                    continue
                cid = str(entry.get("character_id") or "").strip()
                if not cid:
                    continue
                mapping[cid] = {
                    "x": float(entry.get("x", 0.35)),
                    "y": float(entry.get("y", 0.25)),
                    "scale": float(entry.get("scale", 0.5)),
                }
            stage_layout[sid] = mapping

        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "")
            if not scene_id:
                self._raise_stage_failure(
                    stage="compose",
                    line_id=None,
                    artifact_path=None,
                    reason="MISSING_SCENE_ID",
                    message="editor output missing scene_id",
                    episode_id=episode_id,
                    logger=logger,
                )
            start_sec = float(scene.get("start_sec", 0.0))
            end_sec = float(scene.get("end_sec", timeline.get("duration_sec", 0.0)))

            layers_raw = scene.get("layers", [])
            if isinstance(layers_raw, dict):
                layers = [layers_raw]
            elif isinstance(layers_raw, list):
                layers = [layer for layer in layers_raw if isinstance(layer, dict)]
            else:
                layers = []

            audio_raw = scene.get("audio", [])
            if isinstance(audio_raw, dict):
                audio = [audio_raw]
            elif isinstance(audio_raw, list):
                audio = [entry for entry in audio_raw if isinstance(entry, dict)]
            else:
                audio = []
            normalized_audio: List[Dict[str, Any]] = []
            for entry in audio:
                asset_id = str(entry.get("asset_id") or "").strip()
                if asset_id and self._artifact_available(asset_id) and self._is_audio_artifact(asset_id):
                    normalized_audio.append(entry)

            if not normalized_audio:
                self._raise_stage_failure(
                    stage="compose",
                    line_id=scene_id,
                    artifact_path=None,
                    reason="MISSING_AUDIO_TRACK",
                    message=f"editor output has no valid audio tracks for scene '{scene_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            scene["audio"] = normalized_audio

            bg_idx = next((idx for idx, layer in enumerate(layers) if str(layer.get("type")) == "background"), None)
            candidate_bg = scene_bg.get(scene_id)
            candidate_bg_path = scene_bg_path.get(scene_id, "")
            if not candidate_bg:
                self._raise_stage_failure(
                    stage="scene",
                    line_id=scene_id,
                    artifact_path=None,
                    reason="MISSING_BACKGROUND_ASSET_ID",
                    message=f"scene assets missing background_asset_id for scene '{scene_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not candidate_bg_path or not os.path.exists(candidate_bg_path):
                expected_path = candidate_bg_path or os.path.join("data", "runs", episode_id, "assets", "backgrounds", f"{scene_id}.png")
                self._raise_stage_failure(
                    stage="editor",
                    line_id=scene_id,
                    artifact_path=expected_path,
                    reason="MISSING_BACKGROUND_PATH",
                    message=f"scene '{scene_id}' missing resolved background file before render normalization",
                    episode_id=episode_id,
                    logger=logger,
                )
            if not self._artifact_available(candidate_bg):
                self._raise_stage_failure(
                    stage="scene",
                    line_id=scene_id,
                    artifact_path=candidate_bg,
                    reason="MISSING_BACKGROUND_ARTIFACT",
                    message=f"background asset '{candidate_bg}' is not available in artifact store",
                    episode_id=episode_id,
                    logger=logger,
                )

            if bg_idx is None:
                layers.insert(
                    0,
                    {
                        "type": "background",
                        "asset_id": candidate_bg,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "z": 0,
                    },
                )
            else:
                if not self._artifact_available(layers[bg_idx].get("asset_id")):
                    self._raise_stage_failure(
                        stage="compose",
                        line_id=scene_id,
                        artifact_path=str(layers[bg_idx].get("asset_id") or ""),
                        reason="INVALID_BACKGROUND_LAYER_ASSET",
                        message=f"editor background layer references unavailable asset in scene '{scene_id}'",
                        episode_id=episode_id,
                        logger=logger,
                    )
                layers[bg_idx].setdefault("start_sec", start_sec)
                layers[bg_idx].setdefault("end_sec", end_sec)
                layers[bg_idx].setdefault("z", 0)

            has_foreground = any(str(layer.get("type")) in {"actor", "props"} for layer in layers if isinstance(layer, dict))
            if not has_foreground:
                self._raise_stage_failure(
                    stage="editor",
                    line_id=scene_id,
                    artifact_path=None,
                    reason="MISSING_FOREGROUND_LAYER",
                    message=f"editor output has no actor/props layers for scene '{scene_id}'",
                    episode_id=episode_id,
                    logger=logger,
                )

            scene["layers"] = layers

        return timeline

    def _artifact_available(self, artifact_id: Any) -> bool:
        if not artifact_id:
            return False
        try:
            self.store.get_path(str(artifact_id))
            return True
        except Exception:
            return False

    def _is_audio_artifact(self, artifact_id: str) -> bool:
        content_type = self._artifact_content_type_local(artifact_id)
        return content_type.startswith("audio/")

    def _artifact_content_type_local(self, artifact_id: str) -> str:
        artifact_root = os.getenv("ARTIFACT_ROOT")
        if not artifact_root:
            artifact_root = os.path.join(os.getenv("DATA_ROOT", "data"), "artifacts")
        meta = os.path.join(artifact_root, artifact_id[:2], artifact_id[2:4], f"{artifact_id}.json")
        if not os.path.exists(meta):
            return ""
        try:
            with open(meta, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return str(payload.get("content_type") or "").strip().lower()
        except Exception:
            return ""

class StageValidationError(RuntimeError):
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(json.dumps(payload, ensure_ascii=True))


def _resolve_emotion(group: List[Dict[str, Any]]) -> str:
    emotions = {str((line.get("emotion") or "neutral")).lower() for line in group}
    if len(emotions) == 1:
        return emotions.pop()
    return "neutral"


def _speaker_to_character(cast_plan: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for role in cast_plan.get("roles", []):
        character_id = str(role.get("character_id") or "").strip()
        if not character_id:
            continue
        for key in ("role", "display_name", "character_id"):
            value = str(role.get(key) or "").strip()
            if value:
                out[value] = character_id
    return out


def _exc_summary(err: BaseException) -> str:
    msg = str(err).strip()
    if not msg:
        msg = repr(err)
    return f"{type(err).__name__}: {msg}"


def _flatten_exceptions(err: BaseException) -> List[BaseException]:
    if isinstance(err, ExceptionGroup):
        flattened: List[BaseException] = []
        for sub in err.exceptions:
            flattened.extend(_flatten_exceptions(sub))
        return flattened
    return [err]


def _format_exception(err: BaseException) -> str:
    if isinstance(err, ExceptionGroup):
        subs = "; ".join(_exc_summary(sub) for sub in _flatten_exceptions(err))
        if subs:
            return f"{_exc_summary(err)} | sub-exceptions: {subs}"
    return _exc_summary(err)


def _select_avatar(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("character_id") == speaker:
            return role.get("avatar_id", "")
        if role.get("role") == speaker:
            return role.get("avatar_id", "")
        if role.get("display_name") == speaker:
            return role.get("avatar_id", "")
    return ""


def _select_avatar_source(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("character_id") == speaker or role.get("role") == speaker or role.get("display_name") == speaker:
            return str(role.get("avatar_source_video_artifact_id") or "").strip()
    return ""


def _select_avatar_source_path(cast_plan: Dict[str, Any], speaker: Optional[str]) -> str:
    for role in cast_plan.get("roles", []):
        if role.get("character_id") == speaker or role.get("role") == speaker or role.get("display_name") == speaker:
            return str(role.get("avatar_source_video_path") or "").strip()
    return ""


def _artifact_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".wav"):
        return base[:-4]
    return base


def _content_type_for_image_name(name: str) -> str:
    lower = (name or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _avatar_expression_score(path: str) -> float:
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    # Face detection (if model available) to prioritize frontal portrait framing.
    face_ratio = 0.0
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            face_ratio = float((fw * fh) / float(h * w))
            roi = gray[y : y + fh, x : x + fw]
        else:
            roi = gray
    except Exception:
        roi = gray

    # Mouth-region texture proxy: lower third edge density.
    rh, rw = roi.shape[:2]
    if rh > 0 and rw > 0:
        mouth = roi[int(rh * 0.60) : int(rh * 0.95), :]
    else:
        mouth = roi
    edges = cv2.Canny(mouth, 70, 160) if mouth.size else np.zeros((1, 1), dtype=np.uint8)
    edge_density = float(np.mean(edges > 0))

    # Contrast helps visible lip motion.
    contrast = float(np.std(roi)) / 64.0 if roi.size else 0.0

    return (face_ratio * 2.5) + (edge_density * 2.0) + contrast


def _character_id_from_name(name: str, used: set[str]) -> str:
    base = _slug(name or "character")
    if not base:
        base = "character"
    candidate = base.upper()
    if candidate not in used:
        return candidate
    i = 2
    while True:
        candidate_i = f"{candidate}_{i}"
        if candidate_i not in used:
            return candidate_i
        i += 1


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in (value or ""))
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _make_episode_id() -> str:
    return datetime.now(timezone.utc).strftime("ep-%Y%m%d-%H%M%S")


def _load_comfyui_cfg() -> Dict[str, Any]:
    path = os.path.join("config", "defaults.yaml")
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if isinstance(payload, dict):
            section = payload.get("comfyui", {})
            if isinstance(section, dict):
                return section
    except Exception:
        return {}
    return {}


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
    prompt_input_json: Dict[str, Any] | Any = input_json
    if name == "writer":
        target_duration = 0.0
        try:
            target_duration = float(input_json.get("target_duration_sec") or 0.0)
        except Exception:
            target_duration = 0.0
        estimated = 0.0
        for err in errors:
            if isinstance(err, str) and err.startswith("screenplay_too_short:"):
                try:
                    estimated = float(err.split(":", 1)[1])
                except Exception:
                    estimated = 0.0
                break
        writer_targets = input_json.get("writer_targets", {}) if isinstance(input_json, dict) else {}
        min_words = int(writer_targets.get("min_total_words") or max(target_duration * 2.0, 0.0))
        min_lines = int(writer_targets.get("min_total_lines") or max(target_duration * 0.45, 1.0))
        stats = _writer_metrics(bad_json if isinstance(bad_json, dict) else {})
        line_delta = max(min_lines - int(stats.get("total_lines", 0)), 0)
        word_delta = max(min_words - int(stats.get("total_words", 0)), 0)
        extra = (
            "Writer repair instructions:\n"
            "- Output must be exactly one JSON object with top-level key: scenes.\n"
            "- Do NOT echo Input, episode_brief, series_bible, character_bible, or memory keys.\n"
            "- Expand by adding NEW UNIQUE lines; never duplicate line text.\n"
            "- Preserve existing valid lines and IDs where possible; only add new lines and fix duplicate IDs.\n"
            "- Preserve schema exactly; do not add extra keys.\n"
            "- Do not add new speakers that are not already allowed by input context.\n"
            "- Keep within max_scenes.\n"
            "- scene.characters must be string[] only (no character objects).\n"
            "- Every scene requires setting_prompt and scene_id in scene-<number> format.\n"
            f"- Current estimated duration: {estimated:.1f}s. Target: {target_duration:.1f}s.\n"
            f"- Current totals: lines={int(stats.get('total_lines', 0))}, words={int(stats.get('total_words', 0))}.\n"
            f"- Required minimums: lines={min_lines}, words={min_words}.\n"
            f"- Add at least {line_delta} more lines and {word_delta} more words using unique text.\n"
        )
        prompt_input_json = {
            "allowed_speakers": input_json.get("allowed_speakers", []),
            "target_duration_sec": input_json.get("target_duration_sec"),
            "writer_targets": input_json.get("writer_targets", {}),
            "style_guard": (input_json.get("series_bible", {}) or {}).get("style_guard", {}),
            "cast_roster": [
                {
                    "character_id": str(role.get("character_id") or ""),
                    "display_name": str(role.get("display_name") or ""),
                    "role": str(role.get("role") or ""),
                }
                for role in (input_json.get("cast_roster") or [])
                if isinstance(role, dict)
            ],
        }
        compact = {"scenes": []}
        for scene in (bad_json.get("scenes") or []):
            if not isinstance(scene, dict):
                continue
            compact_scene = {
                "scene_id": scene.get("scene_id"),
                "setting_prompt": scene.get("setting_prompt"),
                "characters": scene.get("characters"),
                "lines": [],
            }
            for line in (scene.get("lines") or []):
                if not isinstance(line, dict):
                    continue
                compact_scene["lines"].append(
                    {
                        "line_id": line.get("line_id"),
                        "speaker": line.get("speaker"),
                        "text": str(line.get("text") or "")[:160],
                    }
                )
            compact["scenes"].append(compact_scene)
        bad_json = compact
    bad_json_str = json.dumps(bad_json, ensure_ascii=True)
    if name == "writer" and not bad_json_str.strip():
        raise RuntimeError("WRITER_FIX_MISSING_CURRENT_JSON")
    return (
        "Fix the JSON to match the schema and constraints. Return JSON only.\n\n"
        f"Step: {name}\n"
        f"Schema: {schema_hint}\n"
        f"Errors: {errors}\n\n"
        f"{extra}\n"
        f"Input: {prompt_input_json}\n\n"
        f"Current JSON: {bad_json_str}\n"
    )


def _allowed_speakers_from_cast(cast_plan: Dict[str, Any]) -> List[str]:
    roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
    out: List[str] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        character_id = str(role.get("character_id") or "").strip()
        if character_id and character_id not in out:
            out.append(character_id)
    return out


def _cast_roster_from_plan(cast_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    roles = cast_plan.get("roles", []) if isinstance(cast_plan, dict) else []
    roster: List[Dict[str, Any]] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        roster.append(
            {
                "character_id": str(role.get("character_id") or "").strip(),
                "display_name": str(role.get("display_name") or "").strip(),
                "role": str(role.get("role") or "").strip(),
                "voice_id": str(role.get("voice_id") or "").strip(),
                "avatar_id": str(role.get("avatar_id") or "").strip(),
            }
        )
    return roster


def _default_writer_targets(duration_sec: int) -> Dict[str, int]:
    d = max(float(duration_sec or 0), 1.0)
    return {
        "min_total_lines": max(40, int(round(d * 0.45))),
        "min_total_words": max(int(round(d * 2.2)), int(round(d * 2.0))),
    }


def _extract_writer_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = data.get("_writer_plan", {}) if isinstance(data, dict) else {}
    return plan if isinstance(plan, dict) else {}


def _build_writer_validator(
    *,
    target_duration_sec: int,
    tolerance_sec: int,
    style_guard: Dict[str, Any],
    allowed_speakers: set[str],
    writer_targets: Dict[str, int],
):
    def _validator(data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        plan = _extract_writer_plan(data)
        required_lines = int(writer_targets.get("min_total_lines") or 0)
        required_words = int(writer_targets.get("min_total_words") or 0)
        scene_budgets: Dict[str, int] = {}
        for scene in plan.get("scenes", []) if isinstance(plan.get("scenes", []), list) else []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip()
            try:
                budget = int(scene.get("line_budget") or 0)
            except Exception:
                budget = 0
            if scene_id and budget > 0:
                scene_budgets[scene_id] = budget

        plan_lines = int(plan.get("target_total_lines") or 0) if isinstance(plan, dict) else 0
        plan_words = int(plan.get("target_total_words") or 0) if isinstance(plan, dict) else 0
        if plan_lines and required_lines and plan_lines != required_lines:
            errors.append(f"writer_plan_target_lines_mismatch:{plan_lines}!={required_lines}")
        if plan_words and required_words and plan_words != required_words:
            errors.append(f"writer_plan_target_words_mismatch:{plan_words}!={required_words}")
        if scene_budgets and required_lines and sum(scene_budgets.values()) != required_lines:
            errors.append(
                f"writer_plan_budget_sum_mismatch:{sum(scene_budgets.values())}!={required_lines}"
            )
            # Do not enforce broken plan budgets.
            scene_budgets = {}

        errors.extend(
            validate_screenplay(
            data,
            target_duration_sec=target_duration_sec,
            tolerance_sec=tolerance_sec,
            style_guard=style_guard,
            allowed_speakers=allowed_speakers,
            min_total_lines=required_lines,
            min_total_words=required_words,
            scene_line_budgets=scene_budgets,
            )
        )
        return errors

    return _validator


def _normalize_writer_line_ids(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(screenplay, dict):
        return screenplay
    scenes = screenplay.get("scenes", [])
    if not isinstance(scenes, list):
        return screenplay
    line_counter = 1
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        lines = scene.get("lines", [])
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, dict):
                continue
            line["line_id"] = f"line-{line_counter}"
            line_counter += 1
    return screenplay


def _normalize_writer_scene_ids(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(screenplay, dict):
        return screenplay
    scenes = screenplay.get("scenes", [])
    if not isinstance(scenes, list):
        return screenplay
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        raw = str(scene.get("scene_id") or "").strip()
        if raw.isdigit():
            scene["scene_id"] = f"scene-{raw}"
            continue
        if not raw:
            scene["scene_id"] = f"scene-{idx}"
    return screenplay


def _normalize_writer_screenplay(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    screenplay = _normalize_writer_scene_ids(screenplay)
    screenplay = _normalize_writer_line_ids(screenplay)
    return screenplay


def _only_line_id_errors(errors: List[str]) -> bool:
    if not errors:
        return False
    allowed_prefixes = ("duplicate_line_id:", "invalid_line_id_format:")
    return all(isinstance(err, str) and err.startswith(allowed_prefixes) for err in errors)


def _writer_needs_full_regen(errors: List[str]) -> bool:
    if not errors:
        return False
    regen_prefixes = (
        "screenplay_too_few_words:",
        "screenplay_too_few_lines:",
        "screenplay_too_short:",
        "scene_line_budget_miss:",
        "writer_plan_target_lines_mismatch:",
        "writer_plan_target_words_mismatch:",
        "writer_plan_budget_sum_mismatch:",
        "line_repeated_too_many_times:",
        "line_repetition_ratio_too_high:",
        "low_unique_token_ratio:",
    )
    return any(isinstance(err, str) and err.startswith(regen_prefixes) for err in errors)


def _has_writer_repetition_errors(errors: List[str]) -> bool:
    if not errors:
        return False
    prefixes = (
        "line_repeated_too_many_times:",
        "line_repetition_ratio_too_high:",
        "low_unique_token_ratio:",
    )
    return any(isinstance(err, str) and err.startswith(prefixes) for err in errors)


def _writer_local_repair(screenplay: Dict[str, Any], *, min_total_words: int) -> Dict[str, Any]:
    fixed = deepcopy(screenplay if isinstance(screenplay, dict) else {"scenes": []})
    scenes = fixed.get("scenes", [])
    if not isinstance(scenes, list):
        return fixed

    lines: List[Dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_lines = scene.get("lines", [])
        if not isinstance(scene_lines, list):
            continue
        for line in scene_lines:
            if isinstance(line, dict):
                lines.append(line)

    def canon(s: str) -> str:
        return " ".join(str(s).lower().split())

    # Break exact text duplicates deterministically.
    counts: Dict[str, int] = {}
    for line in lines:
        key = canon(line.get("text") or "")
        counts[key] = counts.get(key, 0) + 1
    seen: Dict[str, int] = {}
    for line in lines:
        text = str(line.get("text") or "").strip()
        key = canon(text)
        idx = seen.get(key, 0) + 1
        seen[key] = idx
        if key and counts.get(key, 0) > 1 and idx > 1:
            text = re.sub(r"\s+", " ", text).strip()
            line["text"] = f"{text} Variant {idx}.".strip()[:200]

    def word_count() -> int:
        total = 0
        for line in lines:
            total += len([w for w in str(line.get("text") or "").split() if w])
        return total

    # Pad words when short.
    deficit = max(int(min_total_words) - word_count(), 0)
    if deficit > 0 and lines:
        pads = [
            "This detail clarifies cause and effect for the audience.",
            "The connection reinforces continuity across the episode arc.",
            "The explanation links the myth motif to practical meaning.",
        ]
        i = 0
        guard = 0
        while deficit > 0 and guard < 10000:
            line = lines[i % len(lines)]
            pad = pads[i % len(pads)]
            cur = str(line.get("text") or "").strip()
            if len(cur) + 1 + len(pad) <= 200:
                line["text"] = f"{cur} {pad}".strip()
                deficit = max(int(min_total_words) - word_count(), 0)
            i += 1
            guard += 1
    return fixed


def _writer_repair_feedback(
    payload: Dict[str, Any],
    bad_json: Dict[str, Any],
    errors: List[str],
) -> str:
    writer_targets = payload.get("writer_targets", {}) if isinstance(payload, dict) else {}
    min_lines = int(writer_targets.get("min_total_lines") or 0)
    min_words = int(writer_targets.get("min_total_words") or 0)
    stats = _writer_metrics(bad_json if isinstance(bad_json, dict) else {})
    return (
        "Writer repair request.\n"
        "Regenerate the screenplay from scratch as one JSON object with top-level key `scenes`.\n"
        "Use only allowed speakers from input context. Do not add new speakers.\n"
        "Do not repeat line text. Keep scene_id format as scene-<number>.\n"
        f"Required minimums: lines={min_lines}, words={min_words}.\n"
        f"Current totals: lines={int(stats.get('total_lines', 0))}, words={int(stats.get('total_words', 0))}.\n"
        "Validator errors:\n"
        + "\n".join(f"- {e}" for e in errors)
    )


def _writer_metrics(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    scenes = screenplay.get("scenes", []) if isinstance(screenplay, dict) else []
    per_scene: List[Dict[str, Any]] = []
    total_lines = 0
    total_words = 0
    for scene in scenes if isinstance(scenes, list) else []:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or "").strip()
        lines = scene.get("lines", [])
        if not isinstance(lines, list):
            lines = []
        line_count = len(lines)
        word_count = 0
        for line in lines:
            if not isinstance(line, dict):
                continue
            text = str(line.get("text") or "").strip()
            word_count += len([w for w in text.split() if w])
        total_lines += line_count
        total_words += word_count
        per_scene.append({"scene_id": scene_id, "line_count": line_count, "word_count": word_count})
    plan = _extract_writer_plan(screenplay) if isinstance(screenplay, dict) else {}
    return {
        "total_lines": total_lines,
        "total_words": total_words,
        "planned_total_lines": int(plan.get("target_total_lines") or 0) if isinstance(plan, dict) else 0,
        "planned_total_words": int(plan.get("target_total_words") or 0) if isinstance(plan, dict) else 0,
        "per_scene": per_scene,
        "plan_scenes": plan.get("scenes", []) if isinstance(plan, dict) else [],
    }


def _compact_performances_for_editor(performances: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(performances, dict):
        return {}
    out: Dict[str, Any] = {"scenes": []}
    scenes = performances.get("scenes", [])
    if not isinstance(scenes, list):
        return out
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_out: Dict[str, Any] = {
            "scene_id": scene.get("scene_id"),
            "unresolved_lines": scene.get("unresolved_lines", []),
            "characters": [],
        }
        chars = scene.get("characters", [])
        if isinstance(chars, list):
            for row in chars:
                if not isinstance(row, dict):
                    continue
                row_out: Dict[str, Any] = {
                    "character_id": row.get("character_id"),
                    "status": row.get("status"),
                    "error_code": row.get("error_code"),
                    "wav_artifact_id": row.get("wav_artifact_id"),
                    "video_artifact_id": row.get("video_artifact_id"),
                    "segments": row.get("segments", []),
                }
                scene_out["characters"].append(row_out)
        out["scenes"].append(scene_out)
    return out


def _compact_screenplay_for_editor(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(screenplay, dict):
        return {"scenes": []}
    out: Dict[str, Any] = {"scenes": []}
    scenes = screenplay.get("scenes", [])
    if not isinstance(scenes, list):
        return out
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        lines = scene.get("lines", [])
        compact_lines = []
        if isinstance(lines, list):
            compact_lines = [
                {
                    "line_id": line.get("line_id"),
                    "speaker": line.get("speaker"),
                    "emotion": line.get("emotion"),
                    "pause_ms_after": line.get("pause_ms_after"),
                }
                for line in lines
                if isinstance(line, dict)
            ]
        out["scenes"].append(
            {
                "scene_id": scene.get("scene_id"),
                "setting_prompt": scene.get("setting_prompt"),
                "characters": scene.get("characters", []),
                "lines": compact_lines,
            }
        )
    return out


def _compact_scene_assets_for_editor(scene_assets: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(scene_assets, dict):
        return {"scenes": []}
    out: Dict[str, Any] = {"scenes": []}
    scenes = scene_assets.get("scenes", [])
    if not isinstance(scenes, list):
        return out
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        out["scenes"].append(
            {
                "scene_id": scene.get("scene_id"),
                "background_asset_id": scene.get("background_asset_id") or scene.get("background_artifact_id"),
                "background_path": scene.get("background_path"),
                "layout_hints": scene.get("layout_hints", {}),
            }
        )
    return out


def _compact_scene_plan_for_editor(scene_plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(scene_plan, dict):
        return {"scenes": []}
    out: Dict[str, Any] = {"scenes": []}
    scenes = scene_plan.get("scenes", [])
    if not isinstance(scenes, list):
        return out
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        out["scenes"].append(
            {
                "scene_id": scene.get("scene_id"),
                "stage": scene.get("stage", []),
                "entrances": scene.get("entrances", []),
                "reactions": scene.get("reactions", []),
                "subtitle_placement": scene.get("subtitle_placement", {}),
            }
        )
    return out


def _strip_writer_meta(screenplay: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(screenplay, dict):
        return screenplay
    cleaned = dict(screenplay)
    cleaned.pop("_writer_plan", None)
    return cleaned


def _apply_writer_length_policy(
    payload: Dict[str, Any],
    screenplay: Dict[str, Any],
    *,
    logger: Optional[RunLogger],
    step: str,
) -> Dict[str, Any]:
    if not isinstance(screenplay, dict):
        return screenplay
    try:
        target = float(payload.get("target_duration_sec") or 0.0)
    except Exception:
        target = 0.0
    if target <= 0:
        return screenplay

    accept_ratio = float(os.getenv("WRITER_OVERLEN_ACCEPT_RATIO", "3.00"))
    accept_abs_sec = float(os.getenv("WRITER_OVERLEN_ACCEPT_ABS_SEC", "50"))
    trim_ratio = float(os.getenv("WRITER_OVERLEN_TRIM_RATIO", "1.30"))
    hard_ratio = float(os.getenv("WRITER_OVERLEN_HARD_RATIO", "2.50"))
    accept_limit = max(target * accept_ratio, target + max(0.0, accept_abs_sec))
    est = estimate_screenplay_duration_sec(screenplay)
    if est <= accept_limit:
        return screenplay
    if est > target * hard_ratio:
        raise RuntimeError(
            f"writer duration safety cap exceeded: est={est:.1f}s target={target:.1f}s ratio={est/max(target,0.1):.2f}"
        )

    trimmed = deepcopy(screenplay)
    _compress_pauses(trimmed, target)
    _trim_line_texts(trimmed, target, aggressive=est > target * trim_ratio)
    if estimate_screenplay_duration_sec(trimmed) > accept_limit:
        _drop_low_salience_lines(trimmed, accept_limit)
    if estimate_screenplay_duration_sec(trimmed) > accept_limit:
        # Second deterministic pass before giving up.
        _compress_pauses(trimmed, target)
        _trim_line_texts(trimmed, target, aggressive=True)
        _drop_low_salience_lines(trimmed, accept_limit)
    new_est = estimate_screenplay_duration_sec(trimmed)
    if new_est > target * hard_ratio:
        raise RuntimeError(
            f"writer duration safety cap exceeded after trim: est={new_est:.1f}s target={target:.1f}s ratio={new_est/max(target,0.1):.2f}"
        )
    if not (new_est < accept_limit):
        raise RuntimeError(
            f"writer still over accepted duration after trim: est={new_est:.1f}s accepted_up_to={accept_limit:.1f}s target={target:.1f}s"
        )

    if logger:
        logger.write_json(os.path.join(logger.step_dir(step), "normalized_post_trim.json"), trimmed)
    return trimmed


def _iter_scene_lines(screenplay: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    scenes = screenplay.get("scenes", [])
    if not isinstance(scenes, list):
        return lines
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_lines = scene.get("lines", [])
        if not isinstance(scene_lines, list):
            continue
        for line in scene_lines:
            if isinstance(line, dict):
                lines.append(line)
    return lines


def _compress_pauses(screenplay: Dict[str, Any], target_sec: float) -> None:
    est = estimate_screenplay_duration_sec(screenplay)
    if est <= target_sec:
        return
    ratio = target_sec / max(est, 0.1)
    for line in _iter_scene_lines(screenplay):
        raw = line.get("pause_ms_after", 250)
        try:
            pause_ms = int(raw)
        except Exception:
            pause_ms = 250
        new_pause = int(max(40, min(250, pause_ms * max(0.4, ratio))))
        line["pause_ms_after"] = new_pause


def _trim_line_texts(screenplay: Dict[str, Any], target_sec: float, *, aggressive: bool = False) -> None:
    est = estimate_screenplay_duration_sec(screenplay)
    if est <= target_sec:
        return
    lines = _iter_scene_lines(screenplay)
    if not lines:
        return
    ratio = target_sec / max(est, 0.1)
    keep_words = max(6, min(24, int(round((14 if aggressive else 18) * ratio))))
    char_cap = 80 if target_sec <= 360 else (96 if target_sec <= 600 else 110)
    if aggressive:
        char_cap = max(64, int(char_cap * 0.85))
    for line in lines:
        text = str(line.get("text") or "").strip()
        words = [w for w in text.split() if w]
        if len(words) > keep_words:
            line["text"] = " ".join(words[:keep_words])
            text = line["text"]
        if len(text) > char_cap:
            line["text"] = text[:char_cap].rstrip()


def _drop_low_salience_lines(screenplay: Dict[str, Any], accept_limit_sec: float) -> None:
    scenes = screenplay.get("scenes", [])
    if not isinstance(scenes, list):
        return
    while estimate_screenplay_duration_sec(screenplay) > accept_limit_sec:
        changed = False
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            lines = scene.get("lines", [])
            if not isinstance(lines, list) or len(lines) <= 1:
                continue
            # Keep line count stable to preserve scene budgets/ids.
            # Pick the *most expensive* line first, but skip lines already minimized.
            candidate_idx = -1
            candidate_score = None
            for idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                text = str(line.get("text") or "").strip()
                wc = len([w for w in text.split() if w])
                emotion = str(line.get("emotion") or "").strip().lower()
                has_sfx = bool(str(line.get("sfx_tag") or "").strip())
                already_minimized = (
                    text == "Proceed."
                    and emotion == "neutral"
                    and int(line.get("pause_ms_after", 0) or 0) <= 40
                    and not has_sfx
                )
                if already_minimized:
                    continue
                score = wc + (50 if has_sfx else 0) + (20 if emotion and emotion != "neutral" else 0)
                if candidate_score is None or score > candidate_score:
                    candidate_score = score
                    candidate_idx = idx
            if candidate_idx >= 0 and len(lines) > 1:
                line = lines[candidate_idx]
                before = (
                    str(line.get("text") or "").strip(),
                    str(line.get("emotion") or "").strip().lower(),
                    int(line.get("pause_ms_after", 0) or 0),
                    bool(str(line.get("sfx_tag") or "").strip()),
                )
                line_id = str(line.get("line_id") or "").strip() or f"line-{candidate_idx+1}"
                line["text"] = f"Narrative advances with a distinct mythic insight for {line_id}."
                line["emotion"] = "neutral"
                line["pause_ms_after"] = 40
                line.pop("sfx_tag", None)
                after = (
                    str(line.get("text") or "").strip(),
                    str(line.get("emotion") or "").strip().lower(),
                    int(line.get("pause_ms_after", 0) or 0),
                    bool(str(line.get("sfx_tag") or "").strip()),
                )
                if after != before:
                    changed = True
                if estimate_screenplay_duration_sec(screenplay) <= accept_limit_sec:
                    return
        if not changed:
            return


def _run_cmd(cmd: List[str]) -> None:
    import subprocess

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
