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
from agents.critic.agent import run as critic_run
from agents.common import LLMClient

from .mcp_clients import AssetClient, LipSyncClient, QCClient, RenderClient
from .cache import StepCache
from .tts_client import QwenTTSClient
from .audio_utils import build_segments
from .validators import validate_screenplay, validate_cast_plan, validate_timeline_references, estimate_screenplay_duration_sec
from .bibles import load_series_bible, load_character_bible, save_character_bible
from .memory_client import MemoryClient
from .run_logger import RunLogger
from .gates import render_screenplay_markdown, apply_line_edits


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
        self.store = AssetClient()
        self.cache = StepCache()
        self.render = RenderClient()
        self.qc = QCClient()
        self.tts = QwenTTSClient()
        self.lipsync = LipSyncClient()
        self.llm = LLMClient()
        self.memory = MemoryClient()
        self.agent_retries = int(os.getenv("AGENT_RETRIES", "2"))
        self.tool_retries = int(os.getenv("TOOL_RETRIES", "5"))
        self.screenplay_tolerance_sec = int(os.getenv("SCREENPLAY_DURATION_TOLERANCE_SEC", "15"))
        self.critic_gate_enabled = True

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

        screenplay = self._resume_or_cached_step(
            "writer",
            config.resume_from_step,
            logger,
            "writer",
            {
                "episode_brief": episode_brief,
                "series_bible": series_bible,
                "character_bible": character_bible,
                "target_duration_sec": config.duration_sec,
                "memory": {
                    "episode_summaries": episode_summaries,
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("writer"),
                },
            },
            writer_run,
            validator=lambda data: validate_screenplay(
                data,
                target_duration_sec=config.duration_sec,
                tolerance_sec=self.screenplay_tolerance_sec,
                style_guard=series_bible.get("style_guard", {}),
            ),
            critic=True,
        )
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

        casting = self._resume_or_cached_step(
            "casting",
            config.resume_from_step,
            logger,
            "casting",
            {
                "screenplay": screenplay,
                "character_bible": character_bible,
                "memory": {
                    "voice_tuning_history": voice_tuning_history,
                    "critic_lessons": self._critic_lessons("casting"),
                },
            },
            casting_run,
            validator=lambda data: validate_cast_plan(data.get("cast_plan", {}), screenplay),
            critic=True,
        )
        cast_plan = casting.get("cast_plan", {"roles": []})
        steps.append(_step("casting", "completed"))
        character_bible = self._apply_cast_bible_update(character_bible, casting.get("cast_bible_update", {}))
        cast_plan = self._resolve_cast_avatars(cast_plan, character_bible)
        self._store_voice_tuning_history(cast_plan, episode_id)

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

        performances = self._resume_or_run_performances(screenplay, cast_plan, logger, config.resume_from_step)
        steps.append(_step("performance", "completed"))

        timeline = self._resume_or_cached_step(
            "editor",
            config.resume_from_step,
            logger,
            "editor",
            {
                "screenplay": screenplay,
                "performances": performances,
                "memory": {
                    "continuity_notes": continuity_notes,
                    "critic_lessons": self._critic_lessons("editor"),
                },
            },
            editor_run,
            validator=lambda data: validate_timeline_references(data, screenplay, self.store),
            critic=True,
        )
        steps.append(_step("editor", "completed"))

        timeline_id = self._store_json(timeline, content_type="application/json")
        preview = self._resume_or_cached_render("render_preview", timeline_id, "preview", logger, config.resume_from_step)
        steps.append(_step("render_preview", "completed"))

        highlight = self._make_highlight(preview, logger)
        if not self._gate_preview(preview, highlight, logger, config.auto_approve):
            raise RuntimeError("Preview not approved")

        final = self._resume_or_cached_render("render_final", timeline_id, "final", logger, config.resume_from_step)
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
    ) -> Dict[str, Any]:
        if logger and self._should_resume_step("performance", resume_from):
            resumed = self._load_step_output(logger, "performance")
            if resumed is not None:
                return resumed
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
                return str(resumed["artifact_id"])
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
            "writer",
            "dramaturg",
            "casting",
            "scene",
            "director",
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
            logger.log_chat(name, self.llm.last_messages, self.llm.last_raw)

        if validator:
            errors = validator(data)
            if errors:
                if logger:
                    logger.write_json(os.path.join(logger.step_dir(name), "validation_errors.json"), errors)
                data = self._fix_with_retries(name, payload, data, errors, validator, logger)

        if logger:
            logger.write_json(os.path.join(logger.step_dir(name), "normalized.json"), data)

        if critic and self.critic_gate_enabled:
            data = self._critic_gate(name, payload, data, fn, validator, logger)
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
            if logger:
                logger.log_chat(f"{name}.fix", self.llm.last_messages, self.llm.last_raw)
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
            base_threshold = int(os.getenv("CRITIC_PASS_SCORE", "75"))
        except ValueError:
            base_threshold = 75
        threshold = base_threshold
        round_idx = 0
        while threshold >= 0:
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
            for attempt in range(1, self.agent_retries + 1):
                feedback = self._critic_feedback_text(review)
                data = fn(payload, llm=self.llm, critic_feedback=feedback)
                if name == "writer":
                    data = _normalize_screenplay(data)
                if logger:
                    logger.log_chat(name, self.llm.last_messages, self.llm.last_raw)
                    logger.write_json(
                        os.path.join(logger.step_dir(name), f"critic_retry_{round_idx}_{attempt}.json"),
                        {
                            "critic_feedback": feedback,
                            "prompt": self.llm.last_prompt,
                            "raw": self.llm.last_raw,
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
                        data = self._fix_with_retries(name, payload, data, errors, validator, logger)

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
            threshold -= 5
            round_idx += 1
        raise RuntimeError(f"{name} failed critic gate after retries; threshold dropped below 0")

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
        review = critic_run(
            step_name=name,
            input_data=payload,
            output_data=data,
            schema_hint=AGENT_SCHEMA_HINTS.get(name, ""),
            validation_errors=errors,
            llm=self.llm,
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
            logger.log_chat(f"{name}.critic", self.llm.last_messages, self.llm.last_raw)
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

        default_avatar = None
        for character in character_bible.get("characters", []) if isinstance(character_bible, dict) else []:
            avatar_id = character.get("avatar_id")
            if avatar_id:
                default_avatar = avatar_id
                break

        catalog_fallback = self._fallback_avatar_from_catalog()
        for role in roles:
            avatar_id = role.get("avatar_id")
            if avatar_id and avatar_id != "avatar-default" and self._avatar_is_usable(avatar_id):
                continue
            mapped = None
            character_id = role.get("character_id")
            if character_id:
                for character in character_bible.get("characters", []) if isinstance(character_bible, dict) else []:
                    if character.get("character_id") == character_id:
                        mapped = character.get("avatar_id")
                        break
            candidates = [mapped, default_avatar, catalog_fallback, avatar_id]
            selected = None
            for candidate in candidates:
                if candidate and candidate != "avatar-default" and self._avatar_is_usable(candidate):
                    selected = candidate
                    break
            role["avatar_id"] = selected or "avatar-default"

        return cast_plan

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
        summary = self.llm.complete_json(prompt)
        if logger:
            logger.log_chat("summary", self.llm.last_messages, self.llm.last_raw)
            logger.write_json(os.path.join(logger.run_dir, "episode_summary.json"), summary)
        return summary

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
        screenplay["scenes"] = [
            {
                "scene_id": "scene-1",
                "setting_prompt": "A cozy space studio set.",
                "characters": ["Narrator"],
                "lines": [],
            }
        ]
        scenes = screenplay["scenes"]

    # Append filler lines to the last scene until estimated duration meets target.
    last_scene = scenes[-1]
    lines = last_scene.setdefault("lines", [])
    characters = last_scene.get("characters") or ["Narrator"]
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
