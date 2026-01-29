"""Hard validators for pipeline artifacts."""
import os
from typing import Any, Dict, List, Set

from mcp_servers.assets.artifact_store import ArtifactStore


def estimate_screenplay_duration_sec(screenplay: Dict[str, Any], words_per_sec: float = 2.0) -> float:
    total = 0.0
    for scene in screenplay.get("scenes", []):
        for line in scene.get("lines", []):
            text = (line.get("text") or "").strip()
            words = len(text.split())
            line_sec = max(words / max(words_per_sec, 0.1), 0.3)
            pause_ms = float(line.get("pause_ms_after", 0))
            total += line_sec + (pause_ms / 1000.0)
    return total


def validate_screenplay(
    screenplay: Dict[str, Any],
    target_duration_sec: int | None,
    tolerance_sec: int = 15,
    style_guard: Dict[str, Any] | None = None,
) -> List[str]:
    errors: List[str] = []
    scenes = screenplay.get("scenes") or []
    scene_ids: Set[str] = set()
    line_ids: Set[str] = set()

    max_line_len = None
    max_scenes = None
    forbidden = []
    if style_guard:
        max_line_len = style_guard.get("max_line_length_chars")
        max_scenes = style_guard.get("max_scenes")
        forbidden = [s.lower() for s in style_guard.get("forbidden_content", [])]

    if max_scenes is not None and len(scenes) > int(max_scenes):
        errors.append(f"too_many_scenes:{len(scenes)}")

    for scene in scenes:
        scene_id = (scene.get("scene_id") or "").strip()
        if not scene_id:
            errors.append("missing_scene_id")
        elif scene_id in scene_ids:
            errors.append(f"duplicate_scene_id:{scene_id}")
        else:
            scene_ids.add(scene_id)

        characters = scene.get("characters") or []
        char_set = set()
        for c in characters:
            if isinstance(c, dict):
                value = c.get("character_id") or c.get("name") or c.get("id")
            else:
                value = c
            if value:
                char_set.add(value)
        for line in scene.get("lines", []):
            line_id = (line.get("line_id") or "").strip()
            if not line_id:
                errors.append("missing_line_id")
            elif line_id in line_ids:
                errors.append(f"duplicate_line_id:{line_id}")
            else:
                line_ids.add(line_id)

            speaker = (line.get("speaker") or "").strip()
            if not speaker:
                errors.append("missing_speaker")
            elif characters and speaker not in char_set:
                errors.append(f"speaker_not_in_scene_characters:{speaker}")

            text = (line.get("text") or "").strip()
            if not text:
                errors.append("missing_line_text")
            if max_line_len is not None and len(text) > int(max_line_len):
                errors.append(f"line_too_long:{line_id}")
            if forbidden:
                lower = text.lower()
                for term in forbidden:
                    if term and term in lower:
                        errors.append(f"forbidden_content:{term}")
                        break

    if target_duration_sec:
        est = estimate_screenplay_duration_sec(screenplay)
        if est < max(target_duration_sec - tolerance_sec, 0):
            errors.append(f"screenplay_too_short:{est:.1f}")
        if est > target_duration_sec + tolerance_sec:
            errors.append(f"screenplay_too_long:{est:.1f}")

    return errors


def validate_cast_plan(cast_plan: Dict[str, Any], screenplay: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    roles = cast_plan.get("roles") or []
    role_names = {r.get("role") for r in roles if r.get("role")}

    speakers = set()
    for scene in screenplay.get("scenes", []):
        for line in scene.get("lines", []):
            speaker = line.get("speaker")
            if speaker:
                speakers.add(speaker)

    missing = sorted([s for s in speakers if s not in role_names])
    if missing:
        errors.append(f"missing_cast_for_speakers:{','.join(missing)}")
    return errors


def validate_timeline_references(
    timeline: Dict[str, Any],
    screenplay: Dict[str, Any],
    store: ArtifactStore,
) -> List[str]:
    errors: List[str] = []
    scene_ids = {s.get("scene_id") for s in screenplay.get("scenes", []) if s.get("scene_id")}

    for scene in timeline.get("scenes", []):
        scene_id = scene.get("scene_id")
        if scene_id and scene_id not in scene_ids:
            errors.append(f"timeline_unknown_scene:{scene_id}")
        for layer in scene.get("layers", []):
            asset_id = layer.get("asset_id")
            if not _asset_exists(asset_id, store):
                errors.append(f"missing_asset:{asset_id}")
        for audio in scene.get("audio", []):
            asset_id = audio.get("asset_id")
            if not _asset_exists(asset_id, store):
                errors.append(f"missing_audio_asset:{asset_id}")
    return errors


def _asset_exists(asset_id: Any, store: ArtifactStore) -> bool:
    if not asset_id:
        return False
    if isinstance(asset_id, str) and os.path.exists(asset_id):
        return True
    try:
        store.get_path(str(asset_id))
        return True
    except FileNotFoundError:
        return False
