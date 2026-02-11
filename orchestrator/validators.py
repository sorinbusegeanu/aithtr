"""Hard validators for pipeline artifacts."""
import os
import re
from collections import Counter
from typing import Any, Dict, List, Protocol, Set


class AssetStoreLike(Protocol):
    def get_path(self, artifact_id: str) -> str:
        ...


def estimate_screenplay_duration_sec(screenplay: Dict[str, Any], words_per_sec: float = 2.0) -> float:
    total = 0.0
    for scene in screenplay.get("scenes", []):
        for line in scene.get("lines", []):
            text = _as_text(line.get("text"))
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
    allowed_speakers: Set[str] | None = None,
    min_total_lines: int | None = None,
    min_total_words: int | None = None,
    scene_line_budgets: Dict[str, int] | None = None,
) -> List[str]:
    errors: List[str] = []
    scenes = screenplay.get("scenes") or []
    scene_ids: Set[str] = set()
    line_ids: Set[str] = set()
    line_texts: List[str] = []
    token_uniqueness: List[float] = []
    total_words = 0
    total_lines = 0
    line_canonical_no_punct: List[str] = []
    min_words_per_line = int(os.getenv("SCREENPLAY_MIN_WORDS_PER_LINE", "5"))
    min_chars_per_line = int(os.getenv("SCREENPLAY_MIN_CHARS_PER_LINE", "20"))

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
        scene_id = _as_text(scene.get("scene_id"))
        if not scene_id:
            errors.append("missing_scene_id")
        elif not re.match(r"^scene-\d+$", scene_id):
            errors.append(f"invalid_scene_id_format:{scene_id}")
        elif scene_id in scene_ids:
            errors.append(f"duplicate_scene_id:{scene_id}")
        else:
            scene_ids.add(scene_id)

        setting_prompt = _as_text(scene.get("setting_prompt"))
        if not setting_prompt:
            errors.append(f"missing_setting_prompt:{scene_id or 'unknown'}")

        characters = scene.get("characters")
        if not isinstance(characters, list):
            errors.append(f"characters_not_string_array:{scene_id or 'unknown'}")
            characters = []
        elif any(not isinstance(c, str) or not c.strip() for c in characters):
            errors.append(f"characters_not_string_array:{scene_id or 'unknown'}")
        if allowed_speakers:
            for c in characters:
                if isinstance(c, str) and c.strip() and c.strip() not in allowed_speakers:
                    errors.append(f"scene_character_not_in_allowed_cast:{c.strip()}")

        char_set = set()
        for c in characters:
            value = c if isinstance(c, str) else None
            if value:
                char_set.add(value.strip())
        for line in scene.get("lines", []):
            total_lines += 1
            line_id = _as_text(line.get("line_id"))
            if not line_id:
                errors.append("missing_line_id")
            elif not re.match(r"^[A-Za-z0-9_.-]+$", line_id):
                errors.append(f"invalid_line_id_format:{line_id}")
            elif line_id in line_ids:
                errors.append(f"duplicate_line_id:{line_id}")
            else:
                line_ids.add(line_id)

            speaker = _as_text(line.get("speaker"))
            if not speaker:
                errors.append("missing_speaker")
            elif characters and speaker not in char_set:
                errors.append(f"speaker_not_in_scene_characters:{speaker}")
            elif allowed_speakers and speaker not in allowed_speakers:
                errors.append(f"speaker_not_in_allowed_cast:{speaker}")

            text = _as_text(line.get("text"))
            if not text:
                errors.append("missing_line_text")
            else:
                canonical = " ".join(text.lower().split())
                line_texts.append(canonical)
                canon_no_punct = re.sub(r"[^a-z0-9\s]", "", canonical)
                canon_no_punct = " ".join(canon_no_punct.split())
                if canon_no_punct:
                    line_canonical_no_punct.append(canon_no_punct)
                tokens = [t for t in re.split(r"\W+", canonical) if t]
                total_words += len(tokens)
                if tokens:
                    token_uniqueness.append(len(set(tokens)) / float(len(tokens)))
                else:
                    token_uniqueness.append(0.0)
                if len(tokens) < min_words_per_line or len(text) < min_chars_per_line:
                    errors.append(f"line_too_short_content:{line_id}")
                # Filler/degenerate lines are rejected hard.
                if canonical in {"proceed.", "continue.", "...", "okay.", "next."}:
                    errors.append(f"filler_line_text:{line_id}")
                if re.match(r"^[a-z][a-z0-9 _-]{0,40} continues the mythic narrative\.?$", canonical):
                    errors.append(f"filler_line_text:{line_id}")
                # Reject speaker labels embedded into line text, e.g. "Dr. Brainy: ...".
                if re.match(r"^[A-Za-z][A-Za-z0-9 _-]{0,40}:\s", text):
                    errors.append(f"speaker_label_in_text:{line_id}")
            if max_line_len is not None and len(text) > int(max_line_len):
                errors.append(f"line_too_long:{line_id}")
            if forbidden:
                lower = text.lower()
                for term in forbidden:
                    if term and term in lower:
                        errors.append(f"forbidden_content:{term}")
                        break

        if scene_line_budgets and scene_id in scene_line_budgets:
            budget = int(scene_line_budgets.get(scene_id) or 0)
            if budget > 0:
                scene_lines = scene.get("lines", []) if isinstance(scene.get("lines", []), list) else []
                if len(scene_lines) < budget:
                    errors.append(f"scene_line_budget_miss:{scene_id}:{len(scene_lines)}<{budget}")

    if target_duration_sec:
        est = estimate_screenplay_duration_sec(screenplay)
        if est < max(target_duration_sec - tolerance_sec, 0):
            errors.append(f"screenplay_too_short:{est:.1f}")
    if min_total_lines is not None and total_lines < int(min_total_lines):
        errors.append(f"screenplay_too_few_lines:{total_lines}")
    if min_total_words is not None and total_words < int(min_total_words):
        errors.append(f"screenplay_too_few_words:{total_words}")

    if line_texts:
        counts = Counter(line_texts)
        total = len(line_texts)
        max_repeat = max(counts.values())
        repeated_fraction = max_repeat / float(total)
        repeat_limit = max(int(os.getenv("SCREENPLAY_MAX_IDENTICAL_LINE_REPEAT", "2")), 1)
        repeated_fraction_limit = float(os.getenv("SCREENPLAY_MAX_IDENTICAL_LINE_RATIO", "0.35"))
        if max_repeat > repeat_limit:
            errors.append(f"line_repeated_too_many_times:{max_repeat}")
        if repeated_fraction > repeated_fraction_limit:
            errors.append(f"line_repetition_ratio_too_high:{repeated_fraction:.3f}")
    if line_canonical_no_punct:
        norm_counts = Counter(line_canonical_no_punct)
        max_norm_repeat = max(norm_counts.values())
        norm_repeat_limit = max(int(os.getenv("SCREENPLAY_MAX_NORMALIZED_LINE_REPEAT", "2")), 1)
        if max_norm_repeat > norm_repeat_limit:
            errors.append(f"line_repeated_too_many_times_normalized:{max_norm_repeat}")
    if token_uniqueness:
        avg_unique_ratio = sum(token_uniqueness) / float(len(token_uniqueness))
        min_unique_ratio = float(os.getenv("SCREENPLAY_MIN_UNIQUE_TOKEN_RATIO", "0.45"))
        if avg_unique_ratio < min_unique_ratio:
            errors.append(f"low_unique_token_ratio:{avg_unique_ratio:.3f}")

    return errors


def validate_cast_plan(cast_plan: Dict[str, Any], screenplay: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    roles = cast_plan.get("roles") or []
    character_ids: Set[str] = set()
    role_names = {r.get("role") for r in roles if r.get("role")}
    display_names = {r.get("display_name") for r in roles if r.get("display_name")}

    for role in roles:
        character_id = str(role.get("character_id") or "").strip()
        display_name = str(role.get("display_name") or "").strip()
        voice_id = str(role.get("voice_id") or "").strip()
        if not character_id:
            errors.append("missing_character_id")
        elif character_id in character_ids:
            errors.append(f"duplicate_character_id:{character_id}")
        else:
            character_ids.add(character_id)
        if not display_name:
            errors.append(f"missing_display_name:{character_id or role.get('role') or 'unknown'}")
        if not voice_id:
            errors.append(f"missing_voice_id:{character_id or role.get('role') or 'unknown'}")

    speakers = set()
    for scene in screenplay.get("scenes", []):
        for line in scene.get("lines", []):
            speaker = _as_text(line.get("speaker"))
            if speaker:
                speakers.add(speaker)

    missing = sorted([s for s in speakers if s not in role_names and s not in display_names and s not in character_ids])
    if missing:
        errors.append(f"missing_cast_for_speakers:{','.join(missing)}")
    return errors


def validate_cast_plan_contract(
    cast_plan: Dict[str, Any],
    required_characters: List[str] | None = None,
) -> List[str]:
    errors: List[str] = []
    roles = cast_plan.get("roles") or []
    character_ids: Set[str] = set()
    avatar_ids: Set[str] = set()
    aliases: Set[str] = set()
    for role in roles:
        if not isinstance(role, dict):
            errors.append("invalid_role_entry")
            continue
        character_id = str(role.get("character_id") or "").strip()
        display_name = str(role.get("display_name") or "").strip()
        role_name = str(role.get("role") or "").strip()
        voice_id = str(role.get("voice_id") or "").strip()
        avatar_id = str(role.get("avatar_id") or "").strip()
        if not character_id:
            errors.append("missing_character_id")
        elif character_id in character_ids:
            errors.append(f"duplicate_character_id:{character_id}")
        else:
            character_ids.add(character_id)
        if not display_name:
            errors.append(f"missing_display_name:{character_id or role_name or 'unknown'}")
        if not role_name:
            errors.append(f"missing_role:{character_id or display_name or 'unknown'}")
        if not voice_id:
            errors.append(f"missing_voice_id:{character_id or role_name or 'unknown'}")
        if not avatar_id:
            errors.append(f"missing_avatar_id:{character_id or role_name or 'unknown'}")
        elif avatar_id in avatar_ids:
            errors.append(f"duplicate_avatar_id:{avatar_id}")
        else:
            avatar_ids.add(avatar_id)
        for value in (character_id, display_name, role_name):
            if value:
                aliases.add(value)

    for required in required_characters or []:
        req = str(required or "").strip()
        if req and req not in aliases:
            errors.append(f"missing_required_character:{req}")
    return errors


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("character_id", "name", "display_name", "id", "speaker", "text", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    return str(value).strip()


def validate_timeline_references(
    timeline: Dict[str, Any],
    screenplay: Dict[str, Any],
    store: AssetStoreLike,
) -> List[str]:
    errors: List[str] = []
    scene_ids = {s.get("scene_id") for s in screenplay.get("scenes", []) if s.get("scene_id")}

    for scene in timeline.get("scenes", []):
        scene_id = scene.get("scene_id")
        if scene_id and scene_id not in scene_ids:
            errors.append(f"timeline_unknown_scene:{scene_id}")
        for layer in _timeline_entries(scene.get("layers", [])):
            asset_id = _timeline_asset_id(layer)
            if not _asset_exists(asset_id, store):
                errors.append(f"missing_asset:{asset_id}")
        for audio in _timeline_entries(scene.get("audio", [])):
            asset_id = _timeline_asset_id(audio)
            if not _asset_exists(asset_id, store):
                errors.append(f"missing_audio_asset:{asset_id}")
    return errors


def _timeline_entries(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _timeline_asset_id(entry: Any) -> Any:
    if isinstance(entry, dict):
        return entry.get("asset_id")
    if isinstance(entry, str):
        return entry
    return None


def _asset_exists(asset_id: Any, store: AssetStoreLike) -> bool:
    if not asset_id:
        return False
    if isinstance(asset_id, str) and os.path.exists(asset_id):
        return True
    try:
        store.get_path(str(asset_id))
        return True
    except Exception:
        return False
