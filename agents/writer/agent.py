"""Writer agent: produce screenplay_draft.json."""
from typing import Any, Dict, List

from agents.common import LLMClient


PROMPT_PLAN = """
You are the Writer Planner. Produce plan.json strictly as JSON.

Input:
{input_json}

Hard constraints:
- Allowed speakers only: {allowed_speakers}
- scenes count must be between 1 and {max_scenes}
- scene characters must be a subset of allowed_speakers
- No extra keys.

Requirements:
- Return JSON object with keys: scenes, target_total_lines, target_total_words.
- scenes: array with scene entries: {{scene_id, setting_prompt, characters, line_budget}}.
- scene_id format must be "scene-<number>".
- line_budget must be integer > 0.
- target_total_lines must equal {target_total_lines}.
- target_total_words must equal {target_total_words}.
""".strip()


PROMPT_SCENE = """
You are the Writer. Produce ONE scene JSON strictly as JSON.

Input context:
{input_json}

Scene plan:
{scene_plan}

Style guard (must obey):
- max_line_length_chars: {max_line_length}
- forbidden_content: {forbidden_content}

Closed-world cast contract:
- allowed_speakers: {allowed_speakers}
- scene.characters[] entries must be from allowed_speakers only.
- lines[].speaker must be present in scene.characters only.
- Do not add new speakers.

Hard output requirements:
- Return JSON object with exactly keys: scene_id, setting_prompt, characters, lines.
- scene_id must equal "{scene_id}".
- characters must equal the planned characters list for this scene (same members).
- lines must contain EXACTLY {line_budget} entries. Not fewer. Not more.
- line_id must start at line-{start_line_id} and end at line-{end_line_id}, strictly sequential.
- Keep line_id global; do not reset numbering.
- If space is tight, shorten line text but keep exact line count.
- No repeated line text.
- No extra keys and no alternate formats.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> screenplay draft."""
    llm = llm or LLMClient(agent_name="writer")
    style_guard = (input_data.get("series_bible") or {}).get("style_guard", {})
    allowed_speakers = _as_string_list(input_data.get("allowed_speakers", []))
    target_duration = float(input_data.get("target_duration_sec") or 0.0)
    writer_targets = input_data.get("writer_targets", {}) if isinstance(input_data, dict) else {}
    target_total_lines = int(
        writer_targets.get("min_total_lines")
        or max(40, int(round(max(target_duration, 1.0) * 0.45)))
    )
    target_total_words = int(
        writer_targets.get("min_total_words")
        or max(int(max(target_duration, 1.0) * 2.2), int(max(target_duration, 1.0) * 2.0))
    )
    max_scenes = int(style_guard.get("max_scenes", 5) or 5)
    if max_scenes < 1:
        max_scenes = 1

    plan_prompt = PROMPT_PLAN.format(
        input_json=input_data,
        max_scenes=max_scenes,
        allowed_speakers=allowed_speakers,
        target_total_lines=target_total_lines,
        target_total_words=target_total_words,
    )
    raw_plan_json = llm.complete_json(plan_prompt)
    plan_json = _coerce_plan(
        raw_plan_json,
        allowed_speakers=allowed_speakers,
        max_scenes=max_scenes,
        target_total_lines=target_total_lines,
        target_total_words=target_total_words,
    )
    scenes_plan = plan_json.get("scenes", []) if isinstance(plan_json.get("scenes"), list) else []
    scenes_out: List[Dict[str, Any]] = []
    next_line_id = 1
    for scene in scenes_plan:
        if not isinstance(scene, dict):
            continue
        try:
            line_budget = int(scene.get("line_budget") or 1)
        except Exception:
            line_budget = 1
        if line_budget < 1:
            line_budget = 1
        start_line_id = next_line_id
        end_line_id = start_line_id + line_budget - 1
        scene_prompt = PROMPT_SCENE.format(
            input_json=input_data,
            scene_plan=scene,
            max_line_length=style_guard.get("max_line_length_chars", ""),
            forbidden_content=style_guard.get("forbidden_content", []),
            allowed_speakers=allowed_speakers,
            scene_id=str(scene.get("scene_id") or ""),
            line_budget=line_budget,
            start_line_id=start_line_id,
            end_line_id=end_line_id,
        )
        if critic_feedback:
            scene_prompt = (
                scene_prompt
                + "\n\n# Critic Feedback\n"
                + critic_feedback
                + "\n\nRevise your output accordingly while keeping the required JSON shape."
            )
        scene_json = llm.complete_json(scene_prompt)
        scene_json = _normalize_scene_json(
            scene_json,
            scene_plan=scene,
            allowed_speakers=allowed_speakers,
            start_line_id=start_line_id,
            line_budget=line_budget,
        )
        scenes_out.append(scene_json)
        next_line_id = end_line_id + 1

    realized: Dict[str, Any] = {"scenes": scenes_out, "_writer_plan": _sanitize_plan(plan_json)}
    return realized


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            v = item.strip()
            if v:
                out.append(v)
    return out


def _sanitize_plan(plan_json: Any) -> Dict[str, Any]:
    if not isinstance(plan_json, dict):
        return {}
    scenes_in = plan_json.get("scenes", [])
    scenes_out: List[Dict[str, Any]] = []
    if isinstance(scenes_in, list):
        for scene in scenes_in:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip()
            line_budget = scene.get("line_budget")
            try:
                line_budget_i = int(line_budget)
            except Exception:
                line_budget_i = 0
            if scene_id and line_budget_i > 0:
                scenes_out.append({"scene_id": scene_id, "line_budget": line_budget_i})
    out: Dict[str, Any] = {"scenes": scenes_out}
    try:
        ttl = int(plan_json.get("target_total_lines") or 0)
    except Exception:
        ttl = 0
    try:
        ttw = int(plan_json.get("target_total_words") or 0)
    except Exception:
        ttw = 0
    if ttl > 0:
        out["target_total_lines"] = ttl
    if ttw > 0:
        out["target_total_words"] = ttw
    return out


def _coerce_plan(
    plan_json: Any,
    *,
    allowed_speakers: List[str],
    max_scenes: int,
    target_total_lines: int,
    target_total_words: int,
) -> Dict[str, Any]:
    raw = plan_json if isinstance(plan_json, dict) else {}
    scenes_in = raw.get("scenes", []) if isinstance(raw.get("scenes"), list) else []
    scenes_out: List[Dict[str, Any]] = []
    for idx, scene in enumerate(scenes_in, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or "").strip() or f"scene-{idx}"
        if scene_id.isdigit():
            scene_id = f"scene-{scene_id}"
        if not scene_id.startswith("scene-"):
            scene_id = f"scene-{idx}"
        chars = [c for c in _as_string_list(scene.get("characters", [])) if c in allowed_speakers]
        if not chars and allowed_speakers:
            chars = [allowed_speakers[min(idx - 1, len(allowed_speakers) - 1)]]
        setting_prompt = str(scene.get("setting_prompt") or "").strip() or f"Scene {idx} setting."
        line_budget = int(scene.get("line_budget") or 0) if isinstance(scene.get("line_budget"), (int, float, str)) else 0
        scenes_out.append(
            {
                "scene_id": scene_id,
                "setting_prompt": setting_prompt,
                "characters": chars,
                "line_budget": max(1, line_budget),
            }
        )
        if len(scenes_out) >= max_scenes:
            break
    if not scenes_out:
        count = min(max_scenes, 5)
        if count < 1:
            count = 1
        for i in range(1, count + 1):
            chars = [allowed_speakers[min(i - 1, len(allowed_speakers) - 1)]] if allowed_speakers else []
            scenes_out.append(
                {
                    "scene_id": f"scene-{i}",
                    "setting_prompt": f"Scene {i} setting.",
                    "characters": chars,
                    "line_budget": 1,
                }
            )

    # Distribute exact total line budget.
    n = len(scenes_out)
    base_budget = target_total_lines // n
    rem = target_total_lines % n
    for i, scene in enumerate(scenes_out):
        scene["line_budget"] = base_budget + (1 if i < rem else 0)

    return {
        "scenes": scenes_out,
        "target_total_lines": target_total_lines,
        "target_total_words": target_total_words,
    }


def _normalize_scene_json(
    scene_json: Any,
    *,
    scene_plan: Dict[str, Any],
    allowed_speakers: List[str],
    start_line_id: int,
    line_budget: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = scene_json if isinstance(scene_json, dict) else {}
    scene_id = str(scene_plan.get("scene_id") or "").strip()
    setting_prompt = str(scene_plan.get("setting_prompt") or "").strip()
    planned_chars = _as_string_list(scene_plan.get("characters", []))
    if planned_chars:
        chars = planned_chars
    else:
        chars = [c for c in _as_string_list(out.get("characters", [])) if c in allowed_speakers]
    if not chars and allowed_speakers:
        chars = [allowed_speakers[0]]

    lines_in = out.get("lines", [])
    if not isinstance(lines_in, list):
        lines_in = []
    lines_out: List[Dict[str, Any]] = []
    for idx in range(line_budget):
        src = lines_in[idx] if idx < len(lines_in) and isinstance(lines_in[idx], dict) else {}
        speaker = str(src.get("speaker") or "").strip()
        if speaker not in chars:
            speaker = chars[idx % len(chars)] if chars else ""
        text = str(src.get("text") or "").strip()
        if not text:
            text = f"{speaker} continues the mythic narrative."
        emotion = str(src.get("emotion") or "neutral").strip() or "neutral"
        pause_raw = src.get("pause_ms_after", 250)
        try:
            pause_ms = int(pause_raw)
        except Exception:
            pause_ms = 250
        line_obj: Dict[str, Any] = {
            "line_id": f"line-{start_line_id + idx}",
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
            "pause_ms_after": pause_ms,
        }
        sfx_tag = src.get("sfx_tag")
        if isinstance(sfx_tag, str) and sfx_tag.strip():
            line_obj["sfx_tag"] = sfx_tag.strip()
        lines_out.append(line_obj)

    return {
        "scene_id": scene_id,
        "setting_prompt": setting_prompt,
        "characters": chars,
        "lines": lines_out,
    }
