"""Editor agent: produce timeline.json (EDL)."""
from typing import Any, Dict, List, Tuple

from agents.common import LLMClient


PROMPT = """
You are the Editor. Produce timeline.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object with keys: duration_sec, scenes.
- Each scene: scene_id, start_sec, end_sec, layers[], audio[].
- layers: {{type, asset_id, start_sec, end_sec, z, optional position{{x,y}}, optional scale}}.
- audio: {{type, asset_id, start_sec, end_sec, optional gain_db}}.
- Use performances if provided for ordering and layer assignment only.
- Do not invent timing from fixed scene duration or equal spacing.
- Prefer per-line references (line_id) for both audio and actor layers; timing is injected later by pipeline normalization.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> timeline."""
    del llm, critic_feedback
    sanitized = _sanitize_editor_input(input_data)
    return _build_timeline_deterministic(sanitized)


def _sanitize_editor_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(input_data, dict):
        return {}
    out: Dict[str, Any] = {}

    screenplay = input_data.get("screenplay", {})
    if isinstance(screenplay, dict):
        scenes_out = []
        for scene in screenplay.get("scenes", []) if isinstance(screenplay.get("scenes"), list) else []:
            if not isinstance(scene, dict):
                continue
            lines_out = []
            for line in scene.get("lines", []) if isinstance(scene.get("lines"), list) else []:
                if not isinstance(line, dict):
                    continue
                lines_out.append(
                    {
                        "line_id": line.get("line_id"),
                        "speaker": line.get("speaker"),
                        "emotion": line.get("emotion"),
                        "pause_ms_after": line.get("pause_ms_after"),
                    }
                )
            scenes_out.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "characters": scene.get("characters", []),
                    "lines": lines_out,
                }
            )
        out["screenplay"] = {"scenes": scenes_out}

    performances = input_data.get("performances", {})
    if isinstance(performances, dict):
        perf_scenes_out = []
        for scene in performances.get("scenes", []) if isinstance(performances.get("scenes"), list) else []:
            if not isinstance(scene, dict):
                continue
            chars_out = []
            for row in scene.get("characters", []) if isinstance(scene.get("characters"), list) else []:
                if not isinstance(row, dict):
                    continue
                segs_out = []
                for seg in row.get("segments", []) if isinstance(row.get("segments"), list) else []:
                    if not isinstance(seg, dict):
                        continue
                    segs_out.append(
                        {
                            "line_id": seg.get("line_id"),
                            "duration_sec": seg.get("duration_sec"),
                        }
                    )
                chars_out.append(
                    {
                        "character_id": row.get("character_id"),
                        "status": row.get("status"),
                        "error_code": row.get("error_code"),
                        "wav_artifact_id": row.get("wav_artifact_id"),
                        "video_artifact_id": row.get("video_artifact_id"),
                        "line_audio_artifacts": row.get("line_audio_artifacts", []),
                        "line_video_artifacts": row.get("line_video_artifacts", []),
                        "segments": segs_out,
                    }
                )
            perf_scenes_out.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "unresolved_lines": scene.get("unresolved_lines", []),
                    "characters": chars_out,
                }
            )
        out["performances"] = {"scenes": perf_scenes_out}

    scene_assets = input_data.get("scene_assets", {})
    if isinstance(scene_assets, dict):
        assets_scenes_out = []
        for scene in scene_assets.get("scenes", []) if isinstance(scene_assets.get("scenes"), list) else []:
            if not isinstance(scene, dict):
                continue
            assets_scenes_out.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "background_asset_id": scene.get("background_asset_id") or scene.get("background_artifact_id"),
                    "background_path": scene.get("background_path"),
                    "layout_hints": scene.get("layout_hints", {}),
                }
            )
        out["scene_assets"] = {"scenes": assets_scenes_out}

    scene_plan = input_data.get("scene_plan", {})
    if isinstance(scene_plan, dict):
        plan_scenes_out = []
        for scene in scene_plan.get("scenes", []) if isinstance(scene_plan.get("scenes"), list) else []:
            if not isinstance(scene, dict):
                continue
            plan_scenes_out.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "stage": scene.get("stage", []),
                    "entrances": scene.get("entrances", []),
                    "reactions": scene.get("reactions", []),
                    "subtitle_placement": scene.get("subtitle_placement", {}),
                }
            )
        out["scene_plan"] = {"scenes": plan_scenes_out}

    return out


def _build_timeline_deterministic(input_data: Dict[str, Any]) -> Dict[str, Any]:
    screenplay_scenes = (
        input_data.get("screenplay", {}).get("scenes", [])
        if isinstance(input_data.get("screenplay"), dict)
        else []
    )
    perf_scenes = (
        input_data.get("performances", {}).get("scenes", [])
        if isinstance(input_data.get("performances"), dict)
        else []
    )
    assets_scenes = (
        input_data.get("scene_assets", {}).get("scenes", [])
        if isinstance(input_data.get("scene_assets"), dict)
        else []
    )
    plan_scenes = (
        input_data.get("scene_plan", {}).get("scenes", [])
        if isinstance(input_data.get("scene_plan"), dict)
        else []
    )

    perf_by_scene = {
        str(scene.get("scene_id") or ""): scene
        for scene in perf_scenes
        if isinstance(scene, dict) and str(scene.get("scene_id") or "")
    }
    assets_by_scene = {
        str(scene.get("scene_id") or ""): scene
        for scene in assets_scenes
        if isinstance(scene, dict) and str(scene.get("scene_id") or "")
    }
    plan_by_scene = {
        str(scene.get("scene_id") or ""): scene
        for scene in plan_scenes
        if isinstance(scene, dict) and str(scene.get("scene_id") or "")
    }

    ordered_scene_ids: List[str] = []
    for scene in screenplay_scenes:
        if not isinstance(scene, dict):
            continue
        sid = str(scene.get("scene_id") or "")
        if sid and sid not in ordered_scene_ids:
            ordered_scene_ids.append(sid)
    if not ordered_scene_ids:
        for scene in perf_scenes:
            if not isinstance(scene, dict):
                continue
            sid = str(scene.get("scene_id") or "")
            if sid and sid not in ordered_scene_ids:
                ordered_scene_ids.append(sid)

    timeline_scenes: List[Dict[str, Any]] = []
    cursor = 0.0
    for sid in ordered_scene_ids:
        perf = perf_by_scene.get(sid, {})
        assets = assets_by_scene.get(sid, {})
        plan = plan_by_scene.get(sid, {})
        stage_positions = _stage_positions(plan.get("stage", []))

        rows = perf.get("characters", []) if isinstance(perf, dict) else []
        if not isinstance(rows, list):
            rows = []
        rows = [r for r in rows if isinstance(r, dict) and str(r.get("status") or "") == "ok"]

        scene_duration = _scene_duration_from_rows(rows)
        start = cursor
        end = start + scene_duration
        cursor = end

        layers: List[Dict[str, Any]] = []
        bg_id = ""
        if isinstance(assets, dict):
            bg_id = str(assets.get("background_asset_id") or assets.get("background_artifact_id") or "")
        if bg_id:
            layers.append(
                {
                    "type": "background",
                    "asset_id": bg_id,
                    "start_sec": start,
                    "end_sec": end,
                    "z": 0,
                }
            )

        audio: List[Dict[str, Any]] = []
        z = 10
        for row in rows:
            character_id = str(row.get("character_id") or "")
            wav_id = str(row.get("wav_artifact_id") or "")
            video_id = str(row.get("video_artifact_id") or "")
            segs = row.get("segments", [])
            if not isinstance(segs, list):
                segs = []

            line_audio_artifacts = row.get("line_audio_artifacts", [])
            if isinstance(line_audio_artifacts, list):
                for item in line_audio_artifacts:
                    if not isinstance(item, dict):
                        continue
                    wav_line_id = str(item.get("wav_artifact_id") or "").strip()
                    if not wav_line_id:
                        continue
                    audio.append(
                        {
                            "type": "dialogue",
                            "asset_id": wav_line_id,
                            "line_id": item.get("line_id"),
                            "character_id": character_id,
                            "start_sec": start,
                            "end_sec": start,
                        }
                    )
            elif wav_id:
                audio.append(
                    {
                        "type": "dialogue",
                        "asset_id": wav_id,
                        "start_sec": start,
                        "end_sec": start,
                    }
                )

            line_video_artifacts = row.get("line_video_artifacts", [])
            if isinstance(line_video_artifacts, list) and line_video_artifacts:
                for item in line_video_artifacts:
                    if not isinstance(item, dict):
                        continue
                    line_video_id = str(item.get("video_artifact_id") or "").strip()
                    if not line_video_id:
                        continue
                    layer = {
                        "type": "actor",
                        "asset_id": line_video_id,
                        "line_id": item.get("line_id"),
                        "character_id": character_id,
                        "start_sec": start,
                        "end_sec": start,
                        "z": z,
                    }
                    pos = stage_positions.get(character_id)
                    if pos:
                        layer["position"] = {"x": pos[0], "y": pos[1]}
                        layer["scale"] = pos[2]
                    layers.append(layer)
            elif video_id:
                layer = {
                    "type": "actor",
                    "asset_id": video_id,
                    "start_sec": start,
                    "end_sec": start,
                    "z": z,
                    "character_id": character_id,
                }
                pos = stage_positions.get(character_id)
                if pos:
                    layer["position"] = {"x": pos[0], "y": pos[1]}
                    layer["scale"] = pos[2]
                layers.append(layer)
            z += 1

        timeline_scenes.append(
            {
                "scene_id": sid,
                "start_sec": start,
                "end_sec": end,
                "layers": layers,
                "audio": audio,
            }
        )

    return {
        "duration_sec": cursor,
        "scenes": timeline_scenes,
    }


def _scene_duration_from_rows(rows: List[Dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        line_audio_artifacts = row.get("line_audio_artifacts", [])
        if isinstance(line_audio_artifacts, list) and line_audio_artifacts:
            for item in line_audio_artifacts:
                if not isinstance(item, dict):
                    continue
                total += max(float(item.get("duration_sec", 0.0) or 0.0), 0.0)
            continue
        segs = row.get("segments", [])
        if isinstance(segs, list):
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                total += max(float(seg.get("duration_sec", 0.0) or 0.0), 0.0)
    return max(total, 1.0)


def _stage_positions(stage_items: Any) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    if not isinstance(stage_items, list):
        return out
    for entry in stage_items:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("character_id") or "")
        if not cid:
            continue
        x = float(entry.get("x", 0.35))
        y = float(entry.get("y", 0.25))
        scale = float(entry.get("scale", 0.5))
        out[cid] = (x, y, scale)
    return out
