"""Human gate helpers."""
from typing import Any, Dict, List


def render_screenplay_markdown(screenplay: Dict[str, Any]) -> str:
    lines: List[str] = ["# Screenplay\n"]
    for scene in screenplay.get("scenes", []):
        lines.append(f"## Scene {scene.get('scene_id', '')}")
        lines.append(f"**Setting:** {scene.get('setting_prompt', '')}")
        lines.append("**Characters:** " + ", ".join(scene.get("characters", [])))
        lines.append("")
        for line in scene.get("lines", []):
            line_id = line.get("line_id", "")
            speaker = line.get("speaker", "")
            text = line.get("text", "")
            emotion = line.get("emotion", "")
            lines.append(f"- `{line_id}` **{speaker}** ({emotion}): {text}")
        lines.append("")
    return "\n".join(lines)


def apply_line_edits(screenplay: Dict[str, Any], edits: Dict[str, str]) -> Dict[str, Any]:
    for scene in screenplay.get("scenes", []):
        for line in scene.get("lines", []):
            line_id = line.get("line_id")
            if line_id in edits:
                line["text"] = edits[line_id]
    return screenplay
