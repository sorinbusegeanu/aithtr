"""Editor agent: produce timeline.json (EDL)."""
from typing import Any, Dict

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
- Use performances if provided: for each (scene_id, character_id), you may have one video_artifact_id and segments with line_id + start_sec/end_sec.
  In that case, create one layer per line segment that references the same video_artifact_id and uses the segment start/end times.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> timeline."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
