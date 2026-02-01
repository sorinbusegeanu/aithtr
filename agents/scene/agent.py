"""Scene designer agent: produce scene_assets.json."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Scene Designer. Produce scene_assets.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object: {{scenes:[...]}}
- Each scene: scene_id, background_asset_id, props[], layout_hints.
- layout_hints: {{subtitle_safe_zone:{{x,y,width,height}}}} values 0-1.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> scene assets."""
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
