"""Director agent: produce scene_plan.json."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Director. Produce scene_plan.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object: {{scenes:[...]}}
- Each scene: scene_id, stage[], entrances[], reactions[], subtitle_placement.
- stage items: {{character_id,x,y,scale}}.
- entrances: {{character_id,time_sec,action(enter|exit)}}.
- reactions: {{character_id,time_sec,duration_sec,reaction}}.
- subtitle_placement: {{x,y}} values 0-1.
- No extra keys.
""".strip()


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> scene plan."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    return llm.complete_json(prompt)
