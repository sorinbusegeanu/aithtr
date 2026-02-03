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


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> scene plan."""
    llm = llm or LLMClient(agent_name="director")
    prompt = PROMPT.format(input_json=input_data)
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
