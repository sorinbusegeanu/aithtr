"""Showrunner agent: produce episode_brief.json."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Showrunner. Produce episode_brief.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object with keys: premise (string), beats (array), tone (string), cast_constraints (object).
- beats: array of {{label: string, duration_sec: number}}.
- cast_constraints: {{required_characters: string[], max_cast_per_scene: int, max_scenes: int}}.
- Keep total duration near input duration_sec if provided.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> episode brief."""
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
