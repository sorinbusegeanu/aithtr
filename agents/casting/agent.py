"""Casting agent: produce cast_plan.json and update cast bible."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Casting Director. Produce JSON only.

Input:
{input_json}

Requirements:
- Return JSON object with keys: cast_plan, cast_bible_update.
- cast_plan: {{roles:[{{role, character_id, voice_id, avatar_id, emotion_map}}]}}
- emotion_map maps emotion -> {{style, optional rate, pitch, energy}}.
- cast_bible_update: object (can be empty) with any new/updated character details.
- No extra keys.
""".strip()


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> cast plan + bible update."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    return llm.complete_json(prompt)
