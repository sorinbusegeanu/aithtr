"""Casting agent: produce cast_plan.json and update cast bible."""
from typing import Any, Dict, List

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> cast plan + bible update."""
    llm = llm or LLMClient()
    roles = []
    for role in input_data.get("roles", []):
        roles.append(
            {
                "role": role,
                "character_id": role,
                "voice_id": "voice-default",
                "avatar_id": "avatar-default",
                "emotion_map": {"neutral": {"style": "neutral"}},
            }
        )
    return {
        "cast_plan": {"roles": roles},
        "cast_bible_update": input_data.get("cast_bible", {}),
    }
