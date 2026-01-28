"""Showrunner agent: produce episode_brief.json."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> episode brief."""
    llm = llm or LLMClient()
    # TODO: replace with real LLM prompt. Deterministic config set in LLMClient.
    return {
        "premise": input_data.get("premise", "A short character-driven scene."),
        "beats": input_data.get("beats", [{"label": "beat-1", "duration_sec": 60}]),
        "tone": input_data.get("tone", "light"),
        "cast_constraints": input_data.get(
            "cast_constraints",
            {"required_characters": [], "max_cast_per_scene": 2, "max_scenes": 3},
        ),
    }
