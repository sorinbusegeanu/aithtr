"""Writer agent: produce screenplay_draft.json."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> screenplay draft."""
    llm = llm or LLMClient()
    # TODO: replace with real LLM prompt. Deterministic config set in LLMClient.
    scene = {
        "scene_id": "scene-1",
        "setting_prompt": "A simple stage.",
        "characters": input_data.get("characters", ["A", "B"]),
        "lines": [
            {
                "line_id": "line-1",
                "speaker": "A",
                "text": "Hello.",
                "emotion": "neutral",
                "pause_ms_after": 200,
            }
        ],
    }
    return {"scenes": [scene]}
