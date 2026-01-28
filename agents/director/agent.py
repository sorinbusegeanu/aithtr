"""Director agent: produce scene_plan.json."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> scene plan."""
    llm = llm or LLMClient()
    scenes = []
    for scene in input_data.get("scenes", []):
        scenes.append(
            {
                "scene_id": scene.get("scene_id", "scene-1"),
                "stage": scene.get("stage", []),
                "entrances": scene.get("entrances", []),
                "reactions": scene.get("reactions", []),
                "subtitle_placement": scene.get("subtitle_placement", {"x": 0.1, "y": 0.85}),
            }
        )
    return {"scenes": scenes}
