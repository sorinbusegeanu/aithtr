"""Scene designer agent: produce scene_assets.json."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> scene assets."""
    llm = llm or LLMClient()
    scenes = []
    for scene in input_data.get("scenes", []):
        scenes.append(
            {
                "scene_id": scene.get("scene_id", "scene-1"),
                "background_asset_id": scene.get("background_asset_id", "bg-default"),
                "props": scene.get("props", []),
                "layout_hints": scene.get(
                    "layout_hints",
                    {"subtitle_safe_zone": {"x": 0.1, "y": 0.8, "width": 0.8, "height": 0.15}},
                ),
            }
        )
    return {"scenes": scenes}
