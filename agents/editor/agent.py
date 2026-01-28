"""Editor agent: produce timeline.json (EDL)."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> timeline."""
    llm = llm or LLMClient()
    return {
        "duration_sec": input_data.get("duration_sec", 1.0),
        "scenes": input_data.get("scenes", []),
    }
