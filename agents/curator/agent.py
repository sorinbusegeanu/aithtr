"""Curator agent: produce memory updates."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> memory updates."""
    llm = llm or LLMClient()
    return {
        "updates": input_data.get("updates", []),
        "embeddings": input_data.get("embeddings", []),
    }
