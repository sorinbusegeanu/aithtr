"""Dramaturg agent: produce revision notes."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> revision notes."""
    llm = llm or LLMClient()
    # TODO: replace with real LLM prompt.
    return {
        "required_edits": [],
        "suggested_changes": [],
        "duration_targets": {},
    }
