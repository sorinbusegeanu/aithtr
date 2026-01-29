"""Dramaturg agent: produce revision notes."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Dramaturg. Produce screenplay_revision_notes.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object with keys: required_edits (array), suggested_changes (array), duration_targets (object).
- required_edits and suggested_changes are arrays of strings.
- duration_targets can include per-scene or total duration guidance.
- No extra keys.
""".strip()


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> revision notes."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    return llm.complete_json(prompt)
