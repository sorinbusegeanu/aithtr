"""Curator agent: produce memory updates."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Memory Curator. Produce JSON only.

Input:
{input_json}

Requirements:
- Return JSON object with keys: updates, embeddings.
- updates: array of objects describing memory/store operations.
- embeddings: array of objects (can be empty).
- No extra keys.
""".strip()


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> memory updates."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    return llm.complete_json(prompt)
