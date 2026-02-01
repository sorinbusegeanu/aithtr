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


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> memory updates."""
    llm = llm or LLMClient()
    prompt = PROMPT.format(input_json=input_data)
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
