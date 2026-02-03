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


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> revision notes."""
    llm = llm or LLMClient(agent_name="dramaturg")
    prompt = PROMPT.format(input_json=input_data)
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
