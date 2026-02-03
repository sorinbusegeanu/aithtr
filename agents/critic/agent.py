"""Critic agent: evaluate agent outputs."""
from typing import Any, Dict, List, Optional

from agents.common import LLMClient


PROMPT = """
You are the Critic. Evaluate the agent output for quality and correctness.
Return JSON only with keys: evaluation (string), quality_score (integer 0-100), passed (boolean).

Step name:
{step_name}

Schema hint (if any):
{schema_hint}

Validation errors (if any):
{validation_errors}

Input:
{input_json}

Output:
{output_json}

Guidance:
- Be descriptive and specific in evaluation.
- Passed should be true only if output is solid and requires no changes.
""".strip()


def run(
    step_name: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    schema_hint: str = "",
    validation_errors: Optional[List[str]] = None,
    llm: LLMClient | None = None,
) -> Dict[str, Any]:
    """Critique a step output."""
    llm = llm or LLMClient(agent_name="critic")
    prompt = PROMPT.format(
        step_name=step_name,
        schema_hint=schema_hint,
        validation_errors=validation_errors or [],
        input_json=input_data,
        output_json=output_data,
    )
    return llm.complete_json(prompt)
