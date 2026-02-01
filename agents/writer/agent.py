"""Writer agent: produce screenplay_draft.json."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Writer. Produce screenplay_draft.json strictly as JSON.

Input:
{input_json}

Style guard (must obey):
- max_line_length_chars: {max_line_length}
- max_scenes: {max_scenes}
- forbidden_content: {forbidden_content}
- minimum_total_words: {min_words}

Requirements:
- Return JSON object: {{scenes: [...]}}.
- Each scene has: scene_id, setting_prompt, characters[], lines[].
- Each line: line_id, speaker, text, emotion, pause_ms_after, optional sfx_tag.
- Keep dialogue turn-based and concise.
- Target duration (seconds): {target_duration_sec}. Ensure the total dialogue roughly matches this duration.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> screenplay draft."""
    llm = llm or LLMClient()
    style_guard = (input_data.get("series_bible") or {}).get("style_guard", {})
    prompt = PROMPT.format(
        input_json=input_data,
        max_line_length=style_guard.get("max_line_length_chars", ""),
        max_scenes=style_guard.get("max_scenes", ""),
        forbidden_content=style_guard.get("forbidden_content", []),
        target_duration_sec=input_data.get("target_duration_sec", ""),
        min_words=int(max((input_data.get("target_duration_sec") or 0) * 2.0, 0)),
    )
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
