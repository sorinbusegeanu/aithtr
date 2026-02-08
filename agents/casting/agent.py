"""Casting agent: produce cast_plan.json and update cast bible."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the Casting Director. Produce JSON only.

Input:
{input_json}

Requirements:
- Return JSON object with keys: cast_plan, cast_bible_update.
- cast_plan: {{roles:[{{role, character_id, display_name, voice_id, voice_seed_text, voice_seed_seconds_target, avatar_id, emotion_map}}]}}
- emotion_map maps emotion -> {{style, optional rate, pitch, energy}}.
- role/display_name should cover the required cast roster from input constraints.
- character_id should be a stable machine key; display_name should be human-readable.
- voice_id must be an explicit Piper-compatible voice id (e.g. en_US-lessac-medium), never "voice-default".
- voice_seed_text should be 60-120 words of clean spoken text (no SFX/stage directions/name prefixes).
- voice_seed_seconds_target default: 10.
- cast_bible_update: object (can be empty) with any new/updated character details.
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> cast plan + bible update."""
    llm = llm or LLMClient(agent_name="casting")
    prompt = PROMPT.format(input_json=input_data)
    if critic_feedback:
        prompt = (
            prompt
            + "\n\n# Critic Feedback\n"
            + critic_feedback
            + "\n\nRevise your output accordingly while keeping the required JSON shape."
        )
    return llm.complete_json(prompt)
