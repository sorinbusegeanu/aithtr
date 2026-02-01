"""QC agent: produce qc_report.json."""
from typing import Any, Dict

from agents.common import LLMClient


PROMPT = """
You are the QC Agent. Produce qc_report.json strictly as JSON.

Input:
{input_json}

Requirements:
- Return JSON object with keys: duration_sec, audio, video, subtitles, errors.
- audio: {{clipping: bool, silence_gaps:[{{start_sec,end_sec}}]}}
- video: {{black_frames:[{{start_sec,end_sec}}], fps:number}}
- subtitles: {{missing: bool}}
- errors: string[]
- No extra keys.
""".strip()


def run(
    input_data: Dict[str, Any],
    llm: LLMClient | None = None,
    critic_feedback: str | None = None,
) -> Dict[str, Any]:
    """Pure function: input -> QC report."""
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
