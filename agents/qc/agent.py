"""QC agent: produce qc_report.json."""
from typing import Any, Dict

from agents.common import LLMClient


def run(input_data: Dict[str, Any], llm: LLMClient | None = None) -> Dict[str, Any]:
    """Pure function: input -> QC report."""
    llm = llm or LLMClient()
    return {
        "duration_sec": input_data.get("duration_sec", 0.0),
        "audio": input_data.get("audio", {"clipping": False, "silence_gaps": []}),
        "video": input_data.get("video", {"black_frames": [], "fps": 0}),
        "subtitles": input_data.get("subtitles", {"missing": False}),
        "errors": input_data.get("errors", []),
    }
