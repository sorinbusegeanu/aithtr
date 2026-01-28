"""Shared helpers for agent functions."""
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    seed: Optional[int] = 42
    max_tokens: int = 2048
    json_only: bool = True


class LLMClient:
    """Placeholder LLM client. Replace with real provider integration."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        """Return JSON-only output. Replace with real call."""
        raise NotImplementedError("LLM integration not implemented")


def ensure_json_only(text: str) -> Dict[str, Any]:
    """Parse a JSON-only response string."""
    return json.loads(text)
