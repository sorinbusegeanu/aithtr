"""Shared helpers for agent functions."""
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


_DEFAULT_TEMPERATURES: Dict[str, float] = {
    "showrunner": 0.25,
    "writer": 0.8,
    "dramaturg": 0.5,
    "casting": 0.4,
    "scene": 0.6,
    "director": 0.5,
    "editor": 0.15,
    "qc": 0.05,
    "curator": 0.25,
    "critic": 0.05,
}


def _agent_temp_env_key(agent_name: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in (agent_name or "default"))
    return f"LLM_TEMPERATURE_{normalized.upper()}"


def _resolve_temperature(agent_name: str) -> float:
    # Per-agent override wins (e.g., LLM_TEMPERATURE_WRITER).
    value = os.getenv(_agent_temp_env_key(agent_name))
    if value is not None:
        return float(value)
    return _DEFAULT_TEMPERATURES.get(agent_name, 0.1)


@dataclass
class LLMConfig:
    agent_name: str = "default"
    model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    temperature: Optional[float] = None
    seed: Optional[int] = int(os.getenv("LLM_SEED", "42"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    json_only: bool = True
    timeout_sec: int = int(os.getenv("LLM_TIMEOUT_SEC", "60"))

    def __post_init__(self) -> None:
        if self.temperature is None:
            self.temperature = _resolve_temperature(self.agent_name)


class LLMClient:
    """vLLM OpenAI-compatible client."""

    def __init__(self, config: Optional[LLMConfig] = None, agent_name: str = "default") -> None:
        self.config = config or LLMConfig(agent_name=agent_name)
        self.last_raw: Optional[str] = None
        self.last_prompt: Optional[str] = None
        self.last_messages: Optional[list[dict[str, str]]] = None

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        """Return JSON-only output from vLLM."""
        self.last_prompt = prompt
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        system_msg = "You must respond with JSON only. No prose."
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        self.last_messages = list(payload["messages"])
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if self.config.json_only:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        last_err: Optional[Exception] = None
        for attempt in range(3):
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            self.last_raw = raw

            parsed = json.loads(raw)
            choices = parsed.get("choices", [])
            if not choices:
                raise RuntimeError("LLM returned no choices")
            content = choices[0].get("message", {}).get("content", "")

            try:
                return ensure_json_only(content)
            except json.JSONDecodeError as err:
                last_err = err
                # Try to salvage JSON substring
                salvage = _extract_json(content)
                if salvage is not None:
                    return salvage
                # Ask model to repair the invalid JSON
                repaired = self._repair_json(content)
                if repaired is not None:
                    return repaired
                # Retry with stronger instruction
                prompt = (
                    "Return valid JSON only. Do not include any other text.\n\n"
                    + self.last_prompt
                )
                payload["messages"] = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers)
        raise last_err or RuntimeError("Failed to parse JSON from LLM")

    def _repair_json(self, bad_json: str) -> Optional[Dict[str, Any]]:
        system_msg = "You fix invalid JSON. Return only valid JSON. No prose."
        prompt = (
            "Fix the JSON below. Return valid JSON only.\n\n"
            "<json>\n"
            + bad_json
            + "\n</json>"
        )
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        self.last_messages = list(payload["messages"])
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if self.config.json_only:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            choices = parsed.get("choices", [])
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            return ensure_json_only(content)
        except Exception:
            return None


def ensure_json_only(text: str) -> Dict[str, Any]:
    """Parse a JSON-only response string."""
    return json.loads(text)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None
