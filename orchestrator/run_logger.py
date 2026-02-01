"""Per-run logging and artifact capture for observability."""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RunLogger:
    def __init__(self, run_dir: str, transcript_path: Optional[str] = None) -> None:
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "run.log")
        self.manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        self.transcript_path = transcript_path
        if self.transcript_path:
            transcript_dir = os.path.dirname(self.transcript_path)
            if transcript_dir:
                os.makedirs(transcript_dir, exist_ok=True)
        self.manifest: Dict[str, Any] = {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {}
        if not self.manifest:
            self.manifest = {
                "run_id": os.path.basename(run_dir),
                "started_at": _now(),
                "steps": {},
            }
        else:
            self.manifest.setdefault("run_id", os.path.basename(run_dir))
            self.manifest.setdefault("steps", {})

    def step_dir(self, step: str) -> str:
        path = os.path.join(self.run_dir, step)
        os.makedirs(path, exist_ok=True)
        return path

    def log(self, message: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] {message}\n")

    def save_step(self, step: str, payload: Dict[str, Any]) -> None:
        self.manifest["steps"].setdefault(step, {}).update(payload)
        self._flush()

    def log_chat(self, agent: str, messages: Optional[list[dict[str, str]]], raw: Optional[str]) -> None:
        if not self.transcript_path:
            return
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            if messages:
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    f.write(f"[{_now()}] agent:{agent} role:{role}\n{content}\n\n")
            assistant = _extract_assistant_content(raw)
            if assistant:
                f.write(f"[{_now()}] agent:{agent} role:assistant\n{assistant}\n\n")

    def write_json(self, path: str, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    def _flush(self) -> None:
        self.manifest["updated_at"] = _now()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=True, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_assistant_content(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        choices = parsed.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content
    except Exception:
        return raw
    return raw
