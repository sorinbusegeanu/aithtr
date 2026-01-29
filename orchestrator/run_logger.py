"""Per-run logging and artifact capture for observability."""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RunLogger:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "run.log")
        self.manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        self.manifest: Dict[str, Any] = {
            "run_id": os.path.basename(run_dir),
            "started_at": _now(),
            "steps": {},
        }

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

    def write_json(self, path: str, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    def _flush(self) -> None:
        self.manifest["updated_at"] = _now()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=True, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
