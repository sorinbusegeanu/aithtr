"""Simple file-backed cache for step outputs."""
import hashlib
import json
import os
from typing import Any, Dict, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, "data", "artifacts", "cache_index.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StepCache:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.getenv("ORCH_CACHE_PATH", DEFAULT_CACHE_PATH)
        _ensure_dir(self.path)
        self._data: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._data = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def _save(self) -> None:
        _ensure_dir(self.path)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=True, indent=2)

    def make_key(self, step_name: str, payload: Any) -> str:
        return f"{step_name}:{_hash_payload(payload)}"

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, artifact_id: str) -> None:
        self._data[key] = artifact_id
        self._save()
