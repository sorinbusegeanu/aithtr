import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_ARTIFACT_ROOT = os.path.join(DEFAULT_DATA_DIR, "artifacts")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_path(root: str, digest: str) -> str:
    return os.path.join(root, digest[:2], digest[2:4], digest)


@dataclass
class ArtifactMetadata:
    artifact_id: str
    size_bytes: int
    content_type: str
    tags: List[str]
    created_at: str


class ArtifactStore:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.getenv("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)
        _ensure_dir(self.root)

    def put(self, data: bytes, content_type: str, tags: Optional[Iterable[str]] = None) -> str:
        digest = _hash_bytes(data)
        path = _artifact_path(self.root, digest)
        meta_path = path + ".json"
        _ensure_dir(os.path.dirname(path))

        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)

        if not os.path.exists(meta_path):
            metadata = {
                "artifact_id": digest,
                "size_bytes": len(data),
                "content_type": content_type,
                "tags": list(tags or []),
                "created_at": _now_iso(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=True)

        return digest

    def get_path(self, artifact_id: str) -> str:
        path = _artifact_path(self.root, artifact_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        return path

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata:
        meta_path = _artifact_path(self.root, artifact_id) + ".json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata not found: {artifact_id}")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return ArtifactMetadata(
            artifact_id=raw["artifact_id"],
            size_bytes=int(raw["size_bytes"]),
            content_type=raw.get("content_type", ""),
            tags=list(raw.get("tags", [])),
            created_at=raw.get("created_at", ""),
        )

    def list(self, tags: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        tags = list(tags or [])
        results: List[Dict[str, Any]] = []
        for root, _dirs, files in os.walk(self.root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                meta_path = os.path.join(root, name)
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if tags:
                    meta_tags = set(meta.get("tags", []))
                    if not meta_tags.intersection(tags):
                        continue
                results.append(meta)
        return results


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
