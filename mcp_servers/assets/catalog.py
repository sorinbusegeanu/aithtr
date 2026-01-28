import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CATALOG_PATH = os.path.join(DEFAULT_DATA_DIR, "assets", "catalog.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: List[Dict[str, Any]]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


@dataclass
class CatalogEntry:
    asset_id: str
    name: str
    kind: str
    tags: List[str]
    description: str


class AssetCatalog:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.getenv("ASSET_CATALOG_PATH", DEFAULT_CATALOG_PATH)

    def list(self) -> List[Dict[str, Any]]:
        return _load_json(self.path)

    def add(self, entry: Dict[str, Any]) -> None:
        items = _load_json(self.path)
        items = [i for i in items if i.get("asset_id") != entry.get("asset_id")]
        items.append(entry)
        _save_json(self.path, items)

    def get(self, asset_id: str) -> Optional[Dict[str, Any]]:
        for item in _load_json(self.path):
            if item.get("asset_id") == asset_id:
                return item
        return None

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        tags = set(tags or [])
        results: List[Dict[str, Any]] = []
        for item in _load_json(self.path):
            if kind and item.get("kind") != kind:
                continue
            if tags:
                item_tags = set(item.get("tags", []))
                if not item_tags.intersection(tags):
                    continue
            if q:
                hay = " ".join(
                    [
                        str(item.get("name", "")),
                        str(item.get("description", "")),
                        " ".join(item.get("tags", [])),
                    ]
                ).lower()
                if q not in hay:
                    continue
            results.append(item)
        return results
