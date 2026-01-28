"""
Minimal, dependency-free memory service implementation.
Wire this into the MCP Python SDK in a later phase.
"""
import json
from typing import Any, Dict, Iterable, List, Optional

from .db import MemoryDB


class MemoryService:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db = MemoryDB(db_path=db_path)

    def memory_store(
        self,
        type: str,
        text: str,
        tags: Optional[Iterable[str]] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        note_id = self.db.store(type_=type, text=text, tags=tags, episode_id=episode_id)
        return {"note_id": note_id}

    def memory_retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = self.db.retrieve(query=query, k=k, filters=filters)
        return {
            "results": [
                {
                    "note_id": r.note_id,
                    "type": r.type,
                    "text": r.text,
                    "tags": r.tags,
                    "episode_id": r.episode_id,
                    "score": r.score,
                }
                for r in results
            ]
        }

    def memory_get_bible(self, name: str) -> Dict[str, Any]:
        data = self.db.get_bible(name)
        return {"name": name, "data": data}

    def memory_put_bible(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        version = self.db.put_bible(name, data)
        return {"name": name, "version": version}


def _example() -> None:
    service = MemoryService()
    res = service.memory_store("screenplay", "hello world", tags=["test"], episode_id="ep1")
    print(json.dumps(res, indent=2))
    res = service.memory_retrieve("hello", k=3)
    print(json.dumps(res, indent=2))
    res = service.memory_put_bible("main", {"characters": []})
    print(json.dumps(res, indent=2))
    res = service.memory_get_bible("main")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _example()
