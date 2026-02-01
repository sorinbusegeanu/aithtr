"""Memory client backed by MCP HTTP or local SQLite (FTS5)."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from mcp_servers.memory import MemoryDB

from .mcp_clients import _make_client, _transport


class MemoryClient:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.url = (os.getenv("MCP_MEMORY_URL") or "").strip()
        if self.url:
            self.client = _make_client(self.url, timeout_sec=os.getenv("MCP_MEMORY_TIMEOUT_SEC"))
            self.mode = _transport()
        else:
            self.db = MemoryDB(db_path=db_path)
            self.mode = "local"

    def store(
        self,
        type_: str,
        text: str,
        tags: Optional[Iterable[str]] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        if self.mode != "local":
            res = self.client.call(
                "memory_store",
                {"type": type_, "text": text, "tags": list(tags or []), "episode_id": episode_id},
            )
            return str(res.get("note_id"))
        return self.db.store(type_=type_, text=text, tags=tags, episode_id=episode_id)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.mode != "local":
            res = self.client.call("memory_retrieve", {"query": query, "k": k, "filters": filters})
            return res.get("results", [])
        results = self.db.retrieve(query=query, k=k, filters=filters)
        return [
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

    def get_bible(self, name: str) -> Optional[Dict[str, Any]]:
        if self.mode != "local":
            res = self.client.call("memory_get_bible", {"name": name})
            return res.get("data")
        return self.db.get_bible(name)

    def put_bible(self, name: str, data: Dict[str, Any]) -> int:
        if self.mode != "local":
            res = self.client.call("memory_put_bible", {"name": name, "data": data})
            return int(res.get("version", 0))
        return self.db.put_bible(name, data)
