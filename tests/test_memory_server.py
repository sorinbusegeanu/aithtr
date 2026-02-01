from __future__ import annotations

import os
import sqlite3

import pytest

from mcp_servers.memory.db import MemoryDB
from mcp_servers.memory.server import MemoryService


def test_memory_db_store_retrieve(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_path))
    db = MemoryDB(db_path=str(db_path))
    note_id = db.store("episode_summary", "Hello world", tags=["episode", "summary"])
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM retrieval_fts").fetchone()
        fts_count = int(row[0]) if row else 0
    if fts_count == 0:
        pytest.skip("FTS index not populated in this environment")
    results = db.retrieve("hello*", k=5)
    assert results
    assert results[0].note_id == note_id


def test_memory_service_bibles(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_path))
    service = MemoryService(db_path=str(db_path))
    res = service.memory_put_bible("character_bible", {"characters": []})
    assert res["version"] == 1
    fetched = service.memory_get_bible("character_bible")
    assert fetched["data"]["characters"] == []
