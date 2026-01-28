import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DATA_DIR, "sqlite", "memory.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


@dataclass
class RetrievalResult:
    note_id: str
    type: str
    text: str
    tags: List[str]
    episode_id: Optional[str]
    score: float


class MemoryDB:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH", DEFAULT_DB_PATH)
        _ensure_dir(self.db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

    def store(self, type_: str, text: str, tags: Optional[Iterable[str]] = None, episode_id: Optional[str] = None) -> str:
        note_id = str(uuid.uuid4())
        tags_json = json.dumps(list(tags or []))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_notes(note_id, type, text, tags_json, episode_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (note_id, type_, text, tags_json, episode_id),
            )
        return note_id

    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        filters = filters or {}
        where = []
        params: List[Any] = []

        if "type" in filters:
            where.append("retrieval_notes.type = ?")
            params.append(filters["type"])
        if "episode_id" in filters:
            where.append("retrieval_notes.episode_id = ?")
            params.append(filters["episode_id"])
        if "tags" in filters:
            tags = list(filters["tags"]) if filters["tags"] else []
            if tags:
                where.append(
                    "EXISTS (SELECT 1 FROM json_each(retrieval_notes.tags_json) WHERE value IN (%s))"
                    % ",".join(["?"] * len(tags))
                )
                params.extend(tags)

        where_sql = " AND ".join(where)
        if where_sql:
            where_sql = "AND " + where_sql

        sql = (
            """
            SELECT retrieval_notes.note_id, retrieval_notes.type, retrieval_notes.text,
                   retrieval_notes.tags_json, retrieval_notes.episode_id,
                   bm25(retrieval_fts) AS score
            FROM retrieval_fts
            JOIN retrieval_notes ON retrieval_notes.note_id = retrieval_fts.note_id
            WHERE retrieval_fts MATCH ?
            """
            + where_sql
            + " ORDER BY score LIMIT ?"
        )
        params = [query] + params + [k]

        results: List[RetrievalResult] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                results.append(
                    RetrievalResult(
                        note_id=row["note_id"],
                        type=row["type"],
                        text=row["text"],
                        tags=json.loads(row["tags_json"] or "[]"),
                        episode_id=row["episode_id"],
                        score=float(row["score"]),
                    )
                )
        return results

    def get_bible(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT json FROM character_bible_versions
                WHERE name = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["json"])

    def put_bible(self, name: str, data: Dict[str, Any]) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM character_bible_versions WHERE name = ?",
                (name,),
            ).fetchone()
            next_version = int(row["v"]) + 1
            conn.execute(
                "INSERT INTO character_bible_versions(name, version, json) VALUES (?, ?, ?)",
                (name, next_version, json.dumps(data)),
            )
        return next_version
