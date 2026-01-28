# memory

SQLite-backed memory store + retrieval index (FTS5), plus character bible versioning.

## Files
- `schema.sql` Database schema and FTS triggers.
- `db.py` Storage API for memory operations.
- `server.py` Minimal service wrapper (ready to wire into MCP SDK).

## MCP methods (planned)
- `memory.store(type, text, tags, episode_id)`
- `memory.retrieve(query, k, filters)`
- `memory.get_bible(name)`
- `memory.put_bible(name, json)`
