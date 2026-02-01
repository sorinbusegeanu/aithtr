# memory

SQLite-backed memory store + retrieval index (FTS5), plus character bible versioning.
Used by the orchestrator for summaries, continuity, asset/voice reuse, and critic learning.

## Files
- `schema.sql` Database schema and FTS triggers.
- `db.py` Storage API for memory operations.
- `server.py` Minimal service wrapper (ready to wire into MCP SDK).
- `mcp_server.py` MCP entrypoint for memory tools.

## MCP methods
- `memory.store(type, text, tags, episode_id)`
- `memory.retrieve(query, k, filters)`
- `memory.get_bible(name)`
- `memory.put_bible(name, json)`

## Orchestrator usage
The orchestrator uses a local memory client (SQLite directly) to:
- Store episode summaries and continuity notes after each run.
- Retrieve past summaries/continuity to seed showrunner/writer/director/editor prompts.
- Persist character bible updates from casting for consistency.
- Store and retrieve voice tuning history for casting.
- Store recent scene assets for reuse hints.
- Store critic reviews and inject step-specific lessons.

Common memory `type` values:
- `episode_summary`
- `continuity`
- `voice_tuning`
- `scene_assets`
- `critic_review`

Suggested tags:
- `episode`, `summary`, `continuity`
- `voice`, `voice_tuning`, `character:<id>`, `voice:<id>`
- `asset`, `scene_assets`
- `critic`, `step:<step_name>`, `passed` / `failed`

Filters are applied via `filters` in `memory.retrieve`, e.g.:
- `{ "type": "episode_summary" }`
- `{ "tags": ["step:writer"] }`
