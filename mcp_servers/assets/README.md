# assets

Filesystem-backed, content-addressed artifact store.

## Files
- `artifact_store.py` Content-addressed artifact storage with metadata sidecars.

## MCP methods (planned)
- `asset.put(bytes, kind, tags)`
- `asset.get(asset_id)`
- `asset.search(tags|text)`
