"""
Minimal, dependency-free asset service wrapper.
Wire into MCP Python SDK in a later phase.
"""
from typing import Any, Dict, Iterable, Optional

from .artifact_store import ArtifactStore
from .catalog import AssetCatalog


class AssetService:
    def __init__(self, catalog_path: Optional[str] = None, root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=root)
        self.catalog = AssetCatalog(path=catalog_path)

    def asset_put(
        self,
        data: bytes,
        kind: str,
        tags: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        asset_id = self.store.put(data=data, content_type=content_type or "application/octet-stream", tags=tags)
        entry = {
            "asset_id": asset_id,
            "name": name or asset_id,
            "kind": kind,
            "tags": list(tags or []),
            "description": description or "",
        }
        self.catalog.add(entry)
        return entry

    def asset_get(self, asset_id: str) -> Dict[str, Any]:
        path = self.store.get_path(asset_id)
        meta = self.store.get_metadata(asset_id)
        entry = self.catalog.get(asset_id) or {"asset_id": asset_id}
        entry.update({"path": path, "content_type": meta.content_type, "size_bytes": meta.size_bytes})
        return entry

    def asset_search(
        self,
        query: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"results": self.catalog.search(query=query, tags=tags, kind=kind)}
