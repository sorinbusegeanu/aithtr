from __future__ import annotations

import os

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.assets.catalog import AssetCatalog
from mcp_servers.assets.server import AssetService


def test_artifact_store_put_get(tmp_path):
    store = ArtifactStore(root=str(tmp_path / "artifacts"))
    artifact_id = store.put(b"hello", content_type="text/plain", tags=["test"])
    path = store.get_path(artifact_id)
    meta = store.get_metadata(artifact_id)
    assert os.path.exists(path)
    assert meta.content_type == "text/plain"
    assert meta.size_bytes == 5


def test_asset_catalog_add_search(tmp_path):
    catalog = AssetCatalog(path=str(tmp_path / "catalog.json"))
    entry = {
        "asset_id": "a1",
        "name": "Forest BG",
        "kind": "background",
        "tags": ["forest", "day"],
        "description": "Sunlit forest.",
    }
    catalog.add(entry)
    assert catalog.get("a1")["name"] == "Forest BG"
    results = catalog.search(query="forest")
    assert len(results) == 1
    results = catalog.search(tags=["day"])
    assert len(results) == 1
    results = catalog.search(kind="props")
    assert len(results) == 0


def test_asset_service_put_get_search(tmp_path):
    service = AssetService(
        catalog_path=str(tmp_path / "catalog.json"),
        root=str(tmp_path / "artifacts"),
    )
    entry = service.asset_put(
        data=b"abc",
        kind="props",
        tags=["prop", "small"],
        name="Apple",
        description="Red apple",
        content_type="text/plain",
    )
    fetched = service.asset_get(entry["asset_id"])
    assert fetched["asset_id"] == entry["asset_id"]
    assert fetched["content_type"] == "text/plain"
    results = service.asset_search(query="apple")
    assert len(results["results"]) == 1
