from __future__ import annotations

import os

from orchestrator import mcp_clients


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "asset_put":
            return {"asset_id": "dummy-id"}
        if name == "asset_get":
            return {"path": "/tmp/dummy"}
        return {}


def test_asset_client_uses_remote_client_for_sse(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setenv("MCP_ASSETS_URL", "http://mcp-assets:7101/mcp/sse")
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    monkeypatch.setattr(mcp_clients, "_make_client", lambda url, timeout_sec=None: dummy)

    client = mcp_clients.AssetClient()
    asset_id = client.put(b"hello", content_type="text/plain", tags=["test"])
    path = client.get_path("dummy-id")

    assert asset_id == "dummy-id"
    assert path == "/tmp/dummy"
    assert [name for name, _ in dummy.calls] == ["asset_put", "asset_get"]

