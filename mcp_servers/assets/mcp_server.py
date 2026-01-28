"""MCP server entrypoint for asset service."""
import base64
import os
from typing import Any, Dict, Iterable, Optional

from mcp.server.fastmcp import FastMCP

from .server import AssetService


mcp = FastMCP("mcp-assets", json_response=True)
service = AssetService()


@mcp.tool()
def asset_put(
    data_b64: str,
    kind: str,
    tags: Optional[Iterable[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Store an asset. data_b64 is base64-encoded bytes."""
    data = base64.b64decode(data_b64)
    return service.asset_put(
        data=data,
        kind=kind,
        tags=tags,
        name=name,
        description=description,
        content_type=content_type,
    )


@mcp.tool()
def asset_get(asset_id: str) -> Dict[str, Any]:
    return service.asset_get(asset_id)


@mcp.tool()
def asset_search(
    query: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    return service.asset_search(query=query, tags=tags, kind=kind)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    mcp.run(transport=transport, host=host, port=port, path=path)
