"""MCP server entrypoint for lipsync service."""
import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .server import LipSyncService


mcp = FastMCP("mcp-lipsync", json_response=True)
service = LipSyncService()


@mcp.tool()
def lipsync_render_clip(
    avatar_id: str,
    wav_uri: str,
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return service.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_uri, style=style)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    mcp.run(transport=transport, host=host, port=port, path=path)
