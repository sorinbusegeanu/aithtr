"""MCP server entrypoint for render service."""
import os
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .server import RenderService


host = os.getenv("MCP_HOST", "127.0.0.1")
port = int(os.getenv("MCP_PORT", "8000"))
path = os.getenv("MCP_PATH", "/mcp").rstrip("/")
sse_path = f"{path}/sse"
message_path = f"{path}/messages/"

mcp = FastMCP(
    "mcp-render",
    json_response=True,
    host=host,
    port=port,
    sse_path=sse_path,
    message_path=message_path,
)
service = RenderService()


@mcp.tool()
def render_preview(timeline_uri: str) -> Dict[str, Any]:
    return service.render_preview(timeline_uri)


@mcp.tool()
def render_final(timeline_uri: str) -> Dict[str, Any]:
    return service.render_final(timeline_uri)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
