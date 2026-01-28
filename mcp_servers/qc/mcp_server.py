"""MCP server entrypoint for QC service."""
import os
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .server import QCService


mcp = FastMCP("mcp-qc", json_response=True)
service = QCService()


@mcp.tool()
def qc_audio(video_uri: str) -> Dict[str, Any]:
    return service.qc_audio(video_uri)


@mcp.tool()
def qc_video(video_uri: str) -> Dict[str, Any]:
    return service.qc_video(video_uri)


@mcp.tool()
def qc_timeline_validate(timeline_json: Dict[str, Any]) -> Dict[str, Any]:
    return service.qc_timeline_validate(timeline_json)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    mcp.run(transport=transport, host=host, port=port, path=path)
