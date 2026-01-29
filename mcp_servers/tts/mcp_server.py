"""MCP server entrypoint for TTS service."""
import os
from typing import Any, Dict, Iterable, Optional

from mcp.server.fastmcp import FastMCP

from .server import TTSService


mcp = FastMCP("mcp-tts", json_response=True)
service = TTSService()


@mcp.tool()
def tts_list_voices() -> Dict[str, Any]:
    return service.tts_list_voices()


@mcp.tool()
def tts_synthesize(
    text: str,
    voice_id: str,
    style: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    return service.tts_synthesize(text=text, voice_id=voice_id, style=style, tags=tags)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    mcp.run(transport=transport)
