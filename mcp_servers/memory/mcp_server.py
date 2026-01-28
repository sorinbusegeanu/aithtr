"""MCP server entrypoint for memory service."""
import os
from typing import Any, Dict, Iterable, Optional

from mcp.server.fastmcp import FastMCP

from .server import MemoryService


mcp = FastMCP("mcp-memory", json_response=True)
service = MemoryService()


@mcp.tool()
def memory_store(
    type: str,
    text: str,
    tags: Optional[Iterable[str]] = None,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    return service.memory_store(type=type, text=text, tags=tags, episode_id=episode_id)


@mcp.tool()
def memory_retrieve(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return service.memory_retrieve(query=query, k=k, filters=filters)


@mcp.tool()
def memory_get_bible(name: str) -> Dict[str, Any]:
    return service.memory_get_bible(name)


@mcp.tool()
def memory_put_bible(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return service.memory_put_bible(name, data)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    mcp.run(transport=transport, host=host, port=port, path=path)
