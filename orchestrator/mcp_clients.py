"""MCP clients with local fallbacks for core services."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional
from datetime import timedelta

from mcp_servers.assets.artifact_store import ArtifactStore
from mcp_servers.qc.server import QCService
from mcp_servers.render.server import RenderService
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
import anyio


def _env_url(name: str) -> Optional[str]:
    value = (os.getenv(name) or "").strip()
    return value or None


class MCPHttpClient:
    def __init__(self, url: str, timeout_sec: Optional[float] = None) -> None:
        self.url = url
        self.timeout_sec = float(timeout_sec or os.getenv("MCP_HTTP_TIMEOUT_SEC", "60"))

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"MCP HTTP {exc.code} {exc.reason}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MCP HTTP connection failed: {exc}") from exc
        parsed = json.loads(raw)
        if "error" in parsed:
            raise RuntimeError(parsed["error"])
        return parsed.get("result", {})


class MCPSSEClient:
    def __init__(self, url: str, timeout_sec: Optional[float] = None) -> None:
        self.url = url
        self.timeout_sec = float(timeout_sec or os.getenv("MCP_HTTP_TIMEOUT_SEC", "60"))
        self.read_timeout_sec = float(os.getenv("MCP_SSE_READ_TIMEOUT_SEC", "300"))

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return anyio.run(self._call_async, name, args)

    async def _call_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        async with sse_client(
            self.url,
            timeout=self.timeout_sec,
            sse_read_timeout=self.read_timeout_sec,
        ) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.read_timeout_sec),
            ) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=args)
                if result.isError:
                    raise RuntimeError(result)
                if result.structuredContent is not None:
                    payload = result.structuredContent
                    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
                        return payload["result"]
                    return payload
                # Fallback to JSON in text content if present.
                for item in result.content:
                    if getattr(item, "type", None) == "text":
                        try:
                            payload = json.loads(item.text)
                            if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
                                return payload["result"]
                            return payload
                        except Exception:
                            break
                raise RuntimeError("MCP SSE response missing structured content")


def _transport() -> str:
    return (os.getenv("MCP_TRANSPORT") or "http").strip().lower()


def _make_client(url: str, timeout_sec: Optional[float] = None) -> MCPHttpClient | MCPSSEClient:
    if _transport() == "sse":
        return MCPSSEClient(url, timeout_sec=timeout_sec)
    return MCPHttpClient(url, timeout_sec=timeout_sec)


class AssetClient:
    def __init__(self) -> None:
        self.url = _env_url("MCP_ASSETS_URL")
        if self.url:
            self.client = _make_client(self.url)
            self.mode = _transport()
        else:
            self.store = ArtifactStore()
            self.mode = "local"

    def put(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        tags: Optional[Iterable[str]] = None,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        if self.mode != "local":
            tag_list = list(tags or [])
            kind = kind or (tag_list[0] if tag_list else "artifact")
            payload = {
                "data_b64": base64.b64encode(data).decode("ascii"),
                "kind": kind,
                "tags": tag_list,
                "name": name,
                "description": description,
                "content_type": content_type,
            }
            res = self.client.call("asset_put", payload)
            asset_id = res.get("asset_id")
            if not asset_id:
                raise RuntimeError("asset_put missing asset_id")
            return asset_id
        return self.store.put(data=data, content_type=content_type, tags=tags)

    def get_path(self, artifact_id: str) -> str:
        if self.mode != "local":
            res = self.client.call("asset_get", {"asset_id": artifact_id})
            path = res.get("path")
            if not path:
                raise FileNotFoundError(artifact_id)
            return path
        return self.store.get_path(artifact_id)


class RenderClient:
    def __init__(self) -> None:
        self.url = _env_url("MCP_RENDER_URL")
        if self.url:
            self.client = _make_client(self.url, timeout_sec=os.getenv("MCP_RENDER_TIMEOUT_SEC"))
            self.mode = _transport()
        else:
            self.service = RenderService()
            self.mode = "local"

    def render_preview(self, timeline_uri: str) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call("render_preview", {"timeline_uri": timeline_uri})
        return self.service.render_preview(timeline_uri)

    def render_final(self, timeline_uri: str) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call("render_final", {"timeline_uri": timeline_uri})
        return self.service.render_final(timeline_uri)


class QCClient:
    def __init__(self) -> None:
        self.url = _env_url("MCP_QC_URL")
        if self.url:
            self.client = _make_client(self.url, timeout_sec=os.getenv("MCP_QC_TIMEOUT_SEC"))
            self.mode = _transport()
        else:
            self.service = QCService()
            self.mode = "local"

    def qc_audio(self, video_uri: str) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call("qc_audio", {"video_uri": video_uri})
        return self.service.qc_audio(video_uri)

    def qc_video(self, video_uri: str) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call("qc_video", {"video_uri": video_uri})
        return self.service.qc_video(video_uri)

    def qc_timeline_validate(self, timeline_json: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call("qc_timeline_validate", {"timeline_json": timeline_json})
        return self.service.qc_timeline_validate(timeline_json)


class LipSyncClient:
    def __init__(self) -> None:
        self.url = _env_url("MCP_LIPSYNC_URL")
        if self.url:
            self.client = _make_client(self.url, timeout_sec=os.getenv("MCP_LIPSYNC_TIMEOUT_SEC"))
            self.mode = _transport()
        else:
            from mcp_servers.lipsync.server import LipSyncService

            self.service = LipSyncService()
            self.mode = "local"

    def lipsync_render_clip(
        self,
        avatar_id: str,
        wav_id: str,
        style: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.mode != "local":
            return self.client.call(
                "lipsync_render_clip",
                {"avatar_id": avatar_id, "wav_uri": wav_id, "style": style},
            )
        return self.service.lipsync_render_clip(avatar_id=avatar_id, wav_id=wav_id, style=style)
