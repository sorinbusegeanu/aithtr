"""Client for Coqui XTTSv2 service."""
import json
import os
import urllib.request
from typing import Any, Dict, Optional


class XTTSClient:
    def __init__(self) -> None:
        # Keep QWEN_TTS_URL as fallback for backward compatibility.
        self.url = (os.getenv("XTTS_URL") or os.getenv("QWEN_TTS_URL", "")).strip()
        transport = (os.getenv("XTTS_TRANSPORT") or os.getenv("MCP_TRANSPORT") or "").strip().lower()
        if transport == "local" or not self.url:
            from mcp_servers.qwen_tts.server import XTTSService

            self.service = XTTSService()
            self.mode = "local"
        else:
            self.mode = "http"

    def tts_list_voices(self) -> Dict[str, Any]:
        return self._call("tts_list_voices", {})

    def tts_synthesize(
        self,
        text: str,
        character_id: str,
        voice_id: Optional[str],
        emotion: str,
        style: Optional[str] = None,
        output_format: str = "wav",
    ) -> Dict[str, Any]:
        return self._call(
            "tts_synthesize",
            {
                "text": text,
                "character_id": character_id,
                "voice_id": voice_id,
                "emotion": emotion,
                "style": style,
                "output_format": output_format,
            },
        )

    def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "local":
            if name == "tts_list_voices":
                return self.service.tts_list_voices()
            if name == "tts_synthesize":
                return self.service.tts_synthesize(
                    text=args.get("text", ""),
                    character_id=args.get("character_id", "unknown"),
                    voice_id=args.get("voice_id"),
                    emotion=args.get("emotion", "neutral"),
                    style=args.get("style"),
                    output_format=args.get("output_format", "wav"),
                )
            raise ValueError(f"Unknown XTTS tool: {name}")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        if "error" in parsed:
            raise RuntimeError(parsed["error"])
        return parsed.get("result", {})


# Backward compatibility for older imports.
QwenTTSClient = XTTSClient
