"""Minimal HTTP JSON-RPC server for Qwen TTS tools."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from .server import QwenTTSService


service = QwenTTSService()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload: Any = None
        try:
            payload = json.loads(body)
            result = handle_rpc(payload)
            response = {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": payload.get("id") if isinstance(payload, dict) else None,
                "error": {"message": str(exc)},
            }

        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def handle_rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = payload.get("method")
    params = payload.get("params", {})
    if method != "tools/call":
        raise ValueError("Unsupported method")

    name = params.get("name")
    args = params.get("arguments", {})

    if name == "tts_list_voices":
        return service.tts_list_voices()
    if name == "tts_synthesize":
        return service.tts_synthesize(
            text=args.get("text", ""),
            character_id=args.get("character_id", "unknown"),
            emotion=args.get("emotion", "neutral"),
            style=args.get("style"),
            output_format=args.get("output_format", "wav"),
        )
    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "7203"))
    server = HTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
