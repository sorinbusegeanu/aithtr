"""
Minimal TTS wrapper. Expects an external engine (Piper or Coqui).
Wire into MCP Python SDK in a later phase.
"""
import os
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional

from mcp_servers.assets.artifact_store import ArtifactStore


class TTSService:
    def __init__(self, artifact_root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=artifact_root)

    def tts_list_voices(self) -> Dict[str, Any]:
        voices_path = os.getenv("TTS_VOICES_PATH")
        if voices_path and os.path.exists(voices_path):
            import json

            with open(voices_path, "r", encoding="utf-8") as f:
                return {"voices": json.load(f)}
        return {"voices": []}

    def tts_synthesize(
        self,
        text: str,
        voice_id: str,
        style: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        engine = os.getenv("TTS_ENGINE", "piper")
        if engine == "piper":
            return self._synthesize_piper(text=text, voice_id=voice_id, style=style, tags=tags)
        if engine == "coqui":
            return self._synthesize_coqui(text=text, voice_id=voice_id, style=style, tags=tags)
        raise ValueError(f"Unsupported TTS_ENGINE: {engine}")

    def _resolve_voice_path(self, voice_id: str) -> str:
        if os.path.exists(voice_id):
            return voice_id
        base = os.getenv("PIPER_MODEL_DIR", "")
        candidate = os.path.join(base, voice_id) if base else voice_id
        return candidate

    def _synthesize_piper(
        self,
        text: str,
        voice_id: str,
        style: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        piper_bin = os.getenv("PIPER_BIN", "piper")
        model_path = self._resolve_voice_path(voice_id)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [piper_bin, "--model", model_path, "--output_file", out_path]
        if style and "speaker" in style:
            cmd += ["--speaker", str(style["speaker"])]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        with open(out_path, "rb") as f:
            data = f.read()
        artifact_id = self.store.put(data=data, content_type="audio/wav", tags=tags)
        return {"artifact_id": artifact_id}

    def _synthesize_coqui(
        self,
        text: str,
        voice_id: str,
        style: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        tts_bin = os.getenv("COQUI_TTS_BIN", "tts")
        model_path = voice_id
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [tts_bin, "--model_name", model_path, "--text", text, "--out_path", out_path]
        subprocess.run(cmd, check=True)
        with open(out_path, "rb") as f:
            data = f.read()
        artifact_id = self.store.put(data=data, content_type="audio/wav", tags=tags)
        return {"artifact_id": artifact_id}
