"""
Minimal Wav2Lip wrapper. Expects an external Wav2Lip install.
Wire into MCP Python SDK in a later phase.
"""
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from mcp_servers.assets.artifact_store import ArtifactStore


class LipSyncService:
    def __init__(self, artifact_root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=artifact_root)

    def lipsync_render_clip(
        self,
        avatar_id: str,
        wav_id: str,
        style: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        wav2lip_py = os.getenv("WAV2LIP_PYTHON", "python")
        wav2lip_script = os.getenv("WAV2LIP_SCRIPT", "")
        if not wav2lip_script:
            raise ValueError("WAV2LIP_SCRIPT is required")
        avatar_path = self.store.get_path(avatar_id)
        wav_path = self.store.get_path(wav_id)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            wav2lip_py,
            wav2lip_script,
            "--face",
            avatar_path,
            "--audio",
            wav_path,
            "--outfile",
            out_path,
        ]
        if style and "pads" in style:
            cmd += ["--pads", str(style["pads"])]
        subprocess.run(cmd, check=True)
        with open(out_path, "rb") as f:
            data = f.read()
        artifact_id = self.store.put(data=data, content_type="video/mp4", tags=["lipsync"])
        return {"artifact_id": artifact_id}
