#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMELINE="$ROOT_DIR/runtime/sample/timeline.json"

python - <<'PY'
import json
from mcp_servers.render.server import RenderService
from mcp_servers.qc.server import QCService

render = RenderService()
qc = QCService()

result = render.render_preview("runtime/sample/timeline.json")
print("render_preview:", json.dumps(result, indent=2))

artifact_id = result["artifact_id"]
print("qc_audio:", json.dumps(qc.qc_audio(artifact_id), indent=2))
print("qc_video:", json.dumps(qc.qc_video(artifact_id), indent=2))
PY
