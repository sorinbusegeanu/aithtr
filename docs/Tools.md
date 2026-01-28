# Tools.md

Open-source tools only. Each tool runs as an **MCP server** inside a **Docker container**. The orchestrator talks to them over MCP transports (stdio or HTTP/SSE), using the official MCP Python SDK.

---

## A) Core orchestration + MCP

### MCP SDK (server/client)
- GitHub: https://github.com/modelcontextprotocol/python-sdk

### Agent orchestration (pick one)
- LangGraph (graph orchestration, HITL interrupts): https://github.com/langchain-ai/langgraph
- AutoGen (multi-agent framework): https://github.com/microsoft/autogen
- CrewAI (multi-agent framework): https://github.com/crewAIInc/crewAI

Recommendation for “fully automated but deterministic”: **LangGraph** for the orchestrator + checkpointing.

---

## B) Memory store (supports learning)

You need 3 stores:
1) **Relational** (episode metadata, manifests, critique labels)
2) **Object store** (all media artifacts)
3) **Vector store** (semantic retrieval over scripts, critiques, tags, transcripts)

### 1) Relational DB (metadata)
- PostgreSQL (Docker official image):
  - https://hub.docker.com/_/postgres

### 2) Object store (media artifacts)
Option 1:
- MinIO (S3-compatible object store): https://github.com/minio/minio
Option 2 (alternative object store):
- SeaweedFS: https://github.com/seaweedfs/seaweedfs

### 3) Vector DB (semantic memory)
Option 1:
- Qdrant: https://github.com/qdrant/qdrant
- Official Qdrant MCP server: https://github.com/qdrant/mcp-server-qdrant
Option 2:
- Chroma: https://github.com/chroma-core/chroma
- Chroma MCP server: https://github.com/chroma-core/chroma-mcp

### Memory schema (minimum)
- `episodes` table: id, date, brief, duration, status
- `assets` table: URIs, hashes, types
- `critiques` table: gate, labels, free-text notes, accepted(bool)
- `memories` (vector store):
  - items: {type: screenplay|critique|qc|summary|character_bible, text, tags, episode_id, timestamp}

**Learning loop**
- Store: approved screenplay, final timeline, critique notes, QC report
- Retrieve: top-k relevant prior critique + similar episodes during planning/writing/casting

---

## C) Media generation tools (audio, lipsync, visuals)

### 1) Text-to-Speech (TTS) MCP server
Fast/local:
- Piper TTS: https://github.com/rhasspy/piper

Higher flexibility / training:
- Coqui TTS: https://github.com/coqui-ai/TTS

Server responsibilities:
- `tts.synthesize(text, voice_id, style)` → WAV
- optional `tts.list_voices()` → voice catalog

### 2) Lip-sync / talking head MCP server
Baseline (lip sync expert):
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip

More motion (heavier):
- SadTalker: https://github.com/OpenTalker/SadTalker

Server responsibilities:
- `lipsync.render_clip(avatar_id, wav_uri, style)` → MP4/WebM (optionally alpha)

### 3) Face restoration / enhancement MCP server (optional)
- GFPGAN: https://github.com/TencentARC/GFPGAN
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN

### 4) Speech-to-text for subtitles / QC (optional)
- Whisper: https://github.com/openai/whisper
Faster inference:
- faster-whisper: https://github.com/SYSTRAN/faster-whisper

Server responsibilities:
- `asr.transcribe(wav_uri)` → segments + text
- used for subtitle alignment + QC checks

### 5) Backgrounds (choose one strategy)
Asset-library (recommended for stability):
- Your own curated pack in the Asset Store (no external dependency)

Optional open-source image generation (heavier, more variance):
- Stable Diffusion (A1111): https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Stable Diffusion (ComfyUI): https://github.com/comfyanonymous/ComfyUI

Server responsibilities:
- `imagegen.generate_background(prompt, seed)` → PNG

---

## D) Compositing / rendering

### FFmpeg (core)
- FFmpeg: https://github.com/FFmpeg/FFmpeg

### Python convenience (optional)
- MoviePy: https://github.com/Zulko/moviepy

Renderer MCP server responsibilities:
- `render.preview(timeline_json_uri)` → MP4 (fast preset)
- `render.final(timeline_json_uri)` → MP4 (final preset)
- `render.thumbnail(episode_mp4_uri)` → PNG

---

## E) QC / analysis

Implement as a single QC MCP server wrapping open-source utilities:

Audio checks:
- FFmpeg/ffprobe (silence detect, peaks)
Video checks:
- ffprobe (black frames, duration)
Subtitles:
- ffmpeg subtitles burn-in or ASS render

QC MCP server responsibilities:
- `qc.audio(episode_uri)` → clipping/silence report
- `qc.video(episode_uri)` → black frames/fps/duration
- `qc.timeline_validate(timeline_json)` → structural validation

---

## F) Asset management (MCP server)

A custom MCP server backed by object storage.

Responsibilities:
- `asset.put(bytes, kind, tags)` → `asset_id`
- `asset.get(asset_id)` → signed URL or stream
- `asset.search(tags|text)` → matches

---

## G) Containerization (runtime model)

Each service runs independently as:
- Docker image
- one exposed MCP transport (HTTP/SSE recommended)
- GPU access only for TTS/lipsync/enhance/imagegen (as needed)

Minimum service set to ship:
1) `mcp-orchestrator` (agents)
2) `mcp-memory` (Qdrant MCP + Postgres + object store)
3) `mcp-tts` (Piper or Coqui)
4) `mcp-lipsync` (Wav2Lip)
5) `mcp-render` (FFmpeg)
6) `mcp-qc` (ffprobe/ffmpeg checks)
7) `mcp-assets` (object store wrapper)

---

## H) Security note (MCP servers)

Treat MCP servers like privileged tooling:
- run with least privileges
- isolate filesystem mounts
- only allow the orchestrator network access
- validate/escape all tool inputs
