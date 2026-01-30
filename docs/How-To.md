# How-To

## Main Modules
- `orchestrator/` AutoGen-based supervisor and run graph.
- `agents/` Specialist agents (showrunner, writer, dramaturg, casting, scene, director, editor, qc, curator).
- `mcp_servers/` MCP servers for assets, TTS, lipsync, render, QC, memory.
- `schemas/` JSON schemas for all agent I/O and artifacts.
- `runtime/` Docker-compose, env templates, deployment wiring.
- `data/` Local artifacts, memory stores, and sqlite.

## Container Images
- Distinct images referenced across compose files: 2
  - `vllm/vllm-openai:latest` (root `docker-compose.yml`)
  - `python:3.11-slim` (all 6 services in `runtime/docker-compose.yml`)
- Services built from local Dockerfiles (root `docker-compose.yml`): orchestrator and MCP servers.

## Start the Containers
From the repo root:

Full stack (GPU-enabled vLLM + orchestrator + MCP services):
```bash
docker compose -f docker-compose.yml up --build
```

Local MCP servers over HTTP (runtime helpers):
```bash
docker compose -f runtime/docker-compose.yml up
```
