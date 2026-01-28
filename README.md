# AI Puppet Theatre

Automated pipeline to generate animated theatre scenes from text, producing MP4 episodes via modular agents and MCP servers.

## Structure
- `orchestrator/` AutoGen-based supervisor and run graph.
- `agents/` Specialist agents (showrunner, writer, dramaturg, casting, scene, director, editor, qc, curator).
- `mcp_servers/` MCP servers for assets, TTS, lipsync, render, QC, memory.
- `schemas/` JSON schemas for all agent I/O and artifacts.
- `runtime/` Docker-compose, env templates, deployment wiring.
- `data/` Local artifacts, memory stores, and sqlite.

## Status
Scaffolded repository layout with placeholder READMEs. Implementation plan pending.
