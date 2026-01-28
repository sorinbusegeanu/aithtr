# runtime

Local runtime helpers.

## Compose
`runtime/docker-compose.yml` brings up MCP servers over HTTP. It installs `mcp` at container start.

## Demo
One-command render + QC (uses local Python + ffmpeg):

```
./runtime/demo.sh
```

Inputs:
- `runtime/sample/timeline.json`
- `runtime/sample/background.png`
- `runtime/sample/actor.png`
- `runtime/sample/subtitles.srt`
- `runtime/sample/tone.wav`
