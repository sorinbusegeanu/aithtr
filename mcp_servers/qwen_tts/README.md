# qwen_tts

Qwen3-TTS MCP server.

## MCP methods
- `tts.list_voices()`
- `tts.synthesize(text, character_id, emotion, style, output_format="wav")`

## Env
- `QWEN_TTS_MODEL` (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- `QWEN_TTS_FLASH_ATTN` (1/0)
- `QWEN_TTS_SPEAKERS` (comma-separated override)
- `VOICE_MAP_PATH` (default: /data/tts/voice_map.json)
- `ARTIFACT_ROOT` (default: /data/artifacts)
- `ARTIFACT_AUDIO_ROOT` (default: /data/artifacts/audio)
