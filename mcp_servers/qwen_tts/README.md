# qwen_tts

Coqui XTTSv2 MCP server.

## MCP methods
- `tts.list_voices()`
- `tts.synthesize(text, character_id, emotion, style, output_format="wav")`

## Env
- `XTTS_MODEL` (default: tts_models/multilingual/multi-dataset/xtts_v2)
- `XTTS_DEVICE` (`cuda` or `cpu`)
- `XTTS_LANGUAGE` (default: `en`)
- `XTTS_SPEAKER_WAV_DIR` (optional directory with `<character_id>.wav` reference voices)
- `VOICE_MAP_PATH` (default: /data/tts/voice_map.json)
- `ARTIFACT_ROOT` (default: /data/artifacts)
- `ARTIFACT_AUDIO_ROOT` (default: /data/artifacts/audio)
