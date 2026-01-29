#!/usr/bin/env python
import os
import subprocess

from mcp_servers.qwen_tts.server import QwenTTSService


def main() -> int:
    if os.getenv("QWEN_TTS_TEST", "0") != "1":
        print("SKIP: set QWEN_TTS_TEST=1 to run")
        return 0

    service = QwenTTSService()
    result = service.tts_synthesize(
        text="Hello from Qwen TTS.",
        character_id="test_character",
        emotion="happy",
        style=None,
        output_format="wav",
    )
    wav_path = result["wav_path"]
    duration_ms = result["duration_ms"]

    assert duration_ms > 0, "duration_ms should be > 0"

    ffprobe_bin = os.getenv("FFPROBE_BIN", "ffprobe")
    proc = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", wav_path],
        capture_output=True,
        check=True,
    )
    duration = float(proc.stdout.decode("utf-8").strip() or 0.0)
    assert duration > 0, "ffprobe duration should be > 0"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
