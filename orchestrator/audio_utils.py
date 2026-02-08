"""Audio helpers for batching and splitting."""
import os
import subprocess
import tempfile
import wave
from typing import Dict, Iterable, List, Tuple


def build_segments(
    lines: List[Dict[str, str]],
    total_duration_sec: float,
    pause_sec: float,
) -> List[Dict[str, float]]:
    """Compute per-line segments within a single synthesized waveform."""
    if not lines:
        return []
    n = len(lines)
    pause_total = pause_sec * (n - 1)
    if total_duration_sec <= 0:
        return [{"start_sec": 0.0, "duration_sec": 0.0} for _ in lines]

    if pause_total >= total_duration_sec and n > 1:
        pause_sec = max(0.0, total_duration_sec / (n - 1))
        pause_total = pause_sec * (n - 1)

    usable = max(total_duration_sec - pause_total, 0.001 * n)
    weights = [max(len((line.get("text") or "").strip()), 1) for line in lines]
    total_weight = sum(weights) or 1

    segments: List[Dict[str, float]] = []
    cursor = 0.0
    for weight in weights:
        dur = usable * (weight / total_weight)
        segments.append({"start_sec": cursor, "duration_sec": dur})
        cursor += dur + pause_sec
    return segments


def split_wav_ffmpeg(
    input_path: str,
    segments: Iterable[Dict[str, float]],
    sample_rate: int = 48000,
) -> List[str]:
    """Split a wav into segments using ffmpeg, returning file paths."""
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
    min_duration = max(float(os.getenv("TTS_MIN_LINE_DURATION_SEC", "0.2")), 0.25)
    outputs: List[str] = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start_sec", 0.0))
        duration = float(seg.get("duration_sec", 0.0))
        # Extremely short segments (e.g. 1ms) can fail with some ffmpeg/wav builds.
        # Clamp to a practical minimum so split commands are stable.
        if duration <= 0:
            # create a 1-sample silent file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=mono:sample_rate=%d" % sample_rate,
                "-t",
                f"{max(min_duration, 0.2):.3f}",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                out_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outputs.append(out_path)
            continue

        duration = max(duration, min_duration, 0.2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            input_path,
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-af",
            f"apad=whole_dur={duration:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Fallback to silence so downstream timeline integrity remains intact.
            _write_silence_wav(ffmpeg_bin, out_path, sample_rate, duration)
        if _wav_duration_sec(out_path) < min_duration:
            _write_silence_wav(ffmpeg_bin, out_path, sample_rate, duration)
        outputs.append(out_path)
    return outputs


def _wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def _write_silence_wav(ffmpeg_bin: str, out_path: str, sample_rate: int, duration: float) -> None:
    cmd_silence = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=%d" % sample_rate,
        "-t",
        f"{max(duration, 0.25):.3f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    subprocess.run(cmd_silence, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
