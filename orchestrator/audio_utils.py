"""Audio helpers for batching and splitting."""
import os
import subprocess
import tempfile
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
    sample_fmt: str = "flt",
) -> List[str]:
    """Split a wav into segments using ffmpeg, returning file paths."""
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
    outputs: List[str] = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start_sec", 0.0))
        duration = float(seg.get("duration_sec", 0.0))
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
                "0.001",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-sample_fmt",
                sample_fmt,
                out_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outputs.append(out_path)
            continue

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
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            sample_fmt,
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outputs.append(out_path)
    return outputs
