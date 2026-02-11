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
    min_line_sec = max(float(os.getenv("TTS_MIN_SPLIT_LINE_DURATION_SEC", "0.6")), 0.2)
    pause_total = pause_sec * (n - 1)
    if total_duration_sec <= 0:
        return [{"start_sec": 0.0, "duration_sec": 0.0} for _ in lines]

    if n > 1 and pause_total >= total_duration_sec:
        pause_sec = max(0.0, total_duration_sec / (n - 1))
        pause_total = pause_sec * (n - 1)
    # Prefer minimum per-line durations. If impossible, reduce pause first.
    required_min_total = min_line_sec * n + pause_total
    if required_min_total > total_duration_sec and n > 1:
        pause_sec = max(0.0, (total_duration_sec - min_line_sec * n) / (n - 1))
        pause_total = pause_sec * (n - 1)
        required_min_total = min_line_sec * n + pause_total

    usable = max(total_duration_sec - pause_total, 0.001 * n)
    weights = [max(len((line.get("text") or "").strip()), 1) for line in lines]
    total_weight = sum(weights) or 1

    durations = [usable * (weight / total_weight) for weight in weights]
    if required_min_total <= total_duration_sec:
        deficit_idx = [i for i, d in enumerate(durations) if d < min_line_sec]
        if deficit_idx:
            needed = sum(min_line_sec - durations[i] for i in deficit_idx)
            surplus_idx = [i for i, d in enumerate(durations) if d > min_line_sec]
            for i in surplus_idx:
                if needed <= 0:
                    break
                take = min(durations[i] - min_line_sec, needed)
                durations[i] -= take
                needed -= take
            for i in deficit_idx:
                durations[i] = min_line_sec
    else:
        # Not enough total time for strict minimums; use uniform durations.
        durations = [max(usable / n, 0.001) for _ in lines]

    segments: List[Dict[str, float]] = []
    cursor = 0.0
    for dur in durations:
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
    split_pad = max(float(os.getenv("AVATAR_SPLIT_PAD_SEC", "0.15")), 0.0)
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

        # Add small context padding so very short clips still contain visible motion.
        start = max(0.0, start - (split_pad * 0.5))
        duration = max(duration + split_pad, min_duration, 0.2)
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
