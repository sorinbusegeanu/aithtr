"""
QC wrapper using ffprobe/ffmpeg filters.
"""
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from mcp_servers.assets.artifact_store import ArtifactStore


class QCService:
    def __init__(self, artifact_root: Optional[str] = None) -> None:
        self.store = ArtifactStore(root=artifact_root)

    def qc_audio(self, video_uri: str) -> Dict[str, Any]:
        path = self._resolve(video_uri)
        info = self._ffprobe(path)
        silence = self._detect_silence(path)
        return {
            "duration_sec": info.get("duration_sec"),
            "clipping": False,
            "silence_gaps": silence,
        }

    def qc_video(self, video_uri: str) -> Dict[str, Any]:
        path = self._resolve(video_uri)
        info = self._ffprobe(path)
        black_frames = self._detect_black(path)
        return {
            "duration_sec": info.get("duration_sec"),
            "fps": info.get("fps"),
            "black_frames": black_frames,
        }

    def qc_timeline_validate(self, timeline_json: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        if "scenes" not in timeline_json:
            errors.append("missing_scenes")
        if "duration_sec" not in timeline_json:
            errors.append("missing_duration")
        return {"ok": not errors, "errors": errors}

    def _resolve(self, uri: str) -> str:
        if os.path.exists(uri):
            return uri
        return self.store.get_path(uri)

    def _ffprobe(self, path: str) -> Dict[str, Any]:
        ffprobe_bin = os.getenv("FFPROBE_BIN", "ffprobe")
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True)
        data = json.loads(result.stdout.decode("utf-8"))
        duration = float(data.get("format", {}).get("duration", 0.0))
        fps = None
        streams = data.get("streams", [])
        if streams:
            rate = streams[0].get("r_frame_rate", "0/1")
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        return {"duration_sec": duration, "fps": fps}

    def _detect_silence(self, path: str) -> List[Dict[str, float]]:
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        noise = os.getenv("SILENCE_NOISE", "-35dB")
        duration = os.getenv("SILENCE_DURATION", "0.5")
        cmd = [
            ffmpeg_bin,
            "-i",
            path,
            "-af",
            f"silencedetect=noise={noise}:d={duration}",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        stderr = result.stderr.decode("utf-8")
        starts = [float(x) for x in re.findall(r"silence_start: ([0-9\.]+)", stderr)]
        ends = [float(x) for x in re.findall(r"silence_end: ([0-9\.]+)", stderr)]
        gaps = []
        for i, start in enumerate(starts):
            end = ends[i] if i < len(ends) else start
            gaps.append({"start_sec": start, "end_sec": end})
        return gaps

    def _detect_black(self, path: str) -> List[Dict[str, float]]:
        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        threshold = os.getenv("BLACK_THRESHOLD", "0.10")
        duration = os.getenv("BLACK_DURATION", "0.2")
        cmd = [
            ffmpeg_bin,
            "-i",
            path,
            "-vf",
            f"blackdetect=d={duration}:pic_th={threshold}",
            "-an",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        stderr = result.stderr.decode("utf-8")
        starts = [float(x) for x in re.findall(r"black_start:([0-9\.]+)", stderr)]
        ends = [float(x) for x in re.findall(r"black_end:([0-9\.]+)", stderr)]
        ranges = []
        for i, start in enumerate(starts):
            end = ends[i] if i < len(ends) else start
            ranges.append({"start_sec": start, "end_sec": end})
        return ranges
