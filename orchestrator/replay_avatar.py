import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from mcp_servers.lipsync.server import LipSyncService


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _artifact_id_from_any(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    base = os.path.basename(text)
    if len(base) == 64 and all(c in "0123456789abcdef" for c in base):
        return base
    stem, _ = os.path.splitext(base)
    if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem):
        return stem
    return text


def _find_line_context(
    runs_root: str,
    *,
    line_id: str,
    line_video_artifact_id: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    for run_id in sorted(os.listdir(runs_root), reverse=True):
        perf_path = os.path.join(runs_root, run_id, "performance", "normalized.json")
        if not os.path.exists(perf_path):
            continue
        with open(perf_path, "r", encoding="utf-8") as f:
            perf = json.load(f)
        scenes = perf.get("scenes", []) if isinstance(perf, dict) else []
        for scene in scenes:
            chars = scene.get("characters", []) if isinstance(scene, dict) else []
            for row in chars:
                if not isinstance(row, dict):
                    continue
                for lv in row.get("line_video_artifacts", []) if isinstance(row.get("line_video_artifacts"), list) else []:
                    if not isinstance(lv, dict):
                        continue
                    if str(lv.get("line_id") or "") != line_id:
                        continue
                    if str(lv.get("video_artifact_id") or "") != line_video_artifact_id:
                        continue
                    return run_id, scene, row
    raise RuntimeError(
        f"Could not find line_id={line_id!r} video_artifact_id={line_video_artifact_id!r} in data/runs/*/performance/normalized.json"
    )


def _find_line_audio_artifact(row: Dict[str, Any], line_id: str) -> str:
    for la in row.get("line_audio_artifacts", []) if isinstance(row.get("line_audio_artifacts"), list) else []:
        if isinstance(la, dict) and str(la.get("line_id") or "") == line_id:
            aid = str(la.get("wav_artifact_id") or "").strip()
            if aid:
                return aid
    raise RuntimeError(f"Missing line audio artifact for line_id={line_id!r}")


def _load_avatar_id_for_character(runs_root: str, run_id: str, character_id: str) -> str:
    for candidate in ("normalized_post_critic.json", "normalized.json"):
        cast_path = os.path.join(runs_root, run_id, "casting", candidate)
        if not os.path.exists(cast_path):
            continue
        with open(cast_path, "r", encoding="utf-8") as f:
            cast = json.load(f)
        roles = (
            cast.get("cast_plan", {}).get("roles", [])
            if isinstance(cast, dict)
            else []
        )
        for role in roles if isinstance(roles, list) else []:
            if not isinstance(role, dict):
                continue
            if str(role.get("character_id") or "") == character_id:
                avatar_id = str(role.get("avatar_id") or "").strip()
                if avatar_id:
                    return avatar_id
    raise RuntimeError(f"Missing avatar_id for character_id={character_id!r} in run {run_id}")


def run_replay(args: argparse.Namespace) -> Dict[str, Any]:
    runs_root = os.path.join(_project_root(), "data", "runs")
    line_video_artifact_id = _artifact_id_from_any(args.artifact_path)
    if not line_video_artifact_id:
        raise RuntimeError("artifact id/path is required")
    line_id = str(args.line_id or "").strip()
    if not line_id:
        raise RuntimeError("line_id is required")

    run_id, scene, row = _find_line_context(
        runs_root,
        line_id=line_id,
        line_video_artifact_id=line_video_artifact_id,
    )
    character_id = str(row.get("character_id") or "").strip()
    if not character_id:
        raise RuntimeError("Performance row missing character_id")
    line_wav_artifact_id = _find_line_audio_artifact(row, line_id)
    avatar_id = _load_avatar_id_for_character(runs_root, run_id, character_id)

    replay_dir = args.output_dir or os.path.join(
        runs_root,
        run_id,
        "debug",
        "avatar_replay",
        f"{line_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    )
    os.makedirs(replay_dir, exist_ok=True)

    style = {"debug_dir": replay_dir}
    service = LipSyncService()
    replay_result = service.lipsync_render_clip(avatar_id=avatar_id, wav_id=line_wav_artifact_id, style=style)
    replay_artifact_id = str(replay_result.get("artifact_id") or "").strip()
    replay_artifact_path = service.store.get_path(replay_artifact_id) if replay_artifact_id else ""
    if replay_artifact_path:
        shutil.copy2(replay_artifact_path, os.path.join(replay_dir, "replay_output.mp4"))

    output = {
        "run_id": run_id,
        "scene_id": scene.get("scene_id"),
        "line_id": line_id,
        "character_id": character_id,
        "source_line_video_artifact_id": line_video_artifact_id,
        "source_line_wav_artifact_id": line_wav_artifact_id,
        "avatar_id": avatar_id,
        "replay_artifact_id": replay_artifact_id,
        "replay_dir": replay_dir,
        "replay_diagnostics": replay_result.get("diagnostics"),
    }
    with open(os.path.join(replay_dir, "replay_result.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay avatar rendering for a specific line artifact")
    parser.add_argument("--artifact-path", required=True, help="line video artifact id or path")
    parser.add_argument("--line-id", required=True, help="line id (e.g. line-2)")
    parser.add_argument("--output-dir", default=None, help="optional output directory")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_replay(args)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
