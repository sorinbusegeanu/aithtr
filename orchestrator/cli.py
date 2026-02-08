"""CLI entrypoint for daily episode runs."""
import argparse
import json
import os
import shutil
from typing import Any, Dict

from .pipeline import EpisodeConfig, Orchestrator


def run_daily_episode(args: argparse.Namespace) -> Dict[str, Any]:
    orch = Orchestrator()
    config = EpisodeConfig(
        theme=args.theme,
        mood=args.mood,
        duration_sec=args.duration,
        auto_approve=args.auto_approve,
        seed=args.seed,
        transcript=args.transcript,
        transcript_path=args.transcript_path,
        resume_run_id=args.resume_run_id,
        resume_from_step=args.resume_from_step,
        force_regenerate_voices=args.force_regenerate_voices,
    )
    manifest = orch.run_daily_episode(config)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "episode_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    preview_id = manifest["artifacts"]["preview_mp4"]
    episode_id = manifest["artifacts"]["episode_mp4"]
    preview_path = _resolve_artifact_path(orch, preview_id)
    episode_path = _resolve_artifact_path(orch, episode_id)
    run_preview_path, run_episode_path = _write_run_playables(
        data_root=os.getenv("DATA_ROOT", "data"),
        episode_id=str(manifest.get("episode_id", "")).strip(),
        preview_src=preview_path,
        episode_src=episode_path,
    )

    print(out_path)
    print("preview_mp4:", run_preview_path)
    print("episode_mp4:", run_episode_path)
    return manifest


def _resolve_artifact_path(orch: Orchestrator, artifact_id: str) -> str:
    try:
        return orch.store.get_path(artifact_id)
    except Exception:
        return str(artifact_id)


def _write_run_playables(data_root: str, episode_id: str, preview_src: str, episode_src: str) -> tuple[str, str]:
    run_dir = os.path.abspath(os.path.join(data_root, "runs", episode_id))
    os.makedirs(run_dir, exist_ok=True)
    run_preview = os.path.join(run_dir, "preview.mp4")
    run_episode = os.path.join(run_dir, "episode.mp4")
    if os.path.isfile(preview_src):
        shutil.copy2(preview_src, run_preview)
    if os.path.isfile(episode_src):
        shutil.copy2(episode_src, run_episode)
    return run_preview, run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run daily episode pipeline")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--mood", required=True)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--output-dir", default=os.getenv("DATA_ROOT", "data"))
    parser.add_argument("--transcript", action="store_true", help="Write a chat-style transcript per run")
    parser.add_argument("--transcript-path", default=None, help="Override transcript log path")
    parser.add_argument(
        "--resume",
        "--resume-run-id",
        dest="resume_run_id",
        default=None,
        help="Resume an existing run id in data/runs",
    )
    parser.add_argument(
        "--resume-from-step",
        default=None,
        help="Re-run from this step onward (showrunner, writer, dramaturg, casting, scene, director, voice_seeder, performance, editor, render_preview, render_final, qc, curator)",
    )
    parser.add_argument(
        "--force-regenerate-voices",
        action="store_true",
        help="Regenerate all Piper seed speaker WAVs before XTTS performance generation",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_daily_episode(args)


if __name__ == "__main__":
    main()
