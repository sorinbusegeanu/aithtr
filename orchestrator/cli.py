"""CLI entrypoint for daily episode runs."""
import argparse
import json
import os
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
    )
    manifest = orch.run_daily_episode(config)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "episode_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(out_path)
    print("preview_mp4:", manifest["artifacts"]["preview_mp4"])
    print("episode_mp4:", manifest["artifacts"]["episode_mp4"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run daily episode pipeline")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--mood", required=True)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--output-dir", default="data")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_daily_episode(args)


if __name__ == "__main__":
    main()
