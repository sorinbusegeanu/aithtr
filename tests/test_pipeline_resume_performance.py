from __future__ import annotations

from orchestrator.pipeline import Orchestrator


class _FakeLogger:
    def __init__(self) -> None:
        self.logged: list[str] = []

    def log(self, message: str) -> None:
        self.logged.append(message)

    def write_json(self, _path: str, _data) -> None:
        return None

    def step_dir(self, _step: str) -> str:
        return "."


def test_resume_or_run_performances_reruns_when_resumed_output_has_no_success(monkeypatch):
    orch = Orchestrator.__new__(Orchestrator)
    logger = _FakeLogger()

    resumed = {
        "scenes": [
            {
                "scene_id": "scene-1",
                "characters": [
                    {
                        "character_id": "A",
                        "status": "failed",
                        "error_code": "LIPSYNC_FAILED",
                    }
                ],
            }
        ]
    }
    rerun = {
        "scenes": [
            {
                "scene_id": "scene-1",
                "characters": [
                    {
                        "character_id": "A",
                        "status": "ok",
                        "wav_artifact_id": "wav",
                        "video_artifact_id": "vid",
                    }
                ],
            }
        ]
    }

    monkeypatch.setattr(orch, "_should_resume_step", lambda *_: True)
    monkeypatch.setattr(orch, "_load_step_output", lambda *_: resumed)
    monkeypatch.setattr(orch, "_run_performances", lambda *_: rerun)

    out = orch._resume_or_run_performances(
        screenplay={},
        cast_plan={},
        logger=logger,
        resume_from=None,
        force=False,
    )
    assert out == rerun
    assert any("re-running performance step" in msg for msg in logger.logged)
