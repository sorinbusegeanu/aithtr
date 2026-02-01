from __future__ import annotations

from typing import Any, Dict

import pytest

from agents import common as agents_common
from agents.casting import agent as casting_agent
from agents.curator import agent as curator_agent
from agents.director import agent as director_agent
from agents.dramaturg import agent as dramaturg_agent
from agents.editor import agent as editor_agent
from agents.qc import agent as qc_agent
from agents.scene import agent as scene_agent
from agents.showrunner import agent as showrunner_agent
from agents.writer import agent as writer_agent


class DummyLLM(agents_common.LLMClient):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__()
        self._payload = payload

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        self.last_prompt = prompt
        return self._payload


def test_showrunner_agent():
    payload = {"premise": "x", "beats": [], "tone": "funny", "cast_constraints": {}}
    res = showrunner_agent.run({"theme": "space"}, llm=DummyLLM(payload))
    assert res["premise"] == "x"


def test_writer_agent():
    payload = {"scenes": [{"scene_id": "s1", "setting_prompt": "", "characters": [], "lines": []}]}
    res = writer_agent.run({"episode_brief": {}}, llm=DummyLLM(payload))
    assert "scenes" in res


def test_dramaturg_agent():
    payload = {"required_edits": [], "suggested_changes": [], "duration_targets": {}}
    res = dramaturg_agent.run({"screenplay": {}}, llm=DummyLLM(payload))
    assert "required_edits" in res


def test_casting_agent():
    payload = {"cast_plan": {"roles": []}, "cast_bible_update": {}}
    res = casting_agent.run({"screenplay": {}}, llm=DummyLLM(payload))
    assert "cast_plan" in res


def test_scene_agent():
    payload = {"scenes": []}
    res = scene_agent.run({"screenplay": {}}, llm=DummyLLM(payload))
    assert "scenes" in res


def test_director_agent():
    payload = {"scenes": []}
    res = director_agent.run({"screenplay": {}}, llm=DummyLLM(payload))
    assert "scenes" in res


def test_editor_agent():
    payload = {"duration_sec": 1, "scenes": []}
    res = editor_agent.run({"screenplay": {}}, llm=DummyLLM(payload))
    assert "duration_sec" in res


def test_qc_agent():
    payload = {"duration_sec": 1, "audio": {}, "video": {}, "subtitles": {}, "errors": []}
    res = qc_agent.run({"duration_sec": 1}, llm=DummyLLM(payload))
    assert "errors" in res


def test_curator_agent():
    payload = {"updates": [], "embeddings": []}
    res = curator_agent.run({"updates": []}, llm=DummyLLM(payload))
    assert "updates" in res
