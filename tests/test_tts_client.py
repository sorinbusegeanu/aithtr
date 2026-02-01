import urllib.error
import urllib.request

import pytest

from orchestrator.tts_client import QwenTTSClient


def test_tts_client_propagates_url_error(monkeypatch):
    client = QwenTTSClient()

    def _fail(*_args, **_kwargs):
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)

    with pytest.raises(urllib.error.URLError):
        client.tts_synthesize("hello", "char_1", "neutral")
