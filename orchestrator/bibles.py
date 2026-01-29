"""Load series and character bibles for consistency controls."""
import json
import os
from typing import Any, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SERIES_BIBLE = os.path.join(BASE_DIR, "data", "bibles", "series_bible.json")
DEFAULT_CHARACTER_BIBLE = os.path.join(BASE_DIR, "data", "bibles", "character_bible.json")


def load_series_bible(path: str | None = None) -> Dict[str, Any]:
    bible_path = path or os.getenv("SERIES_BIBLE_PATH", DEFAULT_SERIES_BIBLE)
    if not os.path.exists(bible_path):
        return {}
    with open(bible_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_character_bible(path: str | None = None) -> Dict[str, Any]:
    bible_path = path or os.getenv("CHARACTER_BIBLE_PATH", DEFAULT_CHARACTER_BIBLE)
    if not os.path.exists(bible_path):
        return {}
    with open(bible_path, "r", encoding="utf-8") as f:
        return json.load(f)
