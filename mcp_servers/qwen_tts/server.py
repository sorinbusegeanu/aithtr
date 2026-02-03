"""Coqui XTTSv2 MCP service implementation."""
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import soundfile as sf

from mcp_servers.assets.artifact_store import ArtifactStore


DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"


@dataclass
class XTTSConfig:
    model_name: str
    device: str
    language: str


class VoiceMap:
    def __init__(self, path: str, speakers: List[str]) -> None:
        self.path = path
        self.speakers = speakers
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=True, indent=2)

    def load(self) -> Dict[str, str]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, mapping: Dict[str, str]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=True, indent=2)

    def resolve(self, character_id: str) -> Optional[str]:
        mapping = self.load()
        if character_id in mapping:
            return mapping[character_id]
        if not self.speakers:
            return None
        used = set(mapping.values())
        for speaker in self.speakers:
            if speaker not in used:
                mapping[character_id] = speaker
                self.save(mapping)
                return speaker
        speaker = self.speakers[0]
        mapping[character_id] = speaker
        self.save(mapping)
        return speaker


class XTTSService:
    def __init__(self) -> None:
        self.config = XTTSConfig(
            model_name=os.getenv("XTTS_MODEL", DEFAULT_MODEL),
            device=(os.getenv("XTTS_DEVICE") or ("cuda" if _cuda_available() else "cpu")).lower(),
            language=os.getenv("XTTS_LANGUAGE", DEFAULT_LANGUAGE),
        )
        self.model = None
        self.speakers: List[str] = []
        self._model_lock = threading.Lock()

        artifact_root = os.getenv("ARTIFACT_ROOT", "/data/artifacts")
        self.store = ArtifactStore(root=artifact_root)
        self.audio_root = os.getenv("ARTIFACT_AUDIO_ROOT", "/data/artifacts/audio")
        os.makedirs(self.audio_root, exist_ok=True)

        voice_map_path = os.getenv("VOICE_MAP_PATH", "/data/tts/voice_map.json")
        self.voice_map = VoiceMap(voice_map_path, [])

    def tts_list_voices(self) -> Dict[str, Any]:
        self._ensure_model()
        return {
            "voices": self.speakers,
            "model": self.config.model_name,
            "language": self.config.language,
        }

    def tts_synthesize(
        self,
        text: str,
        character_id: str,
        emotion: str,
        style: Optional[str],
        output_format: str = "wav",
    ) -> Dict[str, Any]:
        if output_format != "wav":
            raise ValueError("Only wav output_format is supported")
        if not text.strip():
            raise ValueError("text is required")

        self._ensure_model()

        speaker_wav = self._speaker_wav_for(character_id)
        speaker = self.voice_map.resolve(character_id)
        synthesis_text = self._style_text(text=text, emotion=emotion, style=style)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        kwargs: Dict[str, Any] = {
            "text": synthesis_text,
            "file_path": tmp_path,
            "language": self.config.language,
        }
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker:
            kwargs["speaker"] = speaker

        self.model.tts_to_file(**kwargs)

        audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        num_samples = int(audio.shape[0]) if hasattr(audio, "shape") else len(audio)
        duration_ms = int(round((num_samples / float(sr)) * 1000)) if sr else 0

        with open(tmp_path, "rb") as f:
            data = f.read()

        digest = hashlib.sha256(data).hexdigest()
        audio_path = os.path.join(self.audio_root, f"{digest}.wav")
        if not os.path.exists(audio_path):
            with open(audio_path, "wb") as f:
                f.write(data)

        self.store.put(data=data, content_type="audio/wav", tags=["tts", "xtts"])

        return {
            "wav_path": audio_path,
            "duration_ms": duration_ms,
            "speaker": speaker,
            "speaker_wav": speaker_wav,
        }

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        with self._model_lock:
            if self.model is not None:
                return
            tts_mod = _import_tts()
            model = tts_mod.TTS(model_name=self.config.model_name, progress_bar=False)
            if self.config.device == "cuda":
                model = model.to("cuda")
            self.model = model
            self.speakers = list(getattr(model, "speakers", []) or [])
            self.voice_map = VoiceMap(os.getenv("VOICE_MAP_PATH", "/data/tts/voice_map.json"), self.speakers)
            print(
                f"[xtts] device={self.config.device} model={self.config.model_name} speakers={len(self.speakers)}",
                flush=True,
            )

    def _speaker_wav_for(self, character_id: str) -> Optional[str]:
        root = os.getenv("XTTS_SPEAKER_WAV_DIR", "").strip()
        if not root:
            return None
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            candidate = os.path.join(root, f"{character_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    def _style_text(self, text: str, emotion: str, style: Optional[str]) -> str:
        # Keep payload compatible with existing inputs; XTTS does not directly expose
        # an emotion control API, so we pass plain text by default.
        _ = self._emotion_to_instruct(emotion, style)
        return text

    def _emotion_to_instruct(self, emotion: str, style: Optional[str]) -> str:
        base = (emotion or "neutral").strip()
        if not base:
            base = "neutral"
        base = base.capitalize()
        if style:
            return f"{base}. {style}"
        return base


_TTS_MODULE = None
_TTS_LOCK = threading.Lock()


def _import_tts():
    global _TTS_MODULE
    with _TTS_LOCK:
        if _TTS_MODULE is not None:
            return _TTS_MODULE
        from TTS.api import TTS as _TTS  # lazy import; heavy dependency

        class _Wrapper:
            TTS = _TTS

        _TTS_MODULE = _Wrapper
        return _TTS_MODULE


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Backward-compatible symbol used by imports/tests.
QwenTTSService = XTTSService
