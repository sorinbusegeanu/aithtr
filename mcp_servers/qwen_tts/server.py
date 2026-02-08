"""Coqui XTTSv2 MCP service implementation."""
import hashlib
import json
import os
import tempfile
import threading
import wave
import math
import struct
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
        self._fallback_reason: Optional[str] = None

        artifact_root = os.getenv("ARTIFACT_ROOT", "/data/artifacts")
        self.store = ArtifactStore(root=artifact_root)
        self.audio_root = os.getenv("ARTIFACT_AUDIO_ROOT", "/data/artifacts/audio")
        os.makedirs(self.audio_root, exist_ok=True)

    def tts_list_voices(self) -> Dict[str, Any]:
        self._ensure_model()
        return {
            "voices": self.speakers,
            "model": self.config.model_name,
            "language": self.config.language,
            "fallback": bool(self._fallback_reason),
        }

    def tts_synthesize(
        self,
        text: str,
        character_id: str,
        voice_id: Optional[str],
        emotion: str,
        style: Optional[str],
        output_format: str = "wav",
    ) -> Dict[str, Any]:
        if output_format != "wav":
            raise ValueError("Only wav output_format is supported")
        if not text.strip():
            raise ValueError("text is required")

        self._ensure_model()

        if self._fallback_reason:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            self._synthesize_fallback_tone(text=text, out_path=tmp_path, character_id=character_id)
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
            self.store.put(data=data, content_type="audio/wav", tags=["tts", "xtts", "fallback"])
            return {
                "wav_path": audio_path,
                "duration_ms": duration_ms,
                "speaker": None,
                "speaker_wav": None,
                "fallback": True,
                "fallback_reason": self._fallback_reason,
            }

        speaker_wav = self._speaker_wav_for(character_id, voice_id)
        speaker = None
        if not speaker_wav and self.speakers:
            preferred = os.getenv("XTTS_DEFAULT_SPEAKER", "").strip()
            if preferred and preferred in self.speakers:
                speaker = preferred
            else:
                speaker = self.speakers[0]
        if not speaker_wav and not speaker:
            raise ValueError(
                f"missing voice_id for character_id={character_id}; expected speaker_wav at XTTS_SPEAKER_WAV_DIR/<voice_id>.wav"
            )
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

        try:
            self.model.tts_to_file(**kwargs)
        except RuntimeError as err:
            msg = str(err)
            if self.config.device == "cuda" and ("CUDA error" in msg or "device-side assert" in msg):
                # Recover from poisoned CUDA context by reloading XTTS on CPU.
                self._reload_model(device="cpu")
                self.model.tts_to_file(**kwargs)
            else:
                raise

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
        if self.model is not None or self._fallback_reason is not None:
            return
        with self._model_lock:
            if self.model is not None or self._fallback_reason is not None:
                return
            try:
                tts_mod = _import_tts()
                model = tts_mod.TTS(model_name=self.config.model_name, progress_bar=False)
                if self.config.device == "cuda":
                    model = model.to("cuda")
                self.model = model
                self.speakers = _discover_speakers(model)
                print(
                    f"[xtts] device={self.config.device} model={self.config.model_name} speakers={len(self.speakers)}",
                    flush=True,
                )
            except Exception as exc:
                self._fallback_reason = str(exc)
                self.model = None
                self.speakers = []
                print(f"[xtts] fallback enabled: {self._fallback_reason}", flush=True)

    def _reload_model(self, device: str) -> None:
        with self._model_lock:
            tts_mod = _import_tts()
            model = tts_mod.TTS(model_name=self.config.model_name, progress_bar=False)
            self.config.device = device
            if device == "cuda":
                model = model.to("cuda")
            self.model = model
            self.speakers = _discover_speakers(model)
            print(
                f"[xtts] reloaded device={self.config.device} model={self.config.model_name} speakers={len(self.speakers)}",
                flush=True,
            )

    def _speaker_wav_for(self, character_id: str, voice_id: Optional[str]) -> Optional[str]:
        root = os.getenv("XTTS_SPEAKER_WAV_DIR", "").strip()
        if not root:
            return None
        if voice_id:
            for ext in ("", ".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                candidate = os.path.join(root, f"{voice_id}{ext}")
                if os.path.exists(candidate):
                    return candidate
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

    def _synthesize_fallback_tone(self, text: str, out_path: str, character_id: str) -> None:
        sample_rate = 22050
        duration_sec = max(1.0, min(10.0, len(text) / 14.0))
        frames = int(sample_rate * duration_sec)
        base = 170 + (abs(hash(character_id)) % 70)
        amp = 0.2 * 32767.0
        with wave.open(out_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(frames):
                t = i / float(sample_rate)
                x = math.sin(2.0 * math.pi * base * t) + 0.5 * math.sin(2.0 * math.pi * (base * 2.0) * t)
                sample = int(max(-32767, min(32767, amp * x)))
                wav.writeframes(struct.pack("<h", sample))


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


def _discover_speakers(model: Any) -> List[str]:
    candidates: List[Any] = []
    candidates.append(getattr(model, "speakers", None))

    speaker_manager = getattr(model, "speaker_manager", None)
    if speaker_manager is not None:
        candidates.append(getattr(speaker_manager, "speaker_names", None))
        candidates.append(getattr(speaker_manager, "name_to_id", None))

    synthesizer = getattr(model, "synthesizer", None)
    if synthesizer is not None:
        tts_model = getattr(synthesizer, "tts_model", None)
        if tts_model is not None:
            manager = getattr(tts_model, "speaker_manager", None)
            if manager is not None:
                candidates.append(getattr(manager, "speaker_names", None))
                candidates.append(getattr(manager, "name_to_id", None))

    out: List[str] = []
    for value in candidates:
        names: List[str] = []
        if isinstance(value, dict):
            names = [str(k) for k in value.keys() if str(k).strip()]
        elif isinstance(value, (list, tuple, set)):
            names = [str(v) for v in value if str(v).strip()]
        for name in names:
            if name not in out:
                out.append(name)
    return out


# Backward-compatible symbol used by imports/tests.
QwenTTSService = XTTSService


class VoiceMap:
    """Backward-compatible deterministic voice mapper used by tests."""

    def __init__(self, path: str, available_voices: List[str]) -> None:
        self.path = path
        self.available_voices = [str(v) for v in available_voices if str(v).strip()]
        self._mapping: Dict[str, str] = self._load()

    def resolve(self, character_id: str) -> str:
        key = str(character_id or "").strip()
        if not key:
            raise ValueError("character_id is required")
        existing = self._mapping.get(key)
        if existing:
            return existing
        if not self.available_voices:
            raise ValueError("available_voices is empty")
        assigned = self.available_voices[len(self._mapping) % len(self.available_voices)]
        self._mapping[key] = assigned
        self._save()
        return assigned

    def _load(self) -> Dict[str, str]:
        if not self.path or not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            out: Dict[str, str] = {}
            allowed = set(self.available_voices)
            for k, v in raw.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs and (not allowed or vs in allowed):
                    out[ks] = vs
            return out
        except Exception:
            return {}

    def _save(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._mapping, f, ensure_ascii=True, indent=2)
