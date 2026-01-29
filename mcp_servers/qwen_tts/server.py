"""
Qwen3-TTS MCP server implementation.
"""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from mcp_servers.assets.artifact_store import ArtifactStore


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_AUDIO_SR = 48000

EMOTION_INSTRUCT = {
    "sad": "Sad, quiet, low energy",
    "angry": "Angry, controlled, sharp",
    "happy": "Warm, upbeat, energetic",
}


@dataclass
class QwenConfig:
    model_name: str
    dtype: torch.dtype
    device: str
    use_flash_attn: bool


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

    def resolve(self, character_id: str) -> str:
        mapping = self.load()
        if character_id in mapping:
            return mapping[character_id]
        used = set(mapping.values())
        for speaker in self.speakers:
            if speaker not in used:
                mapping[character_id] = speaker
                self.save(mapping)
                return speaker
        speaker = self.speakers[0] if self.speakers else "default"
        mapping[character_id] = speaker
        self.save(mapping)
        return speaker


class QwenTTSService:
    def __init__(self) -> None:
        self.config = self._load_config()
        self.model = None
        self.speakers = []
        voice_map_path = os.getenv("VOICE_MAP_PATH", "/data/tts/voice_map.json")
        self.voice_map = VoiceMap(voice_map_path, [])
        self.store = ArtifactStore(root=os.getenv("ARTIFACT_ROOT", "/data/artifacts"))
        self.audio_root = os.getenv("ARTIFACT_AUDIO_ROOT", "/data/artifacts/audio")
        os.makedirs(self.audio_root, exist_ok=True)

    def tts_list_voices(self) -> Dict[str, Any]:
        self._ensure_model()
        return {"voices": self.speakers}

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

        self._ensure_model()
        speaker = self.voice_map.resolve(character_id)
        instruct = self._emotion_to_instruct(emotion, style)

        waveform, sr = self._synthesize_audio(text=text, speaker=speaker, instruct=instruct)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr, DEFAULT_AUDIO_SR)
        waveform = waveform.astype(np.float32)

        duration_ms = int(round((waveform.shape[0] / DEFAULT_AUDIO_SR) * 1000))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, waveform, DEFAULT_AUDIO_SR, subtype="FLOAT")

        with open(tmp_path, "rb") as f:
            data = f.read()

        digest = hashlib.sha256(data).hexdigest()
        audio_path = os.path.join(self.audio_root, f"{digest}.wav")
        if not os.path.exists(audio_path):
            with open(audio_path, "wb") as f:
                f.write(data)

        self.store.put(data=data, content_type="audio/wav", tags=["tts", "qwen"])

        return {"wav_path": audio_path, "duration_ms": duration_ms}

    def _emotion_to_instruct(self, emotion: str, style: Optional[str]) -> str:
        base = EMOTION_INSTRUCT.get(emotion.lower(), emotion)
        if style:
            return f"{base}. {style}"
        return base

    def _load_config(self) -> QwenConfig:
        model_name = os.getenv("QWEN_TTS_MODEL", DEFAULT_MODEL)
        device = os.getenv("QWEN_TTS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_env = os.getenv("QWEN_TTS_DTYPE", "")
        if dtype_env.lower() == "float16":
            dtype = torch.float16
        elif dtype_env.lower() == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_env.lower() == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        use_flash = os.getenv("QWEN_TTS_FLASH_ATTN", "1") == "1"
        return QwenConfig(model_name=model_name, dtype=dtype, device=device, use_flash_attn=use_flash)

    def _load_model(self) -> Any:
        qwen_tts = _import_qwen_tts()

        if hasattr(qwen_tts, "Qwen3TTSModel"):
            model_cls = qwen_tts.Qwen3TTSModel
            model = model_cls.from_pretrained(self.config.model_name)
        else:
            raise ImportError(f"Unsupported qwen_tts API. Available: {dir(qwen_tts)}")

        if hasattr(model, "to"):
            model.to(self.config.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = self._load_model()
            self.speakers = self._load_speakers()
            self.voice_map = VoiceMap(os.getenv("VOICE_MAP_PATH", "/data/tts/voice_map.json"), self.speakers)

    def _load_speakers(self) -> List[str]:
        env = os.getenv("QWEN_TTS_SPEAKERS", "")
        if env.strip():
            return [s.strip() for s in env.split(",") if s.strip()]
        if hasattr(self.model, "speakers"):
            return list(self.model.speakers)
        if hasattr(self.model, "get_speakers"):
            return list(self.model.get_speakers())
        return ["speaker_0"]

    def _synthesize_audio(self, text: str, speaker: str, instruct: str) -> Tuple[np.ndarray, int]:
        if hasattr(self.model, "synthesize"):
            audio = self.model.synthesize(text=text, speaker=speaker, instruct=instruct)
        elif hasattr(self.model, "infer"):
            audio = self.model.infer(text=text, speaker=speaker, instruct=instruct)
        elif callable(self.model):
            audio = self.model(text=text, speaker=speaker, instruct=instruct)
        else:
            raise RuntimeError("QwenTTS model does not support synthesis")

        if isinstance(audio, tuple) and len(audio) == 2:
            waveform, sr = audio
        elif isinstance(audio, dict):
            waveform = audio.get("waveform")
            sr = audio.get("sample_rate")
        else:
            raise RuntimeError("Unexpected QwenTTS output")

        waveform = self._to_numpy(waveform)
        if sr is None:
            raise RuntimeError("Sample rate missing from QwenTTS output")
        return waveform, int(sr)

    def _to_numpy(self, waveform: Any) -> np.ndarray:
        if isinstance(waveform, np.ndarray):
            return waveform
        if torch.is_tensor(waveform):
            return waveform.detach().cpu().numpy()
        return np.asarray(waveform)

    def _to_mono(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 1:
            return waveform
        return waveform.mean(axis=-1)

    def _resample(self, waveform: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return waveform
        import scipy.signal

        gcd = np.gcd(sr_in, sr_out)
        up = sr_out // gcd
        down = sr_in // gcd
        return scipy.signal.resample_poly(waveform, up, down).astype(np.float32)


def _import_qwen_tts():
    try:
        import qwen_tts
        return qwen_tts
    except Exception as exc:
        raise ImportError(f"Failed to import qwen_tts: {exc}")
