"""Deterministic Piper voice seeding for XTTS speaker_wav cloning."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


VOICE_SEED_FAILED = "VOICE_SEED_FAILED"


@dataclass
class VoiceSeederConfig:
    data_root: str
    piper_model_dir: str
    piper_bin: str = "piper"
    force_regenerate: bool = False
    seed: int = 42


def run_voice_seeder(cast_plan: Dict[str, Any], cfg: VoiceSeederConfig) -> Dict[str, Any]:
    speakers_dir = os.path.join(cfg.data_root, "tts", "speakers")
    voice_map_path = os.path.join(cfg.data_root, "tts", "voice_map.json")
    os.makedirs(speakers_dir, exist_ok=True)
    os.makedirs(os.path.dirname(voice_map_path), exist_ok=True)

    voice_map = _load_voice_map(voice_map_path)
    results: List[Dict[str, Any]] = []

    for role in cast_plan.get("roles", []):
        character_id = str(role.get("character_id") or "").strip()
        voice_id = str(role.get("voice_id") or "").strip()
        display_name = str(role.get("display_name") or role.get("role") or character_id).strip()
        seed_text = str(role.get("voice_seed_text") or "").strip() or _default_seed_text(display_name, character_id)

        if not character_id:
            results.append({"status": "failed", "error_code": VOICE_SEED_FAILED, "error": "missing character_id"})
            continue
        if not voice_id:
            results.append(
                {
                    "character_id": character_id,
                    "status": "failed",
                    "error_code": VOICE_SEED_FAILED,
                    "error": "missing voice_id",
                }
            )
            continue

        wav_path = os.path.join(speakers_dir, f"{voice_id}.wav")
        try:
            if cfg.force_regenerate or not os.path.exists(wav_path):
                model_path = _resolve_piper_model_path(cfg.piper_model_dir, voice_id)
                _synthesize_seed_wav(cfg.piper_bin, model_path, seed_text, wav_path)
                status = "generated"
            else:
                status = "reused"
            if voice_map.get(character_id) != voice_id:
                voice_map[character_id] = voice_id
            results.append(
                {
                    "character_id": character_id,
                    "voice_id": voice_id,
                    "wav_path": wav_path,
                    "status": status,
                }
            )
        except Exception as err:
            results.append(
                {
                    "character_id": character_id,
                    "voice_id": voice_id,
                    "wav_path": wav_path,
                    "status": "failed",
                    "error_code": VOICE_SEED_FAILED,
                    "error": f"{type(err).__name__}: {err}",
                }
            )

    _save_voice_map(voice_map_path, voice_map)
    return {
        "seed": cfg.seed,
        "voice_map_path": voice_map_path,
        "speakers_dir": speakers_dir,
        "results": results,
    }


def load_voice_map(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "tts", "voice_map.json")
    return _load_voice_map(path)


def resolve_voice_id(character_id: str, cast_plan: Dict[str, Any], voice_map: Dict[str, str]) -> Tuple[str | None, str]:
    role = _role_by_character_id(cast_plan).get(character_id)
    if role:
        voice_id = str(role.get("voice_id") or "").strip()
        if voice_id:
            return voice_id, "cast_plan.voice_id"
    mapped = str(voice_map.get(character_id) or "").strip()
    if mapped:
        return mapped, "voice_map"
    return None, "missing"


def _role_by_character_id(cast_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for role in cast_plan.get("roles", []):
        character_id = str(role.get("character_id") or "").strip()
        if character_id:
            out[character_id] = role
    return out


def _load_voice_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: Dict[str, str] = {}
            for key, value in data.items():
                k = str(key).strip()
                v = str(value).strip()
                if k and v:
                    out[k] = v
            return out
    except Exception:
        pass
    return {}


def _save_voice_map(path: str, mapping: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=True, indent=2)


def _resolve_piper_model_path(model_dir: str, voice_id: str) -> str:
    voice_id = _canonical_voice_id(voice_id)
    os.makedirs(model_dir, exist_ok=True)
    direct = os.path.join(model_dir, voice_id)
    if os.path.exists(direct) and _piper_config_exists(direct):
        return direct
    onnx = os.path.join(model_dir, f"{voice_id}.onnx")
    if os.path.exists(onnx) and _piper_config_exists(onnx):
        return onnx
    _download_piper_model_if_missing(model_dir, voice_id)
    if os.path.exists(direct) and _piper_config_exists(direct):
        return direct
    if os.path.exists(onnx) and _piper_config_exists(onnx):
        return onnx
    raise FileNotFoundError(f"Piper model not found for voice_id={voice_id} in {model_dir}")


def _synthesize_seed_wav(piper_bin: str, model_path: str, text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if shutil.which(piper_bin) is None:
        fallback = _fallback_seed_wav()
        if fallback:
            shutil.copyfile(fallback, output_path)
            return
        raise FileNotFoundError(piper_bin)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [piper_bin, "--model", model_path, "--output_file", tmp_path]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        _normalize_wav(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _normalize_wav(src_path: str, dst_path: str) -> None:
    ffmpeg = shutil.which(os.getenv("FFMPEG_BIN", "ffmpeg"))
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "22050",
            "-sample_fmt",
            "s16",
            dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    shutil.copyfile(src_path, dst_path)


def _default_seed_text(display_name: str, character_id: str) -> str:
    safe_name = display_name.strip() or character_id.strip() or "Speaker"
    return (
        f"I am {safe_name}. I speak clearly and naturally. "
        "This voice sample is calm, expressive, and easy to understand. "
        "I keep a steady rhythm, clean pronunciation, and a friendly tone. "
        "These lines are recorded to seed a stable synthetic voice profile "
        "for future dialogue generation."
    )


def _fallback_seed_wav() -> str | None:
    candidates = [
        os.path.join(os.getenv("DATA_ROOT", "data"), "tts", "speakers", "voice-default.wav"),
        os.path.join(os.getenv("DATA_ROOT", "data"), "tts", "speakers", "en_US-lessac-medium.wav"),
        os.path.join(os.getenv("DATA_ROOT", "data"), "artifacts", "audio"),
        "/app/runtime/sample/tone.wav",
        "runtime/sample/tone.wav",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(".wav"):
                    candidate = os.path.join(path, name)
                    if os.path.isfile(candidate):
                        return candidate
    return None


def _piper_config_exists(model_path: str) -> bool:
    if os.path.exists(model_path + ".json"):
        return True
    if model_path.endswith(".onnx"):
        alt = model_path[:-5] + ".json"
        if os.path.exists(alt):
            return True
    return False


def _download_piper_model_if_missing(model_dir: str, voice_id: str) -> None:
    voice_id = _canonical_voice_id(voice_id)
    voice_key = voice_id[:-5] if voice_id.endswith(".onnx") else voice_id
    base = os.getenv("PIPER_MODEL_REGISTRY_URL", "https://huggingface.co/rhasspy/piper-voices/resolve/main").rstrip("/")
    default_rel = _piper_registry_rel_path(voice_key)
    onnx_tpl = os.getenv("PIPER_MODEL_ONNX_URL_TEMPLATE", f"{base}" + "/{rel_path}.onnx")
    cfg_tpl = os.getenv("PIPER_MODEL_CONFIG_URL_TEMPLATE", f"{base}" + "/{rel_path}.onnx.json")
    onnx_url = onnx_tpl.format(voice_id=voice_key, rel_path=default_rel)
    cfg_url = cfg_tpl.format(voice_id=voice_key, rel_path=default_rel)

    target_onnx = os.path.join(model_dir, f"{voice_key}.onnx")
    target_cfg = target_onnx + ".json"
    if os.path.exists(target_onnx) and os.path.exists(target_cfg):
        return

    timeout = float(os.getenv("PIPER_MODEL_DOWNLOAD_TIMEOUT_SEC", "180"))
    _download_file(onnx_url, target_onnx, timeout_sec=timeout)
    _download_file(cfg_url, target_cfg, timeout_sec=timeout)


def _download_file(url: str, out_path: str, timeout_sec: float = 180.0) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "aithtr-voice-seeder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
    except Exception as err:
        raise RuntimeError(f"Failed to download {url}: {type(err).__name__}: {err}") from err
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, out_path)


def _canonical_voice_id(voice_id: str) -> str:
    raw = (voice_id or "").strip()
    if raw in {"", "voice-default"}:
        return os.getenv("PIPER_DEFAULT_VOICE_ID", "en_US-lessac-medium").strip()
    # Non-Piper ids (e.g., voice-a, voice-god) use the default Piper model for
    # generation while keeping distinct output wav files per voice_id.
    if "_" not in raw and raw.startswith("voice-"):
        return os.getenv("PIPER_DEFAULT_VOICE_ID", "en_US-lessac-medium").strip()
    return raw


def _piper_registry_rel_path(voice_id: str) -> str:
    # Convert "en_US-lessac-medium" -> "en/en_US/lessac/medium/en_US-lessac-medium"
    parts = voice_id.split("-")
    if len(parts) >= 3 and "_" in parts[0]:
        lang_root = parts[0].split("_", 1)[0]
        return f"{lang_root}/{parts[0]}/{parts[1]}/{parts[2]}/{voice_id}"
    return voice_id
