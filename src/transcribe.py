"""
Offline speech-to-text using faster-whisper.

Downloads the chosen Whisper model (default: "small") from the Hugging
Face Hub once, then runs fully offline thereafter. This is a lean
addition, not a claimed replacement for a fine-tuned West-African-
language ASR model — Whisper's coverage of Nigerian Pidgin, Hausa,
Yoruba, and Twi is limited, so transcripts in those languages should be
treated as a rough draft the clinician can correct before extraction,
not as ground truth. French and English transcription are reliable.
"""
import hashlib
import threading
from typing import Optional, Tuple

import config

_whisper_model = None
_lock = threading.Lock()

_LANGUAGE_CODES = {
    "Auto-detect": None,
    "English": "en",
    "French": "fr",
    "Hausa": "ha",
    "Yoruba": "yo",
    # Nigerian Pidgin and Twi are not distinct Whisper language codes;
    # Pidgin transcribes reasonably as English, Twi is left to auto-detect.
    "Nigerian Pidgin": "en",
    "Twi": None,
}


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        print(f"[transcribe] Loading faster-whisper '{config.WHISPER_MODEL_SIZE}' model...")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("[transcribe] Whisper model ready.")
        return _whisper_model


def _hash_audio_file(audio_path: str) -> str:
    """sha256 of the raw audio bytes — lets us tell apart 'same audio
    submitted twice' from 'different audio, identical transcript' (Whisper
    hallucination/memorization) after the fact, since we don't otherwise
    keep the audio itself."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def transcribe_audio(audio_path: str, language_hint: str = "Auto-detect") -> Tuple[str, str, str]:
    """Returns (transcript_text, detected_language_label, audio_hash)."""
    if not audio_path:
        return "", "", ""
    model = _get_whisper_model()
    lang_code: Optional[str] = _LANGUAGE_CODES.get(language_hint)
    audio_hash = _hash_audio_file(audio_path)

    # vad_filter skips silence/dead air instead of running the (CPU-bound)
    # model over it — meaningful speedup on longer recordings with pauses.
    segments, info = model.transcribe(
        audio_path, language=lang_code, task="transcribe", vad_filter=True
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected = getattr(info, "language", lang_code or "unknown")
    return text, detected, audio_hash
