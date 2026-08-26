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


def transcribe_audio(audio_path: str, language_hint: str = "Auto-detect") -> Tuple[str, str]:
    """Returns (transcript_text, detected_language_label)."""
    if not audio_path:
        return "", ""
    model = _get_whisper_model()
    lang_code: Optional[str] = _LANGUAGE_CODES.get(language_hint)

    segments, info = model.transcribe(audio_path, language=lang_code, task="transcribe")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected = getattr(info, "language", lang_code or "unknown")
    return text, detected
