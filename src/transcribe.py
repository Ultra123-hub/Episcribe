"""
Offline speech-to-text using faster-whisper, with an optional Sahara
(Intron Voice API) backend for higher-accuracy African-language and
code-switched transcription.

Backend selection is controlled by config.STT_BACKEND ("whisper" or
"sahara"). Both backends implement the same interface:
    transcribe_audio(audio_path, language_hint) -> (text, detected_lang, audio_hash)

faster-whisper remains the default: it's fully offline once its model is
cached, matching EpiScribe's offline-first design. Whisper's coverage of
Nigerian Pidgin, Hausa, Yoruba, and Twi is limited, so transcripts in
those languages should be treated as a rough draft the clinician can
correct before extraction, not as ground truth. French and English
transcription are reliable.

Sahara requires an internet connection and an API key
(SAHARA_API_KEY), but is purpose-built for African-accented and
code-switched speech (e.g. Hausa-English, Yoruba-English,
Pidgin-English) and should be materially more accurate on those
languages when connectivity allows. Its sync upload endpoint caps audio
at 120 seconds per request; longer recordings need a different (async)
endpoint we haven't wired up yet.

Twi is Sahara-supported for plain transcription but is NOT one of its
code-switched language pairs (unlike Pidgin, Hausa, and Yoruba) — so
Sahara offers less of an advantage there than for the other three.
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

# Sahara's own language codes (see docs.voice.intron.io/docs/stt/supported-languages).
# "Auto-detect" maps to None deliberately: earlier we forced "en" here,
# which told Sahara every "Auto-detect" recording WAS English, suppressing
# proper Pidgin/Hausa/Yoruba code-switch recognition even when the speaker
# wasn't speaking English at all. Manual testing confirmed Sahara handles
# multilingual/code-switched audio noticeably better when the language
# field is left out of the request entirely, rather than guessed at.
_SAHARA_LANGUAGE_CODES = {
    "Auto-detect": None,
    "English": "en",
    "French": "fr",
    "Hausa": "ha",
    "Yoruba": "yo",
    "Nigerian Pidgin": "pcm",
    "Twi": "tw",  # supported, but not one of Sahara's code-switched pairs
}

# Reverse of the above, for display purposes only. Sahara's sync response
# doesn't include a detected-language field (confirmed against its actual
# response payload), so when a language was explicitly selected we echo
# back that human-readable label rather than the raw code Sahara expects
# ("pcm" is meaningless to a clinician reading the result); when nothing
# was specified (Auto-detect), we're honest that we don't actually know
# what Sahara detected, since it doesn't tell us.
_SAHARA_CODE_TO_LABEL = {v: k for k, v in _SAHARA_LANGUAGE_CODES.items() if v}

_SAHARA_ENDPOINT = "https://infer.voice.intron.io/file/v1/upload/sync"


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


def _transcribe_via_whisper(audio_path: str, language_hint: str) -> Tuple[str, str]:
    """Returns (transcript_text, detected_language_label)."""
    model = _get_whisper_model()
    lang_code: Optional[str] = _LANGUAGE_CODES.get(language_hint)

    # vad_filter skips silence/dead air instead of running the (CPU-bound)
    # model over it — meaningful speedup on longer recordings with pauses.
    segments, info = model.transcribe(
        audio_path, language=lang_code, task="transcribe", vad_filter=True
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected = getattr(info, "language", lang_code or "unknown")
    return text, detected


def _transcribe_via_sahara(audio_path: str, language_hint: str) -> Tuple[str, str]:
    """Returns (transcript_text, detected_language_label).

    Uses Sahara's synchronous file-upload endpoint. Raises RuntimeError
    with a clear message on missing API key, HTTP errors, or the 503
    processing-timeout case documented by Intron (which can occur even
    on a successful upload if the file is still processing after 120s).
    """
    if not config.SAHARA_API_KEY:
        raise RuntimeError(
            "SAHARA_API_KEY is not set. Add it to your .env, or set "
            "EPISCRIBE_STT_BACKEND=whisper to use the offline backend instead."
        )

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed. Run: pip install requests"
        ) from exc

    lang_code = _SAHARA_LANGUAGE_CODES.get(language_hint)

    request_data = {
        "audio_file_name": "episcribe_recording",
        # Telehealth category matches this app's domain and keeps us on
        # Sahara's clinical-tuned post-processing.
        "use_category": "file_category_telehealth",
    }
    # Only send a language hint when one is actually known — omitting the
    # field lets Sahara handle detection/code-switching itself, which
    # tested noticeably better than forcing "en" on every Auto-detect
    # request (see comment on _SAHARA_LANGUAGE_CODES above).
    if lang_code:
        request_data["use_language_asr_input"] = lang_code

    with open(audio_path, "rb") as f:
        try:
            response = requests.post(
                _SAHARA_ENDPOINT,
                headers={"Authorization": f"Bearer {config.SAHARA_API_KEY}"},
                data=request_data,
                files={"audio_file_blob": f},
                timeout=130,  # endpoint itself can take up to 120s
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Sahara request failed: {exc}") from exc

    if response.status_code == 503:
        raise RuntimeError(
            "Sahara timed out processing this file within 120 seconds. "
            "Try a shorter recording, or switch to the whisper backend."
        )
    if response.status_code == 400:
        raise RuntimeError(
            f"Sahara rejected the request (likely audio too long or "
            f"unsupported format): {response.text[:300]}"
        )
    if not response.ok:
        raise RuntimeError(f"Sahara returned HTTP {response.status_code}: {response.text[:300]}")

    payload = response.json()
    data = payload.get("data", {})
    text = (data.get("audio_transcript") or "").strip()
    # Sahara's sync response doesn't include a detected-language field, so
    # this reflects what we told it (as a readable label), or is honest
    # that we don't know if nothing was specified (Auto-detect).
    detected = _SAHARA_CODE_TO_LABEL.get(lang_code, "auto (Sahara)")
    return text, detected


def transcribe_audio(audio_path: str, language_hint: str = "Auto-detect") -> Tuple[str, str, str]:
    """Returns (transcript_text, detected_language_label, audio_hash).

    Backend is chosen via config.STT_BACKEND ("whisper" default, or
    "sahara"). Both backends share the audio hashing step below so
    provenance tracking (see schema.EncounterRecord.audio_hash) works
    identically regardless of which one transcribed the file.
    """
    if not audio_path:
        return "", "", ""

    audio_hash = _hash_audio_file(audio_path)

    if config.STT_BACKEND == "sahara":
        text, detected = _transcribe_via_sahara(audio_path, language_hint)
    else:
        text, detected = _transcribe_via_whisper(audio_path, language_hint)

    return text, detected, audio_hash
