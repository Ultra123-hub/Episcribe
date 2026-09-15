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
languages when connectivity allows.

Uses Sahara's asynchronous file-upload endpoint (upload, then poll
/file/v1/status/{file_id} until FILE_TRANSCRIBED) rather than its
synchronous one, which hard-caps audio at 120s/request -- confirmed via
task-1 verification (docs/sahara_stt_verification.md) that most real
consultation-length recordings exceed that.

IMPORTANT, confirmed directly against the live API (found the hard way
-- see git history for the earlier version of this module, which had
this wrong): use_language_asr_input is REQUIRED by BOTH of Sahara's
file-upload endpoints, sync and async alike. Omitting it, or sending an
empty/"auto"-style placeholder value, is rejected outright on both --
"use_language_asr_input is required" / "... is not supported". There is
NO auto-detect mode on either endpoint despite an earlier version of
this file claiming testing had shown otherwise (that claim was never
actually verified against Sahara's Auto-detect path specifically, only
against explicit-language requests). So: "Auto-detect" in this app's own
language dropdown cannot be sent to Sahara at all -- see
_transcribe_via_sahara's upfront check, which raises a clear error
asking the clinician to pick a specific language instead of surfacing
Sahara's raw rejection.

Twi is Sahara-supported for plain transcription but is NOT one of its
code-switched language pairs (unlike Pidgin, Hausa, and Yoruba) -- so
Sahara offers less of an advantage there than for the other three. (Sahara
does separately support "Akan-English" as a code-switched pair, code
"ak" -- distinct from plain Twi/"tw" -- not currently exposed in this
app's language dropdown.)
"""
import hashlib
import threading
import time
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

# Reverse of the above, for display purposes only. Sahara's response
# doesn't include a detected-language field (confirmed against its actual
# response payload), so when a language was explicitly selected we echo
# back that human-readable label rather than the raw code Sahara expects
# ("pcm" is meaningless to a clinician reading the result); when nothing
# was specified (Auto-detect), we're honest that we don't actually know
# what Sahara detected, since it doesn't tell us.
_SAHARA_CODE_TO_LABEL = {v: k for k, v in _SAHARA_LANGUAGE_CODES.items() if v}


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


_SAHARA_UPLOAD_ENDPOINT = "https://infer.voice.intron.io/file/v1/upload"
_SAHARA_STATUS_ENDPOINT = "https://infer.voice.intron.io/file/v1/status/{file_id}"
_SAHARA_POLL_INTERVAL_S = 5
_SAHARA_POLL_TIMEOUT_S = 900  # 15 min ceiling; a real consultation shouldn't approach this


def _transcribe_via_sahara(audio_path: str, language_hint: str) -> Tuple[str, str]:
    """Returns (transcript_text, detected_language_label).

    Uses Sahara's asynchronous file-upload endpoint (upload, then poll
    /file/v1/status/{file_id} until FILE_TRANSCRIBED). use_language_asr_input
    is REQUIRED -- see module docstring for why "Auto-detect" can't be sent
    to Sahara at all, and raises a clear, actionable error here instead of
    Sahara's raw rejection.

    Raises RuntimeError with a clear message on missing API key/language,
    HTTP errors, a FILE_PROCESSING_FAILED status, or a poll timeout.
    """
    if not config.SAHARA_API_KEY:
        raise RuntimeError(
            "SAHARA_API_KEY is not set. Add it to your .env, or set "
            "EPISCRIBE_STT_BACKEND=whisper to use the offline backend instead."
        )

    lang_code = _SAHARA_LANGUAGE_CODES.get(language_hint)
    if not lang_code:
        raise RuntimeError(
            f"Sahara needs a specific language to transcribe with -- "
            f"{language_hint!r} has no Sahara language code (Sahara has no "
            f"auto-detect mode on either of its file-upload endpoints, "
            f"confirmed directly against the API). Please pick a specific "
            f"language from the Language hint dropdown and try again, or "
            f"switch to EPISCRIBE_STT_BACKEND=whisper."
        )

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed. Run: pip install requests"
        ) from exc

    headers = {"Authorization": f"Bearer {config.SAHARA_API_KEY}"}
    request_data = {
        "audio_file_name": "episcribe_recording",
        # Telehealth category matches this app's domain and keeps us on
        # Sahara's clinical-tuned post-processing.
        "use_category": "file_category_telehealth",
        "use_language_asr_input": lang_code,
    }

    with open(audio_path, "rb") as f:
        try:
            response = requests.post(
                _SAHARA_UPLOAD_ENDPOINT,
                headers=headers,
                data=request_data,
                files={"audio_file_blob": f},
                timeout=60,  # just the upload itself, not processing
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Sahara upload failed: {exc}") from exc

    if response.status_code == 400:
        raise RuntimeError(
            f"Sahara rejected the upload (likely unsupported format): "
            f"{response.text[:300]}"
        )
    if not response.ok:
        raise RuntimeError(f"Sahara upload returned HTTP {response.status_code}: {response.text[:300]}")

    file_id = (response.json().get("data") or {}).get("file_id")
    if not file_id:
        raise RuntimeError(f"Sahara upload response had no file_id: {response.text[:300]}")

    detected = _SAHARA_CODE_TO_LABEL.get(lang_code, lang_code)
    status_url = _SAHARA_STATUS_ENDPOINT.format(file_id=file_id)
    deadline = time.monotonic() + _SAHARA_POLL_TIMEOUT_S
    while True:
        try:
            status_resp = requests.get(status_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"Sahara status check failed: {exc}") from exc
        if not status_resp.ok:
            raise RuntimeError(
                f"Sahara status check returned HTTP {status_resp.status_code}: "
                f"{status_resp.text[:300]}"
            )
        status_data = status_resp.json().get("data") or {}
        state = status_data.get("processing_status")
        if state == "FILE_TRANSCRIBED":
            return (status_data.get("audio_transcript") or "").strip(), detected
        if state == "FILE_PROCESSING_FAILED":
            raise RuntimeError(f"Sahara failed to process this file: {status_data}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Sahara did not finish transcribing within "
                f"{_SAHARA_POLL_TIMEOUT_S}s (last status: {state})."
            )
        time.sleep(_SAHARA_POLL_INTERVAL_S)


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
