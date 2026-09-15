"""
Sahara 2.5 streaming TTS (Intron Voice API) for spoken assistant replies.

Protocol: wss://infer.voice.intron.io/tts/v1/stream (see
docs.voice.intron.io/docs/tts/tts-streaming). Hard limits enforced by the
API itself: 300s max session lifetime, 60s max idle between chunks, and
each INPUT_TEXT_CHUNK must be 10-100 characters -- this module splits
longer replies into <=100-char chunks on word boundaries before sending,
and merges a too-short trailing remainder into the previous chunk so it
doesn't get rejected with CHUNK_SIZE_TOO_SMALL.

Uses the `websockets` library (already an indirect dependency via gradio),
run synchronously via asyncio.run() to match the rest of this codebase's
synchronous style (transcribe.py, extraction_agent.py, etc. are all plain
sync functions) -- callers don't need to know this is async underneath.
"""
import asyncio
import base64
import io
import json
import wave
from typing import List, Optional

import config

_WS_URL = "wss://infer.voice.intron.io/tts/v1/stream"
_MAX_CHUNK_CHARS = 100
_MIN_CHUNK_CHARS = 10
_SESSION_TIMEOUT = 300  # seconds, matches Sahara's hard session cap
_RECV_TIMEOUT = 20  # per-message wait, well under the 60s idle cap


def _chunk_text(text: str) -> List[str]:
    """Splits text into <=100-char pieces on word boundaries, merging any
    trailing piece under 10 chars into the previous one."""
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > _MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < _MIN_CHUNK_CHARS:
        merged = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()
        if len(merged) <= _MAX_CHUNK_CHARS:
            chunks[-1] = merged
        else:
            chunks.append(merged)  # rare: can't merge cleanly, send as-is
    return chunks


def _concat_wavs(parts: List[bytes]) -> bytes:
    """Each chunk Sahara returns is its OWN complete WAV file (own RIFF
    header). Naively concatenating the raw bytes of multiple chunks would
    produce a file whose header only describes the first chunk's length --
    a player would cut off after chunk 1 and the rest would be garbage
    trailing bytes. Decode each chunk's PCM frames and re-encode them as
    one WAV with a single correct header instead."""
    if len(parts) == 1:
        return parts[0]
    out_buf = io.BytesIO()
    out_wav: Optional[wave.Wave_write] = None
    try:
        for part in parts:
            with wave.open(io.BytesIO(part), "rb") as in_wav:
                if out_wav is None:
                    out_wav = wave.open(out_buf, "wb")
                    out_wav.setnchannels(in_wav.getnchannels())
                    out_wav.setsampwidth(in_wav.getsampwidth())
                    out_wav.setframerate(in_wav.getframerate())
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
    finally:
        if out_wav is not None:
            out_wav.close()
    return out_buf.getvalue()


async def _synthesize_async(
    text: str, voice_language: str, voice_accent: str, voice_gender: str
) -> bytes:
    import websockets

    if not config.SAHARA_API_KEY:
        raise RuntimeError(
            "SAHARA_API_KEY is not set. Add it to your .env to enable spoken replies."
        )

    chunks = _chunk_text(text)
    if not chunks:
        return b""

    url = (
        f"{_WS_URL}?voice_language={voice_language}"
        f"&voice_accent={voice_accent}&voice_gender={voice_gender}"
    )
    headers = {"Authorization": f"Bearer {config.SAHARA_API_KEY}"}

    audio_parts: List[bytes] = []
    # `extra_headers` is the param name on the installed websockets 12.x
    # (legacy client) API; websockets >=13 renamed it to additional_headers.
    #
    # NOTE on message shapes below: the outbound field is "message_type"
    # (not "type"), chunk_id is 1-based and must be an int (both
    # INPUT_TEXT_CHUNK's "ack_id" and FETCH_AUDIO_CHUNK's "chunk_id"), and
    # the ready/processing field in the server's own response is spelled
    # "processing_staus" (sic -- a typo in the live API, confirmed against
    # the real endpoint, not a typo introduced here). None of this matches
    # the public docs summary closely enough to trust the docs alone --
    # verified directly against docs.voice.intron.io/docs/tts/tts-streaming
    # via the real API before writing this.
    async with websockets.connect(url, extra_headers=headers, open_timeout=15) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)
        session_msg = json.loads(raw)
        if session_msg.get("message_type") != "SESSION_CREATED":
            raise RuntimeError(f"Unexpected session-open response: {session_msg}")

        for i, chunk in enumerate(chunks, start=1):
            await ws.send(
                json.dumps({"message_type": "INPUT_TEXT_CHUNK", "text": chunk, "ack_id": i})
            )
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT))
            if ack.get("message_type") != "TEXT_CHUNK_ACK":
                raise RuntimeError(f"Unexpected INPUT_TEXT_CHUNK response: {ack}")

            # Server processes each chunk asynchronously -- poll
            # FETCH_AUDIO_CHUNK until processing_staus flips to READY.
            deadline = asyncio.get_event_loop().time() + _SESSION_TIMEOUT
            while True:
                await ws.send(json.dumps({"message_type": "FETCH_AUDIO_CHUNK", "chunk_id": i}))
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT))
                status = msg.get("processing_staus")
                if status == "READY":
                    audio_b64 = msg.get("audio_base_64")
                    if audio_b64:
                        audio_parts.append(base64.b64decode(audio_b64))
                    break
                if status == "PROCESSING":
                    if asyncio.get_event_loop().time() > deadline:
                        raise RuntimeError("Timed out waiting for Sahara TTS chunk to render.")
                    await asyncio.sleep(1)
                    continue
                # Unrecognized status (e.g. INPUT_ERROR) -- don't loop
                # forever on something we don't understand.
                raise RuntimeError(f"Unexpected FETCH_AUDIO_CHUNK response: {msg}")

        await ws.send(json.dumps({"message_type": "COMMIT"}))
        try:
            await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)  # COMMITTED_AUDIO summary
        except asyncio.TimeoutError:
            pass  # summary is informational only; audio is already collected

    return _concat_wavs(audio_parts) if audio_parts else b""


def synthesize_speech(
    text: str,
    voice_language: Optional[str] = None,
    voice_accent: Optional[str] = None,
    voice_gender: Optional[str] = None,
) -> bytes:
    """Returns raw audio bytes (WAV by default) for `text`, spoken via
    Sahara 2.5 streaming TTS. Raises RuntimeError on a missing API key or
    any protocol failure -- never silently returns empty audio on error,
    so a UI caller can tell "no reply yet" apart from "TTS broke"."""
    text = (text or "").strip()
    if not text:
        return b""
    try:
        return asyncio.run(
            _synthesize_async(
                text,
                voice_language or config.SAHARA_TTS_VOICE_LANGUAGE,
                voice_accent or config.SAHARA_TTS_VOICE_ACCENT,
                voice_gender or config.SAHARA_TTS_VOICE_GENDER,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Sahara TTS failed: {exc}") from exc
