# Privacy & consent — what leaves the device

EpiScribe is offline-first by default. This note lists every point where data
leaves the device, so a health worker can get informed consent from the
patient before recording, and so this repo's public submission is honest
about what "voice-first" actually implies.

## What stays fully local, always

- **Structured extraction** (turning a narrative into an IDSR record) — runs
  entirely on-device via the local MedGemma GGUF model (`llama-cpp-python`).
  The clinical narrative never leaves the device for this step, regardless
  of configuration.
- **Saved records** — stored in a local SQLite file (`data/episcribe.db`).
  Nothing is synced anywhere unless the clinician explicitly exports a CSV
  and shares it themselves.
- **Default speech-to-text** — `EPISCRIBE_STT_BACKEND` defaults to
  `whisper` (faster-whisper), which runs fully on-device. No audio leaves
  the device unless this is explicitly switched to `sahara`.

## What leaves the device, and when

| Data | Destination | When |
|---|---|---|
| Raw consultation audio | Sahara (Intron Voice API), `infer.voice.intron.io` | Only if `EPISCRIBE_STT_BACKEND=sahara` is set. Off by default. |
| Assistant reply text | Sahara (Intron Voice API), `infer.voice.intron.io` | Whenever the Assistant tab is used and `SAHARA_API_KEY` is set — every reply is sent for spoken playback. If the clinician asks the assistant to "show my recent records," the reply can include real patient-record excerpts, which are then sent to Sahara for speech synthesis. |
| Model weights (MedGemma GGUF, Whisper) | Hugging Face Hub | One-time, on first run only. Downloads the app's own model files — never patient data. |

**Both Sahara calls are opt-in in practice**: STT defaults to fully local
Whisper, and TTS only fires when `SAHARA_API_KEY` is configured. A deployment
that never sets `SAHARA_API_KEY` and leaves `EPISCRIBE_STT_BACKEND` at its
default sends no patient data anywhere.

Sahara audio uploads use Intron's synchronous file endpoint over HTTPS with
a bearer API key. This project doesn't control Sahara's own data retention —
see [Intron's own privacy documentation](https://www.intron.io/) for that.

## Multi-user / public deployment (e.g. Hugging Face Spaces)

If this app is deployed as a shared, public instance, **every visitor reads
and writes the same local database** — there is no per-user account
separation. The Records tab (which lists and exports saved encounters) is
already hidden on Hugging Face Spaces for exactly this reason
(`config.ON_HF_SPACES`), but the underlying encounters are still stored in
one shared file on that instance. A public deployment should not be used
with real patient data unless it's given per-user isolation first — treat a
public demo instance as for synthetic/test narratives only.

## Recommended consent practice

Before recording a consultation with STT set to `sahara`, tell the patient:
*"I'm going to record this conversation and send the audio to a
transcription service to help document your visit. The recording isn't
kept after transcription on our end, but it does leave this device."*
Get a clear yes before recording. If the patient prefers not to have audio
leave the device, switch to typed narrative or the local Whisper backend.

## Known gaps (not addressed in this submission)

- No encryption at rest for `data/episcribe.db`.
- No per-user authentication/isolation for multi-user deployments.
- No documented data-retention policy on Sahara's side from this project's
  perspective (see Intron's own documentation).

This is a hackathon proof-of-concept, not a validated, deployable clinical
tool — see the README's "Important caveats" section.
