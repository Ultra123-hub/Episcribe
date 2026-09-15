---
title: EpiScribe
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: '3.11'
app_file: app.py
pinned: false
---

# EpiScribe

Voice-first, multilingual clinical documentation assistant for WHO IDSR
reporting, with an integrated guidance chatbot — built lean, on Python +
Gradio, reusing [EpiCast](https://github.com/Janeodum/epicast)'s
fine-tuned, quantized MedGemma 4B model. Built for the
[Sahara CodeSwitch Africa Challenge](https://www.intron.io/compete/),
adding Sahara 2.5 speech (streaming STT + TTS) on top of the original
offline-first design.

See `docs/CONCEPT_NOTE.md` (or the original Word concept note) for the
background and architecture rationale. This repo is the implementation.

**Hackathon deliverables:** [benchmark report](docs/benchmark_report.md)
(Sahara 2.5 vs faster-whisper vs MMS) ·
[privacy/consent note](docs/privacy_consent.md) ·
[Sahara STT verification](docs/sahara_stt_verification.md) ·
[extraction pipeline verification](docs/extraction_verification.md)

## What it does

- **New Consultation** — type, or dictate via Sahara 2.5 STT (or
  offline faster-whisper) in English, Nigerian Pidgin, Hausa, Yoruba,
  Twi, or French; get back a validated, WHO IDSR-coded structured record
  (syndrome, severity, ICD-10, reportable flag, confidence) in seconds.
- **Records** — browse, filter, and export saved encounters to CSV for
  district-level IDSR line-list reporting.
- **Assistant** — a chatbot (same model, different prompt) for IDSR
  case-definition questions and quick recall of your own saved records,
  with spoken replies via Sahara 2.5 streaming TTS.

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Model | Extraction + chat engine (one model, two roles) | MedGemma 4B, LoRA-tuned, GGUF Q4_K_M, via `llama-cpp-python` |
| Application | Structured extraction agent | Pydantic schema + JSON-repair retry loop, negation guard |
| Application | Guidance chatbot agent | Same model, IDSR system prompt + local record lookup |
| Speech (STT) | Sahara 2.5 (opt-in) or offline default | Intron Voice API (async upload+poll), or `faster-whisper` |
| Speech (TTS) | Spoken assistant replies | Sahara 2.5 streaming TTS (WebSocket) |
| Data | Local record store | SQLite (single file) |
| UI | Gradio Blocks | 3 tabs, runs on Hugging Face Spaces or locally |

Speech-to-text defaults to fully offline (`faster-whisper`); set
`EPISCRIBE_STT_BACKEND=sahara` and `SAHARA_API_KEY` in `.env` to use
Sahara 2.5 instead — see [privacy_consent.md](docs/privacy_consent.md)
for exactly what that sends off-device. TTS (spoken assistant replies)
requires `SAHARA_API_KEY` regardless of STT backend.

## Prerequisites (Windows + VS Code)

- Python 3.10 or 3.11 (3.12 also generally works)
- [VS Code](https://code.visualstudio.com/) with the Python extension
- ~4GB free disk space (model + Whisper weights)
- Internet access **only for the first run** (to download the model)

> **Note on `llama-cpp-python` on Windows:** pip usually installs a
> prebuilt wheel. If it instead tries to compile from source and fails,
> either install the
> ["Desktop development with C++"](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> workload for Visual Studio Build Tools, **or** use a prebuilt CPU
> wheel:
> ```
> pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```

## Setup

```powershell
git clone <your-repo-url> episcribe
cd episcribe
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_model.py   # one-time, ~2.49GB
python app.py
```

Open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

## Deploying to Hugging Face Spaces

1. Create a new Space (SDK: **Gradio**, hardware: CPU basic is fine).
2. Push this repo's contents to the Space (`app.py` at the root).
3. Do **not** commit the `models/` or `data/` contents — the `.gitignore`
   already excludes them; the model downloads automatically on the
   Space's first boot.

## Configuration

Copy `env.example` to `.env` to override any default (model source,
context size, thread count, Whisper model size, Sahara STT/TTS, etc.).
See that file for all options. To enable Sahara 2.5 speech, set
`SAHARA_API_KEY` (get one from [voice.intron.io](https://voice.intron.io))
and, for STT, `EPISCRIBE_STT_BACKEND=sahara`.

## Benchmark

`scripts/benchmark_stt.py` compares Sahara 2.5, faster-whisper, and MMS
(`facebook/mms-1b-all`) on real code-switched clinical audio, using the
same normalized-WER methodology as the hackathon's own benchmarking
reference. See [docs/benchmark_report.md](docs/benchmark_report.md) for
results and `benchmark/` for raw data. Reproducing it needs extra
dependencies not required to run the app itself — see
`benchmark/requirements-benchmark.txt`.

## Important caveats

- ICD-10 mappings and case definitions in `src/idsr_reference.py` are
  **simplified and indicative**, meant to steer model output — validate
  against your country's official IDSR Technical Guidelines before any
  real-world reporting use.
- Offline speech-to-text (the default) uses general-purpose Whisper,
  which has solid English/French support but limited coverage of
  Nigerian Pidgin, Hausa, Yoruba, and Twi. Sahara 2.5 (opt-in) is
  materially more accurate on those — see
  [docs/benchmark_report.md](docs/benchmark_report.md) — but even so,
  treat transcripts in those languages as a draft to review, not ground
  truth. The narrative box is always editable before extraction.
- Sahara STT/TTS send audio/text off-device when enabled — see
  [docs/privacy_consent.md](docs/privacy_consent.md) before using either
  with real patient data.
- CPU-only inference (no GPU) is slow — extraction can take several
  minutes per encounter on modest hardware. See
  [docs/extraction_verification.md](docs/extraction_verification.md).
- This is a lean proof-of-concept scoped to a hackathon timeline, not a
  validated clinical or regulatory tool.
