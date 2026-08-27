---
title: EpiScribe
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# EpiScribe

Offline-capable, multilingual clinical documentation assistant for WHO
IDSR reporting, with an integrated guidance chatbot — built lean, on
Python + Gradio, reusing [EpiCast](https://github.com/Janeodum/epicast)'s
fine-tuned, quantized MedGemma 4B model.

See `docs/CONCEPT_NOTE.md` (or the original Word concept note) for the
background and architecture rationale. This repo is the implementation.

## What it does

- **New Consultation** — type or dictate a consultation narrative in
  English, Nigerian Pidgin, Hausa, Yoruba, Twi, or French; get back a
  validated, WHO IDSR-coded structured record (syndrome, severity,
  ICD-10, reportable flag, confidence) in seconds, fully offline.
- **Records** — browse, filter, and export saved encounters to CSV for
  district-level IDSR line-list reporting.
- **Assistant** — a chatbot (same model, different prompt) for IDSR
  case-definition questions and quick recall of your own saved records.

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Model | Extraction + chat engine (one model, two roles) | MedGemma 4B, LoRA-tuned, GGUF Q4_K_M, via `llama-cpp-python` |
| Application | Structured extraction agent | Pydantic schema + JSON-repair retry loop |
| Application | Guidance chatbot agent | Same model, IDSR system prompt + local record lookup |
| Speech | Offline transcription | `faster-whisper` |
| Data | Local record store | SQLite (single file) |
| UI | Gradio Blocks | 3 tabs, runs on Hugging Face Spaces or locally |

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

Copy `.env.example` to `.env` to override any default (model source,
context size, thread count, Whisper model size, etc.). See that file
for all options.

## Important caveats

- ICD-10 mappings and case definitions in `src/idsr_reference.py` are
  **simplified and indicative**, meant to steer model output — validate
  against your country's official IDSR Technical Guidelines before any
  real-world reporting use.
- Offline speech-to-text uses general-purpose Whisper, which has solid
  English/French support but limited coverage of Nigerian Pidgin,
  Hausa, Yoruba, and Twi. Treat transcripts in those languages as a
  draft to review, not ground truth — the narrative box is always
  editable before extraction.
- This is a lean proof-of-concept scoped to one week of build time, not
  a validated clinical or regulatory tool.
