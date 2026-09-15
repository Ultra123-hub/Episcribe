# Sahara STT verification (priority task 1)

**Date:** 2026-09-14
**Goal:** confirm Sahara 2.5 STT (`src/transcribe.py`, `config.STT_BACKEND=sahara`) actually
produces usable transcripts on real code-switched audio, not just that the integration
compiles and returns *something*.

## Method

Ran the real `transcribe_audio()` function (unmodified) against real clinical-consultation
recordings from [intronhealth/AfriSwitchCare](https://huggingface.co/datasets/intronhealth/AfriSwitchCare)
— the hackathon-provided medical code-switching dataset, on-domain for this app. One sample
per target pair, manually compared against the dataset's own ground-truth transcript.

Sahara's sync upload endpoint caps audio at 120s/request (see `transcribe.py` docstring).
AfriSwitchCare's consultation clips run 90-800s, so Yoruba and Pidgin samples were trimmed
to their first 110s before submission; Hausa (89s) needed no trimming.

## Results

| Pair | Status | Notes |
|---|---|---|
| Hausa-English | ✅ Pass | Near word-for-word match to ground truth; clearly real content |
| Nigerian Pidgin-English | ✅ Pass | Coherent, faithful to ground truth |
| Yoruba-English | ✅ Pass, weaker | Correctly captures Yoruba diacritics/words, but English portions get noticeably garbled (e.g. "Ok, Google, monidon OK Samuel OK OP" for what should be an English greeting). Real transcription, not broken, but the weakest of the three — flag as a relative weak point in the benchmark report. |
| Akan-English | ❌ Not tested | **Neither hackathon-linked dataset has Akan/Twi audio.** Checked both `intronhealth/AfriSwitch` (gated, couldn't query) and `intronhealth/AfriSwitchCare` (accessible; 9 languages: Hausa, Yoruba, Pidgin, Amharic, French, Igbo, Kinyarwanda, Swahili, Zulu — no Akan/Twi). Decision (2026-09-14): drop Akan-English from the benchmark and document this gap in the final report rather than substitute synthetic audio. |

**Conclusion: Sahara STT integration is confirmed working on real code-switched audio** for
Hausa-EN, Pidgin-EN, and Yoruba-EN. No sign of the "returns something regardless of input"
failure mode — transcripts clearly track actual audio content and vary sensibly per sample.

## Known follow-up (blocks task 5 benchmark, not task 1)

AfriSwitchCare's real consultation clips mostly exceed Sahara's 120s sync cap (only trimming
made today's verification possible). The benchmark (task 5) needs full-length clips to be
representative, so the Sahara async upload endpoint needs to be wired up before that step —
decided 2026-09-14, tracked as follow-up work, not done as part of this verification.

## Raw output

Full transcripts, ground-truth comparisons, and timings from the verification run are not
checked into the repo (audio samples aren't either, per `.gitignore`) — re-run
against fresh `intronhealth/AfriSwitchCare` samples to reproduce.

## Addendum (2026-09-15): "Auto-detect" never actually worked against Sahara

All verification above used an explicit language hint (Hausa/Yoruba/Nigerian Pidgin) for
every Sahara call — "Auto-detect" (the app's own UI default) was never tested against Sahara
here. It turned out not to work: a pre-existing comment in `transcribe.py` claimed omitting
`use_language_asr_input` let Sahara auto-detect the language, based on testing that, in
hindsight, never actually covered the Auto-detect path either. Live testing today (prompted by
a real user hitting this during a demo dry run) confirmed `use_language_asr_input` is REQUIRED
by Sahara's file-upload endpoints — no auto-detect mode exists. Fixed: `_transcribe_via_sahara`
now raises a clear, actionable error when no specific language is selected, instead of
surfacing Sahara's raw rejection. See `src/transcribe.py`'s module docstring and git history
for detail.
