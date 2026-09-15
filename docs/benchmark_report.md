# STT Benchmark Report — Sahara 2.5 vs faster-whisper vs MMS

**Date:** 2026-09-15
**Task:** Sahara CodeSwitch Africa Challenge, deliverable 2 (max 3 pages)

## Scope and method

**Scale:** due to submission time constraints, this is a small pilot
benchmark — **one real clinical-consultation clip per language pair**
(Nigerian Pidgin-English, Hausa-English, Yoruba-English), not the full
AfriSwitchCare test set. Akan-English is not included: neither
`intronhealth/AfriSwitch` nor `intronhealth/AfriSwitchCare` (the two
hackathon-linked datasets) contain Akan/Twi audio at all — confirmed by
querying both datasets' configs directly (see
`docs/sahara_stt_verification.md`). Results below should be read as
directional, not statistically definitive.

**Source data:** real, human-transcribed clinical-consultation audio from
[intronhealth/AfriSwitchCare](https://huggingface.co/datasets/intronhealth/AfriSwitchCare)
(Africa's medical code-switching dataset, on-domain for this app), used
full-length and unmodified — Hausa (89.2s), Yoruba (199.9s), Pidgin (302.1s).

**Models compared:**
- **Sahara 2.5** (this app's STT backend) — via Sahara's async
  upload-and-poll endpoint (see `docs/sahara_stt_verification.md` for why
  async, not the 120s-capped sync endpoint).
- **faster-whisper** ("small") — this app's existing offline default,
  fully local, CPU.
- **MMS** (`facebook/mms-1b-all`) — Meta's Massively Multilingual Speech
  model, chosen as the 3rd model per the "lowest integration risk" rule
  (an existing local HF `transformers` pipeline, no new API signup),
  and because it's purpose-built for exactly this kind of low-resource
  African-language coverage — a more informative contrast than another
  Whisper checkpoint. Ran with its `hau`/`yor`/`pcm` language adapters.

**Metrics:** WER/CER, both normalized and unnormalized, computed with
`jiwer`, using the same methodology as the hackathon's own benchmarking
reference
([intron-innovation/Intron-Multimodal-Benchmarking](https://github.com/intron-innovation/Intron-Multimodal-Benchmarking),
`scripts/evaluations.py`): the normalized pass applies
`whisper_normalizer`'s `BasicTextNormalizer` (lowercase, strip
punctuation/diacritics, collapse whitespace) since all three languages
here are non-English/code-switched — the same normalizer that script uses
for its non-English branch. Latency is wall-clock per full clip, this
machine (CPU-only, 4 threads, no GPU). Reproduce via
`scripts/benchmark_stt.py benchmark/results.json`; raw
transcripts/references are in `benchmark/results.json`.

## Results

### Normalized WER (lower is better)

| Language | Sahara 2.5 | faster-whisper | MMS |
|---|---|---|---|
| Hausa | **50.4%** | 79.3% | 88.9% |
| Pidgin | **25.5%** | 44.1% | 69.9% |
| Yoruba | **62.0%** | 73.1% | 85.5% |

### Normalized CER (lower is better)

| Language | Sahara 2.5 | faster-whisper | MMS |
|---|---|---|---|
| Hausa | **28.8%** | 43.8% | 53.5% |
| Pidgin | **16.3%** | 27.3% | 35.8% |
| Yoruba | 43.3% | **57.7%*** | 49.3% |

*(faster-whisper's Yoruba output was largely non-linguistic hallucination — see below — so its low relative CER here understates how unusable the transcript actually is; WER captures that better.)*

### Latency (wall-clock, seconds, full clip)

| Language | Duration (s) | Sahara 2.5 | faster-whisper | MMS |
|---|---|---|---|---|
| Hausa | 89.2 | **25.1** | 275.2 | 574.6 |
| Pidgin | 302.1 | **44.4** | 317.2 | 1379.5 |
| Yoruba | 199.9 | **72.2** | 2165.5 | 1374.2 |

## Findings

**Sahara 2.5 wins decisively on both accuracy and latency, on every
language tested.** It has the lowest normalized WER and CER on all three
pairs (the one CER exception is explained above), and is 5-30x faster
than either local model — expected, since Sahara runs on Intron's own
managed inference infrastructure while the other two ran unaccelerated on
a CPU-only laptop with no GPU. A GPU-accelerated local deployment would
close the latency gap somewhat, but not the accuracy gap seen here.

**Absolute WER is high across all three models** (25-89%) — this is
real, code-switched, in-the-wild clinical audio, not clean read speech,
and matches the AfriSwitch paper's own finding that African-accented,
code-switched speech is substantially harder than standard benchmarks.
Sahara's ~25-50% range is the most usable of the three as a *first-pass
draft a clinician reviews and corrects* (this app's existing design
intent for non-English transcripts — see README's "Important caveats"),
not as ground truth.

**faster-whisper's Yoruba result was effectively unusable** — largely
non-linguistic hallucinated text ("HY KÖIM NO MÖVi[...]chords or mustard
Ëa penguin veckä İnghis weenu...") rather than a rough transcript, and it
was also the single slowest run recorded (2165s for a 200s clip) —
consistent with this app's own README already flagging limited
Whisper coverage of Yoruba/Hausa/Pidgin/Twi.

**MMS's output frequently lost word boundaries** — e.g. its Hausa
transcript reads "asuankasaayakubsa52yakaasbiyaya asibitisabaacauwa..."
where words have run together with letters dropped, rather than clean
space-separated text. This inflates its WER beyond what a corrected
decoding/post-processing setup might achieve; the model itself may carry
useful signal, but its out-of-the-box `transformers` pipeline output
wasn't competitive as delivered here.

## Per-model strengths/weaknesses

| Model | Strengths | Weaknesses |
|---|---|---|
| **Sahara 2.5** | Best accuracy and by far the best latency on every pair tested; purpose-built for African code-switched speech; async endpoint removes any practical length limit | Requires internet + API key; per-request cost; this app's async integration is new (added today) and only lightly tested |
| **faster-whisper** | Fully offline, no API key, already the app's default; reasonable on Pidgin | Weak on Hausa/Yoruba, occasionally hallucinates badly (Yoruba here); far slower than Sahara on CPU |
| **MMS** | Fully offline; dedicated adapters for Hausa/Yoruba/Pidgin (1000+ languages); no API key | Word-boundary/spacing problems in output as tested; slowest of the three; large model (~4GB) and heaviest dependency footprint (torch + transformers) |

## Caveats

- One clip per language is not enough to draw firm statistical
  conclusions — treat this as a directional pilot, not a final verdict.
- All three models ran on the same CPU-only machine; Sahara's latency
  advantage partly reflects its managed infrastructure vs. no local GPU
  for the other two — a fairer latency comparison would need
  GPU-accelerated local inference.
- MMS's word-boundary issue may be fixable with different decoding
  settings not explored here, given time constraints.
