# Extraction pipeline verification (priority task 2)

**Date:** 2026-09-14
**Goal:** confirm/deny the suspicion that extraction defaults to a generic diagnosis
regardless of input, per CLAUDE.md's unresolved question about commit `caa6bca`
(whose actual diff included substantial, undocumented "extraction reliability
fixes" — SOAP-hallucination guard, templated summary, retry logging — beyond
what its commit message described).

## Method

Ran the real `extraction_agent.extract()` (unmodified) against 4 deliberately
different mock transcripts — different syndrome, severity, onset, and (for 2 of
them) an explicit symptom denial to test negation handling — logging the raw
model output at every attempt, not just the final parsed record. Model:
EpiCast MedGemma 4B GGUF (Q4_K_M), CPU-only, `N_THREADS=4`, `N_GPU_LAYERS=0`
(this machine's defaults).

## Headline finding: the generic-extraction fear is NOT confirmed

Output clearly varies by input. The two cases that completed successfully produced
genuinely different, correct records:

| Case | syndrome_category | severity | onset_days | icd10_codes |
|---|---|---|---|---|
| Cholera-like watery diarrhea, adult | Acute Watery Diarrhea | severe | 1 | A00, A09 |
| Neurological syndrome, infant | Acute Neurological Syndrome | severe | 1 | G04, A39 |

No silent JSON-parse fallback returning a placeholder record exists (checked —
on parse/validation failure the loop retries, then returns `None` with an
explicit error, never a fake record). No rule-based fallback path exists. The
narrative is correctly interpolated into the prompt (confirmed via an assert
before running any case). `TEMPERATURE_EXTRACTION=0.1` is not causing mode
collapse — the two successful outputs are substantively different. **Every
culprit CLAUDE.md listed, in the order it listed them, was checked and ruled
out.**

## But two different, real reliability issues surfaced

**A) ~50% JSON-completion failure rate in this small sample.** The other 2 of
4 cases failed all 3 retries with "Response was not valid JSON." The raw
output in both cases shows the model didn't actually write invalid JSON — it
cut off mid-string, mid-sentence, with no closing quote/brace, after only
~140-150 tokens (well under the 1536-token cap, so this is not a budget
truncation). This is the same premature-stop symptom `model_manager.py`'s
docstring already documents and partially mitigated (`repeat_penalty`
1.3 -> 1.15) — that mitigation is evidently incomplete; it still happens on
roughly half the cases tested here. When all 3 retries fail this way, the
clinician gets nothing and has to document the encounter manually.

**B) Negation handling is unreliable.** Both negation-testing cases explicitly
denied a symptom in the narrative ("No cough", "denies any difficulty
breathing"), and in both, the raw model output listed that denied symptom in
`symptoms` anyway — directly violating the SYSTEM_PROMPT instruction ("never
include a symptom the narrative explicitly denies"). This is visible only
because these two records happened to fail JSON validation and get logged in
full; the current hallucination guard (`_soap_hallucination_flags`) would not
catch this even on a successful record, since it only checks the free-authored
SOAP fields against the narrative/symptoms list, not whether `symptoms` itself
contradicts the narrative.

**Minor, unflagged:** demographic fields get invented when not stated (e.g.
`"sex": "female"` for a child whose sex was never mentioned in the narrative).
Outside current hallucination-guard scope (symptom keywords + overreach
phrases only).

**C) Latency: ~370-560s (6-9 min) per single generation attempt** on this
CPU-only setup (4 threads, no GPU). A 3-retry failure took ~500-550s total.
This is a hardware/config constraint, not an extraction-logic bug, but it's
severe enough to affect the "Technical Execution: latency handling" judging
criterion (15%) and any live demo. Flagged for awareness, not fixed as part
of this verification.

## Fixes applied and re-verified (same day)

Per project decision, fixed (A) and (B) above, left (C, latency) as a known
limitation. Implementation: `_repair_truncated_json()` trims a cut-off
response back to the last complete `"key": value` pair and closes the object
there (drops only the broken trailing field — every field it can drop has a
safe schema default except `syndrome_category`, which is always generated
first, so a repair that loses required data still fails validation exactly
as before, not a new silent-fallback path); `_filter_denied_symptoms()` is a
deterministic keyword guard, same spirit as the existing
`_soap_hallucination_flags`, that drops a symptom explicitly negated near it
in the narrative. Both are surfaced via `hallucination_flags`, never applied
silently. Regression coverage added to `scripts/smoke_test.py`.

Re-ran the same 4 live cases end-to-end afterward:

- **4/4 succeeded** (previously 2/4). The two that hit the truncation bug
  again were recovered instead of hard-failing, each clearly flagged.
- **Negation guard confirmed live**: the measles case's raw output still
  included "cough" despite the narrative saying "No cough" (confirms the
  model doesn't reliably follow the instruction on its own) — the guard
  caught and removed it, flagged: *"Removed symptom(s) the narrative
  appears to explicitly deny: cough."* The Yoruba-style synonym gap remains
  as documented ("dyspnea" vs. "denies difficulty breathing" still isn't
  caught).

**New observation, not fixed:** one recovered record's `soap_assessment`
contained bizarre, off-topic content unrelated to the clinical narrative
(a tangent about social media and economic hardship, in Pidgin) before the
truncation point. This confirms the truncation-adjacent text is sometimes
genuinely low-quality model output, not an artifact of the repair logic —
a real model-quality/hallucination-drift issue in the free-text SOAP fields
under CPU/quantized inference, separate from the two fixed bugs. Flagged for
awareness; not addressed here.

## Conclusion

Extraction is genuinely input-sensitive — the specific bug CLAUDE.md worried
about does not exist. Testing surfaced two different, real issues instead
(JSON-completion reliability, negation handling), both now fixed and
re-verified against the live model, plus a latency concern and a SOAP
content-quality observation, both left as known limitations for now.
