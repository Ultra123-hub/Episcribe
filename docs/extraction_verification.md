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

## Conclusion

Extraction is genuinely input-sensitive — the specific bug CLAUDE.md worried
about does not exist. Testing surfaced two different, real issues instead
(JSON-completion reliability, negation handling) plus a latency concern,
which are separate follow-up decisions.
