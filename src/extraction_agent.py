"""
Turns a free-text (or transcribed) multilingual consultation narrative
into a validated ExtractionResult — now including a SOAP-format note
alongside the IDSR fields, from the same LLM call.

Reliability pattern: ask for strict JSON, try to parse it, and if parsing
or schema validation fails, re-prompt the model with the specific error
so it can self-correct — up to config.MAX_EXTRACTION_RETRIES attempts.
This mirrors EpiCast's intake-agent retry design.

Language detection: asked as part of THIS SAME extraction call, not a
separate LLM pass. An earlier version used a dedicated detect_language()
call run right after transcription (see git history / app.py), which
added a full extra round-trip per submission and, on CPU inference,
noticeably worsened latency — likely partly because interleaving a
different system prompt between it and the main extraction call defeats
any prompt-prefix reuse llama.cpp might otherwise do across repeated,
identical extraction system prompts. One call, one system prompt, is
both faster and simpler; detect_language() is kept below for optional
reuse (e.g. an immediate pre-extraction language hint) but is NOT on the
default critical path.
"""
import json
import re
from typing import Optional, Tuple

import config
from src import model_manager
from src.idsr_reference import SEVERITY_LEVELS, SYNDROME_NAMES, reference_block_for_prompt
from src.schema import ExtractionResult

# NOTE on scope: "summary" and "soap_subjective" used to be asked of the
# model here, alongside everything else, in one JSON object. Testing showed
# these two free-authored fields were the most consistent source of
# hallucination (invented cases, fabricated public-health claims) even when
# the structured fields below came out correct. Both are now assembled
# deterministically after validation instead (see _build_summary and
# extract() below) — soap_subjective becomes the narrative itself, and
# summary is templated from the fields the model DID get right. This also
# shortens the required JSON object, which cuts hallucination surface
# further and leaves less room to run out of budget mid-object.
SYSTEM_PROMPT = f"""You are a clinical documentation assistant for community \
health workers using the WHO Integrated Disease Surveillance and Response \
(IDSR) framework in West Africa. Patients describe symptoms in English, \
Nigerian Pidgin, Hausa, Yoruba, Twi, or French — sometimes mixed together.

Your job: read the clinical narrative and output ONLY a single JSON object \
(no prose, no markdown fences, no explanation) with these exact keys:

{{
  "syndrome_category": one of {SYNDROME_NAMES},
  "symptoms": [short symptom strings, in English, ONLY symptoms the patient \
actually reports having — never include a symptom the narrative explicitly denies. \
Example: narrative says "no cough, denies fever" -> do not include "cough" or \
"fever" in symptoms even if they're mentioned elsewhere for context],
  "onset_days": integer number of days since symptom onset, or null if unknown,
  "severity": one of {SEVERITY_LEVELS},
  "age_group": e.g. "infant", "child", "adult", "elderly", or "unknown",
  "sex": "male", "female", or "unknown",
  "icd10_codes": [relevant ICD-10 code strings],
  "reportable": true or false,
  "confidence": number between 0 and 1,
  "language_detected": the primary language of the narrative (English, Nigerian Pidgin, Hausa, Yoruba, Twi, French, or Other/Unknown),
  "soap_objective": "observable/measurable findings mentioned — ONE short sentence under 20 words, or 'Not documented' if none given",
  "soap_assessment": "brief clinical assessment of THIS patient's narrative only — ONE short sentence, under 20 words. Do not mention outbreaks, disease trends, other patients, or public-health commentary — describe only what was reported.",
  "soap_plan": "recommended next steps for THIS patient only — ONE short sentence, under 20 words"
}}

Reference WHO IDSR syndrome categories:
{reference_block_for_prompt()}

If the narrative is ambiguous, choose the closest matching syndrome and \
lower the confidence score accordingly. Keep every field concise — short \
sentences generate faster and are easier for a clinician to scan. Output \
ONLY the JSON object."""

LANGUAGE_ID_SYSTEM_PROMPT = """Identify the dominant language of the following \
clinical narrative, which may mix languages (code-switching). Reply with \
ONLY one label, exactly as written, nothing else — no punctuation, no \
explanation: English, Nigerian Pidgin, Hausa, Yoruba, Twi, French, or \
Other/Unknown."""

_VALID_LANGUAGE_LABELS = {
    "english", "nigerian pidgin", "hausa", "yoruba", "twi", "french", "other/unknown",
}


def detect_language(narrative: str) -> str:
    """Optional standalone language ID pass — NOT called automatically by
    extract() or app.py's do_transcribe (see module docstring for why).
    Kept available in case a future feature genuinely needs a language
    guess before extraction runs; costs one extra LLM call if used."""
    if not narrative or not narrative.strip():
        return "unknown"
    messages = [
        {"role": "system", "content": LANGUAGE_ID_SYSTEM_PROMPT},
        {"role": "user", "content": narrative.strip()},
    ]
    try:
        raw = model_manager.chat(messages=messages, max_tokens=12, temperature=0.0)
    except Exception:
        return "unknown"
    label = raw.strip().splitlines()[0].strip().strip(".\"'")
    if label.lower() not in _VALID_LANGUAGE_LABELS:
        return "unknown"
    return label


def _extract_json_block(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _repair_truncated_json(text: str) -> Optional[dict]:
    """Best-effort recovery for JSON cut off mid-generation.

    Observed cause (see verification in docs/extraction_verification.md):
    the model sometimes emits an end-of-turn token mid-string -- typically
    inside one of the free-text soap_* fields, which come last in the
    schema -- well before max_tokens (~140-150 tokens generated against a
    1536 cap), leaving the object unterminated. Rather than discarding an
    otherwise-complete, correctly generated record just because the LAST
    field got cut off, trim back to the last complete "key": value pair
    and close the object there.

    Every field this can drop has a safe default in ExtractionResult
    except syndrome_category, which is always the FIRST key in the
    prompt's schema and therefore the first one generated -- if
    truncation happened early enough to lose it, the repaired dict will
    still fail pydantic validation same as before repair, so this can't
    silently produce a placeholder record.
    """
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]
    # Try closing the object at each earlier top-level "key": boundary,
    # starting from the latest (drops the least) and working backward.
    for m in reversed(list(re.finditer(r',\s*"\w+"\s*:', text))):
        candidate = text[: m.start()] + "}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_block_with_repair(text: str) -> Tuple[Optional[dict], bool]:
    """Wraps _extract_json_block with a fallback repair pass for output
    truncated mid-generation. Returns (parsed_dict_or_None, was_repaired)."""
    parsed = _extract_json_block(text)
    if parsed is not None:
        return parsed, False
    repaired = _repair_truncated_json(text)
    return repaired, repaired is not None


# Deterministic guard for the negation-handling failure mode observed in
# testing: the model sometimes lists a symptom in `symptoms` that the
# narrative explicitly denies (e.g. narrative says "No cough" but the
# model still returns "cough"), despite the SYSTEM_PROMPT instructing it
# not to. Same spirit as _soap_hallucination_flags below -- a cheap,
# deterministic keyword check, not a general negation-scope parser. This
# catches an exact-word denial next to the symptom term; it won't catch
# every phrasing (e.g. "denies difficulty breathing" won't match a symptom
# the model wrote as "dyspnea") -- that's a known remaining gap.
_NEGATION_CUES = r"(?:no|denies|denied|without|not|absent|negative for)"


def _filter_denied_symptoms(symptoms: list, narrative: str) -> Tuple[list, list]:
    """Drops any symptom whose exact wording is immediately preceded (within
    ~4 words) by a negation cue in the narrative. Returns (kept, dropped)."""
    lowered = narrative.lower()
    kept, dropped = [], []
    for symptom in symptoms:
        word = symptom.replace("_", " ").strip().lower()
        if not word:
            continue
        pattern = rf"\b{_NEGATION_CUES}\b(?:\s+\w+){{0,4}}\s+{re.escape(word)}\b"
        if re.search(pattern, lowered):
            dropped.append(symptom)
        else:
            kept.append(symptom)
    return kept, dropped


def _build_summary(result: ExtractionResult) -> str:
    """SOAP-assembly: build the one-sentence summary from fields the model
    already produced and pydantic already validated, instead of asking the
    model to freely author a sentence. Nothing here can be a fabricated
    claim — it's a template over data we've already checked."""
    onset = f"{result.onset_days}d" if result.onset_days is not None else "onset unknown"
    return (
        f"{result.age_group}, {result.sex}: {result.syndrome_category} "
        f"({result.severity}, {onset})."
    ).replace("unknown, unknown: ", "").strip()


# Keyword-level guard, not a second model call: cheap, deterministic, and
# specifically targets the two hallucination patterns testing surfaced —
# (a) SOAP text naming a symptom/finding absent from both the validated
# `symptoms` list AND the source narrative, and (b) SOAP text making an
# unprompted public-health claim (outbreak/pandemic commentary) that goes
# beyond describing the single patient in front of it. This won't catch
# every hallucination, but it catches the specific, recurring failure mode
# observed in testing, which is the goal — not a general fact-checker.
_SYMPTOM_KEYWORDS = [
    "blood", "bloody", "mucus", "vomit", "diarrhea", "diarrhoea", "rash",
    "seizure", "unconscious", "dehydrat", "bleeding", "jaundice", "paraly",
    "pneumonia",
]
_OVERREACH_PHRASES = [
    "covid", "pandemic", "outbreak", "potential carrier", "should be treated as",
    "multiple similar cases", "cluster in the area",
]


def _soap_hallucination_flags(result: ExtractionResult, narrative: str) -> list:
    """Compare the model-authored SOAP fields against what's actually
    grounded (validated symptoms + source narrative) and flag mismatches."""
    soap_text = " ".join(
        [result.soap_objective, result.soap_assessment, result.soap_plan]
    ).lower()
    grounded_text = (" ".join(result.symptoms) + " " + narrative).lower()

    flags = []
    for kw in _SYMPTOM_KEYWORDS:
        if kw in soap_text and kw not in grounded_text:
            flags.append(
                f"SOAP note mentions '{kw}', which doesn't appear in the "
                f"extracted symptoms or the source narrative."
            )
    for phrase in _OVERREACH_PHRASES:
        if phrase in soap_text:
            flags.append(
                f"SOAP note contains unprompted public-health commentary "
                f"('{phrase}') beyond this single patient's narrative."
            )
    return flags


def extract(narrative: str, language_hint: str = "Auto-detect") -> Tuple[Optional[ExtractionResult], str]:
    """
    Returns (result, error_message). result is None if all retries failed;
    error_message is "" on success.
    """
    if not narrative or not narrative.strip():
        return None, "Please enter or transcribe a consultation narrative first."

    base_user_prompt = f"Language hint: {language_hint}\n\nClinical narrative:\n{narrative.strip()}"

    # Each retry resends a fresh, compact prompt (system + narrative + a short
    # note about the last error) rather than accumulating the full history of
    # prior bad responses. With a retry conversation that keeps growing,
    # total context (system prompt + narrative + all prior attempts) can
    # exceed EPISCRIBE_N_CTX by the 2nd-3rd retry, silently truncating the
    # JSON schema out of the system prompt and causing exactly the kind of
    # "missing required field" failure this loop is meant to correct.
    last_error = ""
    last_raw = ""
    for attempt in range(1, config.MAX_EXTRACTION_RETRIES + 1):
        user_prompt = base_user_prompt
        if last_error:
            user_prompt += (
                f"\n\n(Your previous reply failed validation: {last_error}. "
                f"Reply again with ONLY a single corrected JSON object, no prose.)"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = model_manager.chat(
                messages=messages,
                max_tokens=config.MAX_TOKENS_EXTRACTION,
                temperature=config.TEMPERATURE_EXTRACTION,
            )
        except Exception as exc:  # model/runtime failure — not retryable via prompt
            return None, f"Model inference failed: {exc}"

        last_raw = raw  # kept so we can surface it if every attempt fails

        parsed, was_repaired = _extract_json_block_with_repair(raw)
        if parsed is None:
            last_error = "Response was not valid JSON."
            continue

        parsed.setdefault("raw_narrative", narrative.strip())
        try:
            result = ExtractionResult(**{k: v for k, v in parsed.items() if k != "raw_narrative"})
        except Exception as exc:  # pydantic ValidationError or similar
            last_error = str(exc)
            continue

        # SOAP-assembly: these two are no longer requested from the model
        # (see SYSTEM_PROMPT note) — assembled here from already-validated
        # data instead, which is the whole point: nothing free-authored,
        # nothing to hallucinate.
        result.soap_subjective = narrative.strip()
        # Negation guard: drop any symptom the narrative appears to
        # explicitly deny (see _filter_denied_symptoms docstring) before
        # building the summary/reportability off of it.
        kept_symptoms, denied_symptoms = _filter_denied_symptoms(result.symptoms, narrative)
        result.symptoms = kept_symptoms
        result.summary = _build_summary(result)
        # Hallucination Guard: runs on whatever the model DID freely author
        # (soap_objective/assessment/plan) and surfaces anything ungrounded
        # rather than silently trusting it.
        result.hallucination_flags = _soap_hallucination_flags(result, narrative)
        if denied_symptoms:
            result.hallucination_flags.append(
                "Removed symptom(s) the narrative appears to explicitly "
                f"deny: {', '.join(denied_symptoms)}. Verify against the "
                "original narrative."
            )
        if was_repaired:
            result.hallucination_flags.append(
                "Model output was cut off before finishing (likely mid "
                "soap_assessment/soap_plan) -- this record was recovered "
                "from a truncated response, so soap_assessment/soap_plan "
                "may be blank or incomplete. Review manually."
            )
        return result, ""

    # DEBUG: every attempt failed. Print the last raw model output so we can
    # tell "truncated JSON" (cut off mid-string/object, usually a
    # max_tokens ceiling) apart from "not JSON at all" (a prompt-adherence
    # problem) instead of guessing from the error message alone.
    print(
        "[extraction_agent] All attempts failed. Last raw model output "
        f"({len(last_raw)} chars):\n{last_raw!r}"
    )
    return None, (
        f"Could not produce a valid structured record after "
        f"{config.MAX_EXTRACTION_RETRIES} attempts ({last_error}). "
        f"Please review manually. (Raw model output logged to console.)"
    )
