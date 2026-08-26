"""
Turns a free-text (or transcribed) multilingual consultation narrative
into a validated ExtractionResult.

Reliability pattern: ask for strict JSON, try to parse it, and if parsing
or schema validation fails, re-prompt the model with the specific error
so it can self-correct — up to config.MAX_EXTRACTION_RETRIES attempts.
This mirrors EpiCast's intake-agent retry design.
"""
import json
import re
from typing import Optional, Tuple

import config
from src import model_manager
from src.idsr_reference import SEVERITY_LEVELS, SYNDROME_NAMES, reference_block_for_prompt
from src.schema import ExtractionResult

SYSTEM_PROMPT = f"""You are a clinical documentation assistant for community \
health workers using the WHO Integrated Disease Surveillance and Response \
(IDSR) framework in West Africa. Patients describe symptoms in English, \
Nigerian Pidgin, Hausa, Yoruba, Twi, or French — sometimes mixed together.

Your job: read the clinical narrative and output ONLY a single JSON object \
(no prose, no markdown fences, no explanation) with these exact keys:

{{
  "syndrome_category": one of {SYNDROME_NAMES},
  "symptoms": [short symptom strings, in English],
  "onset_days": integer number of days since symptom onset, or null if unknown,
  "severity": one of {SEVERITY_LEVELS},
  "age_group": e.g. "infant", "child", "adult", "elderly", or "unknown",
  "sex": "male", "female", or "unknown",
  "icd10_codes": [relevant ICD-10 code strings],
  "reportable": true or false,
  "confidence": number between 0 and 1,
  "summary": one-sentence plain-English clinical summary,
  "language_detected": the primary language of the narrative
}}

Reference WHO IDSR syndrome categories:
{reference_block_for_prompt()}

If the narrative is ambiguous, choose the closest matching syndrome and \
lower the confidence score accordingly. Output ONLY the JSON object."""


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


def extract(narrative: str, language_hint: str = "Auto-detect") -> Tuple[Optional[ExtractionResult], str]:
    """
    Returns (result, error_message). result is None if all retries failed;
    error_message is "" on success.
    """
    if not narrative or not narrative.strip():
        return None, "Please enter or transcribe a consultation narrative first."

    user_prompt = f"Language hint: {language_hint}\n\nClinical narrative:\n{narrative.strip()}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = ""
    for attempt in range(1, config.MAX_EXTRACTION_RETRIES + 1):
        try:
            raw = model_manager.chat(
                messages=messages,
                max_tokens=config.MAX_TOKENS_EXTRACTION,
                temperature=config.TEMPERATURE_EXTRACTION,
            )
        except Exception as exc:  # model/runtime failure — not retryable via prompt
            return None, f"Model inference failed: {exc}"

        parsed = _extract_json_block(raw)
        if parsed is None:
            last_error = "Response was not valid JSON."
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That was not valid JSON ({last_error}). "
                            f"Reply again with ONLY the JSON object, nothing else.",
            })
            continue

        parsed.setdefault("raw_narrative", narrative.strip())
        try:
            result = ExtractionResult(**{k: v for k, v in parsed.items() if k != "raw_narrative"})
            return result, ""
        except Exception as exc:  # pydantic ValidationError or similar
            last_error = str(exc)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That JSON failed validation: {last_error}. "
                            f"Reply again with ONLY a corrected JSON object.",
            })
            continue

    return None, f"Could not produce a valid structured record after {config.MAX_EXTRACTION_RETRIES} attempts ({last_error}). Please review manually."
