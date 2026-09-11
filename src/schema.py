"""
Pydantic schema for a validated, structured IDSR clinical record — now
paired with a SOAP-format note from the same encounter (see module-level
comment in extraction_agent.py for why these come from one LLM call).

Every model extraction is parsed into this schema before it is shown to
the user or written to storage — malformed or out-of-vocabulary model
output is rejected here and triggers a retry (see extraction_agent.py).
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.idsr_reference import SEVERITY_LEVELS, SYNDROME_NAMES, program_category_for


class ExtractionResult(BaseModel):
    syndrome_category: str = Field(..., description="One of the WHO IDSR priority syndromes")
    symptoms: List[str] = Field(default_factory=list)
    onset_days: Optional[int] = Field(default=None, ge=0, le=365)
    severity: str = Field(default="unknown")
    age_group: str = Field(default="unknown")
    sex: str = Field(default="unknown")
    icd10_codes: List[str] = Field(default_factory=list)
    reportable: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = Field(default="")
    # Set by extraction_agent.detect_language() before validation, not
    # asked of the LLM as part of the main extraction JSON — a single
    # dedicated language-ID pass (reused from right after transcription,
    # see app.py) is more consistent than asking the model to re-guess it
    # here, and keeps the extraction prompt shorter.
    language_detected: str = Field(default="unknown")
    # WHO IDSR programmatic category (endemic burden / eradication or
    # elimination target / epidemic-prone) — derived from syndrome_category
    # via static reference data below, never asked of the LLM. Any value
    # the model supplies for this field is ignored/overwritten.
    public_health_category: List[str] = Field(default_factory=list)

    # SOAP-format clinical note fields, produced by the same extraction
    # call as the IDSR fields above (one model, one call, two output
    # shapes) — gives clinicians a familiar clinical-note view of the
    # same encounter alongside the surveillance-focused IDSR view.
    soap_subjective: str = Field(default="", description="Patient's reported complaint, in their own words/context")
    soap_objective: str = Field(default="", description="Observable/measurable findings mentioned in the narrative")
    soap_assessment: str = Field(default="", description="Clinical assessment / likely diagnosis reasoning")
    soap_plan: str = Field(default="", description="Recommended next steps or plan")

    # Populated by extraction_agent AFTER validation, not asked of the LLM.
    # Testing at low temperature showed soap_objective/assessment/plan
    # routinely mention symptoms, diagnoses, or public-health claims with
    # no basis in either the extracted `symptoms` list or the source
    # narrative (e.g. a skin-rash case whose SOAP note described vomiting;
    # a single-case respiratory complaint whose summary invented a COVID
    # outbreak warning). Each string here names one such ungrounded claim,
    # so a reviewer sees exactly what to double-check instead of having to
    # re-read the whole note looking for it. Empty list = nothing flagged.
    hallucination_flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_public_health_category(self) -> "ExtractionResult":
        self.public_health_category = program_category_for(self.syndrome_category)
        return self

    @field_validator("syndrome_category")
    @classmethod
    def syndrome_must_be_known(cls, v: str) -> str:
        # Fuzzy-match against the controlled vocabulary rather than hard-fail,
        # since the model may return slightly different casing/wording.
        normalized = v.strip().lower()
        for name in SYNDROME_NAMES:
            if name.lower() == normalized:
                return name
        for name in SYNDROME_NAMES:
            if normalized in name.lower() or name.lower() in normalized:
                return name
        raise ValueError(
            f"'{v}' is not a recognized WHO IDSR syndrome category. "
            f"Must be one of: {', '.join(SYNDROME_NAMES)}"
        )

    @field_validator("severity")
    @classmethod
    def severity_must_be_known(cls, v: str) -> str:
        v = (v or "unknown").strip().lower()
        return v if v in SEVERITY_LEVELS else "unknown"


class EncounterRecord(ExtractionResult):
    """A saved encounter — extraction result plus storage metadata."""
    id: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    raw_narrative: str = ""
    # sha256 (truncated) of the source audio, when the narrative came from
    # a recording — lets two records be compared to tell "same audio
    # resubmitted" apart from "different audio, identical transcript"
    # (Whisper hallucination/memorization) after the fact. Empty for
    # narratives that were typed rather than transcribed.
    audio_hash: str = ""
