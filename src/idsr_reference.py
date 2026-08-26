"""
Static WHO IDSR priority-syndrome reference data.

NOTE: ICD-10 mappings and case-definition summaries below are simplified
and indicative only, intended to steer the model's structured output and
to ground the chatbot's guidance. They are NOT a substitute for a
country's official IDSR Technical Guidelines and must be validated by a
clinical/surveillance authority before any real-world reporting use.
"""

IDSR_SYNDROMES = {
    "Acute Watery Diarrhea": {
        "icd10": ["A09"],
        "reportable": True,
        "case_definition": "Three or more loose/watery stools in 24 hours, "
                            "any age; suspect cholera if severe dehydration "
                            "or death in a patient aged 5 years or older.",
    },
    "Acute Bloody Diarrhea": {
        "icd10": ["A09.0"],
        "reportable": True,
        "case_definition": "Diarrhea with visible blood in stool — "
                            "suspect shigellosis/dysentery.",
    },
    "Acute Febrile Illness": {
        "icd10": ["R50.9"],
        "reportable": True,
        "case_definition": "Sudden onset of fever, often with chills, "
                            "in the absence of a localized cause — "
                            "includes suspected malaria.",
    },
    "Acute Respiratory Infection": {
        "icd10": ["J06.9"],
        "reportable": True,
        "case_definition": "Cough or sore throat with breathing "
                            "difficulty of recent onset.",
    },
    "Acute Rash with Fever": {
        "icd10": ["B05", "B01"],
        "reportable": True,
        "case_definition": "Fever with maculopapular rash — "
                            "suspect measles or rubella.",
    },
    "Acute Hemorrhagic Fever": {
        "icd10": ["A99"],
        "reportable": True,
        "case_definition": "Fever with bleeding tendency (gums, skin, "
                            "eyes) or unexplained death with bleeding — "
                            "treat as a public health emergency.",
    },
    "Acute Jaundice Syndrome": {
        "icd10": ["B15-B19"],
        "reportable": True,
        "case_definition": "Acute onset of yellow discoloration of eyes "
                            "or skin, often with dark urine.",
    },
    "Acute Neurological Syndrome": {
        "icd10": ["G04.9"],
        "reportable": True,
        "case_definition": "Acute onset of confusion, stiff neck, or "
                            "altered consciousness — suspect meningitis "
                            "or encephalitis.",
    },
    "Acute Flaccid Paralysis": {
        "icd10": ["A80"],
        "reportable": True,
        "case_definition": "Sudden onset of weakness/floppy limb(s) in "
                            "a child under 15 years — suspect polio.",
    },
    "Neonatal Tetanus": {
        "icd10": ["A33"],
        "reportable": True,
        "case_definition": "Normal suck/cry in the first 2 days of life, "
                            "then inability to suck plus stiffness/spasms "
                            "between day 3 and day 28 of life.",
    },
    "Influenza-Like Illness": {
        "icd10": ["J11"],
        "reportable": False,
        "case_definition": "Sudden onset of fever above 38°C with cough "
                            "or sore throat — sentinel surveillance, not "
                            "typically case-based reporting.",
    },
}

SYNDROME_NAMES = list(IDSR_SYNDROMES.keys())

SEVERITY_LEVELS = ["mild", "moderate", "severe", "unknown"]

SUPPORTED_LANGUAGES = [
    "Auto-detect",
    "English",
    "Nigerian Pidgin",
    "Hausa",
    "Yoruba",
    "Twi",
    "French",
]


def reference_block_for_prompt() -> str:
    """Render the syndrome table as plain text for LLM prompts."""
    lines = []
    for name, info in IDSR_SYNDROMES.items():
        icd = ", ".join(info["icd10"])
        lines.append(
            f"- {name} (ICD-10: {icd}; reportable: {info['reportable']}): "
            f"{info['case_definition']}"
        )
    return "\n".join(lines)
