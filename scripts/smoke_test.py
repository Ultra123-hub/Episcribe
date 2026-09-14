"""Smoke test for logic that doesn't require the LLM (run manually, not shipped as a real test suite)."""
import sys, tempfile, os
from pathlib import Path
# Project root, not this file's own directory (scripts/) -- matches the
# pattern in download_model.py. The previous one-line version only added
# scripts/ itself, which doesn't resolve "import config" at the project
# root; commit caa6bca's message claimed this exact fix had already been
# made and verified passing, but the code here was never actually changed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
config.DB_PATH = __import__("pathlib").Path(tempfile.mktemp(suffix=".db"))

from src.schema import ExtractionResult, EncounterRecord
from src.extraction_agent import (
    _extract_json_block,
    _extract_json_block_with_repair,
    _filter_denied_symptoms,
)
from src import storage

print("== Schema validation ==")
ok = ExtractionResult(
    syndrome_category="acute rash with fever",  # lowercase / fuzzy match
    symptoms=["fever", "rash"],
    onset_days=2,
    severity="MODERATE",  # wrong case, should normalize
    icd10_codes=["B05"],
    reportable=True,
    confidence=0.82,
    summary="Likely measles",
    language_detected="Nigerian Pidgin",
)
assert ok.syndrome_category == "Acute Rash with Fever", ok.syndrome_category
assert ok.severity == "moderate", ok.severity
print("PASS: fuzzy syndrome + severity normalization ->", ok.syndrome_category, ok.severity)

try:
    ExtractionResult(syndrome_category="Common Cold", confidence=0.5)
    print("FAIL: should have rejected unknown syndrome")
except Exception:
    print("PASS: unknown syndrome correctly rejected")

print("\n== JSON extraction from messy model output ==")
messy = '```json\n{"syndrome_category": "Acute Watery Diarrhea", "symptoms": ["diarrhea"], "confidence": 0.7}\n```'
parsed = _extract_json_block(messy)
assert parsed and parsed["syndrome_category"] == "Acute Watery Diarrhea"
print("PASS: parsed fenced JSON ->", parsed)

noisy = 'Sure, here is the JSON: {"syndrome_category": "Acute Febrile Illness", "confidence": 0.6} Hope that helps!'
parsed2 = _extract_json_block(noisy)
assert parsed2 and parsed2["syndrome_category"] == "Acute Febrile Illness"
print("PASS: parsed JSON embedded in prose ->", parsed2)

print("\n== Truncated-JSON repair (docs/extraction_verification.md) ==")
# Real raw output captured during task-2 verification: the model hit an
# early end-of-turn token mid-soap_assessment, well under max_tokens,
# leaving the object unterminated.
truncated = (
    '{\n  "syndrome_category": "Acute Rash with Fever",\n  "symptoms": '
    '[\n    "cough",\n    "fever"\n  ],\n  "severity": "moderate",\n  '
    '"confidence": 0.89,\n  "soap_objective": "rash on face.",\n  '
    '"soap_assessment": "this is not typical for measles or chickenpox.'
)
parsed3, was_repaired = _extract_json_block_with_repair(truncated)
assert parsed3 and was_repaired and parsed3["syndrome_category"] == "Acute Rash with Fever"
assert "soap_assessment" not in parsed3, "repair should drop the truncated trailing field, not guess at it"
print("PASS: recovered syndrome_category/symptoms from truncated JSON, dropped the broken trailing field ->", parsed3)

complete = '{"syndrome_category": "Acute Watery Diarrhea", "confidence": 0.7}'
parsed4, was_repaired4 = _extract_json_block_with_repair(complete)
assert parsed4 and not was_repaired4, "a complete object should not be flagged as repaired"
print("PASS: complete JSON is not flagged as repaired")

print("\n== Denied-symptom filter ==")
kept, dropped = _filter_denied_symptoms(
    ["cough", "fever", "rash"], "Patient has fever and rash. No cough reported."
)
assert dropped == ["cough"] and kept == ["fever", "rash"], (kept, dropped)
print("PASS: dropped explicitly denied symptom ->", "kept:", kept, "dropped:", dropped)

kept2, dropped2 = _filter_denied_symptoms(["fever", "rash"], "Patient has fever and rash.")
assert dropped2 == [] and kept2 == ["fever", "rash"]
print("PASS: no false positives when nothing is denied ->", kept2)

print("\n== Storage roundtrip ==")
record = EncounterRecord(**ok.model_dump(), raw_narrative="test narrative")
rid = storage.save_encounter(record)
rows = storage.list_encounters()
assert rows and rows[0]["id"] == rid
assert rows[0]["syndrome_category"] == "Acute Rash with Fever"
print(f"PASS: saved and retrieved encounter #{rid} ->", rows[0]["syndrome_category"])

csv_path = storage.export_csv()
assert os.path.exists(csv_path)
print("PASS: CSV export ->", csv_path)

print("\nALL SMOKE TESTS PASSED")
