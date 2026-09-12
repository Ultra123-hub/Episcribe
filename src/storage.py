"""
Local, single-file SQLite storage for encounter records.

Deliberately not a managed database: zero configuration, fully offline,
and the whole store is one portable file (config.DB_PATH) that can be
copied off a facility laptop or exported to CSV for district reporting.
"""
import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import config
from src.schema import EncounterRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS encounters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    syndrome_category TEXT NOT NULL,
    symptoms TEXT,
    onset_days INTEGER,
    severity TEXT,
    age_group TEXT,
    sex TEXT,
    icd10_codes TEXT,
    reportable INTEGER,
    confidence REAL,
    summary TEXT,
    language_detected TEXT,
    raw_narrative TEXT,
    public_health_category TEXT,
    audio_hash TEXT
);
"""

# Columns added after the initial release — applied via ALTER TABLE against
# any pre-existing database file rather than losing already-saved records.
_MIGRATIONS = [
    ("public_health_category", "TEXT"),
    ("audio_hash", "TEXT"),
    ("soap_subjective", "TEXT"),
    ("soap_objective", "TEXT"),
    ("soap_assessment", "TEXT"),
    ("soap_plan", "TEXT"),
    ("hallucination_flags", "TEXT"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(encounters)")}
        for col_name, col_type in _MIGRATIONS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE encounters ADD COLUMN {col_name} {col_type}")


def save_encounter(record: EncounterRecord) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO encounters
               (timestamp, syndrome_category, symptoms, onset_days, severity,
                age_group, sex, icd10_codes, reportable, confidence, summary,
                language_detected, raw_narrative, public_health_category,
                audio_hash, soap_subjective, soap_objective, soap_assessment,
                soap_plan, hallucination_flags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.timestamp,
                record.syndrome_category,
                json.dumps(record.symptoms),
                record.onset_days,
                record.severity,
                record.age_group,
                record.sex,
                json.dumps(record.icd10_codes),
                int(record.reportable),
                record.confidence,
                record.summary,
                record.language_detected,
                record.raw_narrative,
                json.dumps(record.public_health_category),
                record.audio_hash,
                record.soap_subjective,
                record.soap_objective,
                record.soap_assessment,
                record.soap_plan,
                json.dumps(record.hallucination_flags),
            ),
        )
        return cur.lastrowid


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["symptoms"] = json.loads(d["symptoms"] or "[]")
    d["icd10_codes"] = json.loads(d["icd10_codes"] or "[]")
    d["public_health_category"] = json.loads(d.get("public_health_category") or "[]")
    d["reportable"] = bool(d["reportable"])
    d["soap_subjective"] = d.get("soap_subjective") or ""
    d["soap_objective"] = d.get("soap_objective") or ""
    d["soap_assessment"] = d.get("soap_assessment") or ""
    d["soap_plan"] = d.get("soap_plan") or ""
    d["hallucination_flags"] = json.loads(d.get("hallucination_flags") or "[]")
    return d


def list_encounters(limit: int = 200, keyword: Optional[str] = None) -> List[dict]:
    init_db()
    with _connect() as conn:
        if keyword:
            like = f"%{keyword.lower()}%"
            rows = conn.execute(
                """SELECT * FROM encounters
                   WHERE lower(syndrome_category) LIKE ?
                      OR lower(summary) LIKE ?
                      OR lower(raw_narrative) LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (like, like, like, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM encounters ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def recent_cluster_count(syndrome_category: str, days: int = 7) -> int:
    """Count of encounters with the same syndrome_category logged in the
    last `days` days (inclusive of the one just saved) — a lightweight,
    local-only outbreak/cluster signal. Not epidemiological surveillance
    in any rigorous sense (no denominator, no population data, no spatial
    clustering) — just a same-facility 'this is happening a lot lately'
    flag to prompt a human to look closer, which is honest about what a
    single offline SQLite file can actually tell you."""
    init_db()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS c FROM encounters
               WHERE syndrome_category = ? AND timestamp >= ?""",
            (syndrome_category, cutoff),
        ).fetchone()
    return row["c"] if row else 0


def export_csv(path: Optional[str] = None) -> str:
    init_db()
    out_path = Path(path) if path else config.DATA_DIR / "episcribe_export.csv"
    rows = list_encounters(limit=100000)
    fieldnames = [
        "id", "timestamp", "syndrome_category", "symptoms", "onset_days",
        "severity", "age_group", "sex", "icd10_codes", "reportable",
        "confidence", "summary", "language_detected", "raw_narrative",
        "public_health_category", "audio_hash", "soap_subjective",
        "soap_objective", "soap_assessment", "soap_plan", "hallucination_flags",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["symptoms"] = "; ".join(row["symptoms"])
            row["icd10_codes"] = "; ".join(row["icd10_codes"])
            row["public_health_category"] = "; ".join(row["public_health_category"])
            row["hallucination_flags"] = "; ".join(row["hallucination_flags"])
            writer.writerow(row)
    return str(out_path)
