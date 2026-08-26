"""
Local, single-file SQLite storage for encounter records.

Deliberately not a managed database: zero configuration, fully offline,
and the whole store is one portable file (config.DB_PATH) that can be
copied off a facility laptop or exported to CSV for district reporting.
"""
import csv
import json
import sqlite3
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
    raw_narrative TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def save_encounter(record: EncounterRecord) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO encounters
               (timestamp, syndrome_category, symptoms, onset_days, severity,
                age_group, sex, icd10_codes, reportable, confidence, summary,
                language_detected, raw_narrative)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        return cur.lastrowid


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["symptoms"] = json.loads(d["symptoms"] or "[]")
    d["icd10_codes"] = json.loads(d["icd10_codes"] or "[]")
    d["reportable"] = bool(d["reportable"])
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


def export_csv(path: Optional[str] = None) -> str:
    init_db()
    out_path = Path(path) if path else config.DATA_DIR / "episcribe_export.csv"
    rows = list_encounters(limit=100000)
    fieldnames = [
        "id", "timestamp", "syndrome_category", "symptoms", "onset_days",
        "severity", "age_group", "sex", "icd10_codes", "reportable",
        "confidence", "summary", "language_detected", "raw_narrative",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["symptoms"] = "; ".join(row["symptoms"])
            row["icd10_codes"] = "; ".join(row["icd10_codes"])
            writer.writerow(row)
    return str(out_path)
