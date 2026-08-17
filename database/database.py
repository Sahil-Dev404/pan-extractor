"""
sql lite is used to store the extracted PAN card records
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

DEFAULT_DB_PATH = Path("pan_data.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pan_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    pan_number      TEXT,
    date_of_birth   TEXT,
    pan_valid       INTEGER NOT NULL,
    dob_valid       INTEGER NOT NULL,
    name_valid      INTEGER NOT NULL,
    overall_valid   INTEGER NOT NULL,
    ocr_confidence  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

class PanRecord(TypedDict):
    id: int
    name: str | None
    pan_number: str | None
    date_of_birth: str | None
    pan_valid: bool
    dob_valid: bool
    name_valid: bool
    overall_valid: bool
    ocr_confidence: float | None
    created_at: str

def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Create the pan_records table if it doesn't already exist.
    Safe to call every run -- CREATE TABLE IF NOT EXISTS is idempotent.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()

def insert_record(
    name: str | None,
    pan_number: str | None,
    date_of_birth: str | None,
    pan_valid: bool,
    dob_valid: bool,
    name_valid: bool,
    overall_valid: bool,
    ocr_confidence: float | None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    Insert one extraction result and return the new row's id.

    All values are passed as bound parameters (never string-formatted into
    the SQL) to guard against injection.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO pan_records (
                name, pan_number, date_of_birth,
                pan_valid, dob_valid, name_valid, overall_valid,
                ocr_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                pan_number,
                date_of_birth,
                int(pan_valid),
                int(dob_valid),
                int(name_valid),
                int(overall_valid),
                ocr_confidence,
            ),
        )
        conn.commit()
        
        return int(cursor.lastrowid)

def _row_to_record(row: tuple[Any, ...]) -> PanRecord:
    (
        row_id,
        name,
        pan_number,
        date_of_birth,
        pan_valid,
        dob_valid,
        name_valid,
        overall_valid,
        ocr_confidence,
        created_at,
    ) = row
    return {
        "id": row_id,
        "name": name,
        "pan_number": pan_number,
        "date_of_birth": date_of_birth,
        "pan_valid": bool(pan_valid),
        "dob_valid": bool(dob_valid),
        "name_valid": bool(name_valid),
        "overall_valid": bool(overall_valid),
        "ocr_confidence": ocr_confidence,
        "created_at": created_at,
    }

def get_records(db_path: str | Path = DEFAULT_DB_PATH) -> list[PanRecord]:
    """
    Fetch all stored records, most recent first.
    Returns an empty list if the table has no rows (or doesn't exist yet --
    callers should call initialize_database() first in that case).
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, name, pan_number, date_of_birth,
                   pan_valid, dob_valid, name_valid, overall_valid,
                   ocr_confidence, created_at
            FROM pan_records
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()

    return [_row_to_record(row) for row in rows]



def get_record_by_id(
    record_id: int, db_path: str | Path = DEFAULT_DB_PATH
) -> PanRecord | None:
    """Fetch a single record by id, or None if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, name, pan_number, date_of_birth,
                   pan_valid, dob_valid, name_valid, overall_valid,
                   ocr_confidence, created_at
            FROM pan_records
            WHERE id = ?
            """,
            (record_id,),
        )
        row = cursor.fetchone()

    return _row_to_record(row) if row else None