'''
the output from the ocr engine file - it is not explicitly for the points 
so this file will extract the Pan no. , dob , Name
'''

from __future__ import annotations

import re
from datetime import datetime
from typing import TypedDict

# regex patterns
PAN_REGEX = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

DATE_REGEX = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")

BOILERPLATE_PATTERNS = [
    r"INCOME\s*TAX\s*DEPART?MENT",
    r"GOVT\.?\s*OF\s*INDIA",
    r"GOVERNMENT\s*OF\s*INDIA",
    r"PERMANENT\s*ACCOUNT\s*NUMBER",
    r"\bPAN\b\s*CARD",
    r"^PAN$",
    r"SIGNATURE",
    r"DATE\s*OF\s*BIRTH",
    r"FATHER'?S?\s*NAME",
    r"INCOME\s*TAX",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS))

class FieldConfidence(TypedDict):
    name: float
    pan_number: float
    date_of_birth: float

class ExtractedFields(TypedDict):
    name: str | None
    pan_number: str | None
    date_of_birth: str | None
    field_confidence: FieldConfidence

def normalize_pan_candidate(text: str) -> str:
    # the pan text into uppercase and with no whitespace
    return re.sub(r"\s+","", text).upper()

def _looks_like_boilerplate(line: str) -> bool:
    normalized = re.sub(r"[^A-Z\S'.]", "", line.upper())
    return bool(_BOILERPLATE_RE.search(normalized))

def extract_pan_number(ocr_results: list[dict]) -> tuple[str | None, float]:
    candidates: list[tuple[str, float]] = []

    for item in ocr_results:
        normalized = normalize_pan_candidate(item["text"])
        match = PAN_REGEX.search(normalized)
        if match:
            candidates.append((match.group(0), item["confidence"]))

    if not candidates:
        return None, 0.0

    #prefer the highest confidence candidate
    best_pan, best_conf = max(candidates, key=lambda c: c[1])
    return best_pan, best_conf


def extract_date_of_birth(ocr_results: list[dict]) -> tuple[str | None, float]:
    current_year = datetime.now().year
    candidates: list[tuple[str, float]] = []

    for item in ocr_results:
        for match in DATE_REGEX.finditer(item["text"]):
            day, month, year = match.groups()
            year_int = int(year)
            if 1900 <= year_int <= current_year:
                normalized = f"{int(day):02d}/{int(month):02d}/{year_int}"
                candidates.append((normalized, item["confidence"]))

    if not candidates:
        return None, 0.0

    best_date, best_conf = max(candidates, key=lambda c: c[1])
    return best_date, best_conf

def extract_name(ocr_results: list[dict], pan_number: str | None ,date_of_birth: str | None,) -> tuple[str | None, float]:
    for item in ocr_results:
        text = item["text"].strip()
        if not text:
            continue

        if _looks_like_boilerplate(text):
            continue

        normalized_pan_check = normalize_pan_candidate(text)
        if pan_number and pan_number in normalized_pan_check:
            continue

        if date_of_birth and DATE_REGEX.search(text):
            continue

        letters = re.sub(r"[^A-Za-z]", "", text)
        if len(letters) < 3:
            continue
        
        if len(letters) / max(len(text.replace(" ", "")), 1) < 0.7:
            continue

        heuristic_confidence = item["confidence"] * 0.85
        return text.upper(), heuristic_confidence

    return None, 0.0

def extract_fields(ocr_results: list[dict]) -> ExtractedFields:
    # assemble everything'

    pan_number, pan_conf = extract_pan_number(ocr_results) 
    date_of_birth, dob_conf = extract_date_of_birth(ocr_results)
    name, name_conf = extract_name(ocr_results, pan_number, date_of_birth)

    return {
        "name": name,
        "pan_number": pan_number,
        "date_of_birth": date_of_birth,
        "field_confidence": {
            "name": round(name_conf,4),
            "pan_number": round(pan_conf, 4),
            "date_of_birth": round(dob_conf, 4),
        },
    }