'''
this code only validtes the format of the information retirive from the pan card
'''

from __future__ import __annotations__

import re
from datetime import datetime
from typing import TypedDict

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# labels that should never be mistaken for a persons name 
INVALID_NAME_LABELS = {
    "INCOME TAX DEPARTMENT",
    "GOVT OF INDIA",
    "GOVERNMENT OF INDIA",
    "PERMANENT ACCOUNT NUMBER",
    "PAN CARD",
    "INDIA",
}

_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")

class ValidationResult(TypedDict):
    pan_valid: bool
    dob_valid: bool
    name_valid: bool
    overall_valid: bool

def validate_pan(pan_number: str | None) -> bool:

    if not pan_number:
        return False
    return bool(PAN_REGEX.match(pan_number))

def validate_dob(date_of_birth: str | None) -> bool:
    """
    Validate that date_of_birth is a real calendar date in one of the
    supported formats (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY) and falls in a
    plausible range for a living/recently-living PAN applicant.
    """
    if not date_of_birth:
        return False

    parsed = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(date_of_birth, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        return False

    current_year = datetime.now().year
    if not (1900 <= parsed.year <= current_year):
        return False
    if parsed > datetime.now():
        return False

    return True

def validate_name(name: str | None) -> bool:
    """
    Validate that name is non-empty, contains alphabetic characters, and is
    not one of the well-known boilerplate labels printed on every PAN card.
    """
    if not name:
        return False

    cleaned = name.strip().upper()
    if not cleaned:
        return False

    if cleaned in INVALID_NAME_LABELS:
        return False
    if any(label in cleaned for label in INVALID_NAME_LABELS):
        return False

    letters = re.sub(r"[^A-Z]", "", cleaned)
    if len(letters) < 3:
        return False

    # Require the name to be mostly alphabetic characters/spaces (rejects
    # lines that are mostly digits or symbols slipping through as "names").
    non_space = cleaned.replace(" ", "")
    if len(letters) / max(len(non_space), 1) < 0.8:
        return False

    return True

def validate_fields(
    pan_number: str | None,
    date_of_birth: str | None,
    name: str | None,
) -> ValidationResult:
    """
    Run all three validators and compute an overall_valid flag (True only
    if every individual field is valid).
    """
    pan_valid = validate_pan(pan_number)
    dob_valid = validate_dob(date_of_birth)
    name_valid = validate_name(name)

    return {
        "pan_valid": pan_valid,
        "dob_valid": dob_valid,
        "name_valid": name_valid,
        "overall_valid": pan_valid and dob_valid and name_valid,
    }