'''
main pipeline
Pipeline:
    1. Load image
    2. Preprocess image (grayscale, denoise, contrast, sharpen)
    3. Attempt perspective correction (falls back to preprocessed image if
       no reliable card contour is found)
    4. Run OCR (PaddleOCR)
    5. Print raw OCR results
    6. Extract fields (PAN number, name, DOB)
    7. Validate fields
    8. Print final structured result
    9. Store result in SQLite
    10. Print database record ID
'''

from __future__ import annotations

import sys
from pathlib import Path

from database.database import initialize_database, insert_record
from extraction.field_extractor import extract_fields
from ocr.ocr_engine import extract_text
from preprocessing.image_processor import (
    ImageLoadError,
    correct_perspective,
    preprocess_image,
)
from validation.validator import validate_fields

LOW_CONFIDENCE_THRESHOLD = 0.5

def _average_confidence(ocr_results: list[dict]) ->float:
    if not ocr_results:
        return 0.0
    return sum(item["confidence"] for item in ocr_results) /len(ocr_results)

def run_pipeline(image_path: str) -> int | None:
    """
    Execute the full extraction pipeline for a single image.

    Returns:
        The new database record id on success, or None if the pipeline
        could not produce a storable result (all errors are reported to
        stdout/stderr before returning None; nothing raises out of here).
    """
    # --- Step 1 & 2: load + preprocess -----------------------------------
    try:
        processed_image, processed_path = preprocess_image(image_path)
    except ImageLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: unexpected failure while preprocessing image: {exc}", file=sys.stderr)
        return None

    print(f"Preprocessed image saved to: {processed_path}")

    # --- Step 3: optional perspective correction --------------------------
    corrected_image, was_corrected = correct_perspective(processed_image)
    if was_corrected:
        print("Card contour detected -- applied perspective correction.")
    else:
        print("No reliable card contour found -- using preprocessed image as-is.")

    # --- Step 4 & 5: OCR ----------------------------------------------------
    try:
        ocr_results = extract_text(corrected_image)
    except RuntimeError as exc:
        print(f"ERROR: OCR failed: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: unexpected failure during OCR: {exc}", file=sys.stderr)
        return None

    if not ocr_results:
        print(
            "ERROR: OCR did not detect any text in this image. "
            "Check that the image actually contains a readable PAN card.",
            file=sys.stderr,
        )
        return None

    print("\n----------------------------------------")
    print("RAW OCR RESULTS")
    print("----------------------------------------")
    for item in ocr_results:
        print(f"  '{item['text']}'  (confidence: {item['confidence']:.2f})")

    avg_conf = _average_confidence(ocr_results)
    if avg_conf < LOW_CONFIDENCE_THRESHOLD:
        print(
            f"\nWARNING: average OCR confidence is low ({avg_conf:.2f}). "
            "Extracted fields below may be unreliable.",
            file=sys.stderr,
        )

    # --- Step 6: field extraction -------------------------------------------
    fields = extract_fields(ocr_results)

    if fields["pan_number"] is None:
        print(
            "WARNING: no PAN-number-shaped string was found in the OCR text.",
            file=sys.stderr,
        )
    if fields["date_of_birth"] is None:
        print("WARNING: no date of birth was found in the OCR text.", file=sys.stderr)
    if fields["name"] is None:
        print("WARNING: no name could be heuristically identified.", file=sys.stderr)

    # --- Step 7: validation ---------------------------------------------------
    validation = validate_fields(
        fields["pan_number"], fields["date_of_birth"], fields["name"]
    )

    # --- Step 8: print final structured result --------------------------------
    print("\n========================================")
    print("PAN CARD EXTRACTION")
    print("========================================\n")
    print(f"Name           : {fields['name'] or 'NOT DETECTED'}")
    print(f"PAN Number     : {fields['pan_number'] or 'NOT DETECTED'}")
    print(f"Date of Birth  : {fields['date_of_birth'] or 'NOT DETECTED'}")

    print("\n----------------------------------------")
    print("VALIDATION")
    print("----------------------------------------\n")
    print(f"PAN Valid      : {validation['pan_valid']}")
    print(f"DOB Valid      : {validation['dob_valid']}")
    print(f"Name Valid     : {validation['name_valid']}")
    print(f"Overall Valid  : {validation['overall_valid']}")

    print("\n----------------------------------------")
    print("CONFIDENCE")
    print("----------------------------------------\n")
    print(f"PAN            : {fields['field_confidence']['pan_number']:.2f}")
    print(f"DOB            : {fields['field_confidence']['date_of_birth']:.2f}")
    print(f"Name           : {fields['field_confidence']['name']:.2f}")

    # --- Step 9: store in SQLite -----------------------------------------------
    try:
        initialize_database()
        record_id = insert_record(
            name=fields["name"],
            pan_number=fields["pan_number"],
            date_of_birth=fields["date_of_birth"],
            pan_valid=validation["pan_valid"],
            dob_valid=validation["dob_valid"],
            name_valid=validation["name_valid"],
            overall_valid=validation["overall_valid"],
            ocr_confidence=round(avg_conf, 4),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: failed to store record in database: {exc}", file=sys.stderr)
        return None

    # --- Step 10: print database record ID --------------------------------------
    print(f"\nRecord stored successfully.")
    print(f"Database ID: {record_id}")

    return record_id

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/pan_image.jpg", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"ERROR: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    record_id = run_pipeline(image_path)
    sys.exit(0 if record_id is not None else 1)


if __name__ == "__main__":
    main()