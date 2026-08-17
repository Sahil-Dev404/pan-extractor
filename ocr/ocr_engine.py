'''
using paddleOCR -It extracts text, tables, and structures from images and documents quickly and accurately across dozens of languages
'''

from __future__ import annotations

from typing import Any
import numpy as np

OCRResult = dict[str, Any]

_engine_instance = None

def _get_engine(): # will run only one time

    global _engine_instance
    if _engine_instance is None:
        from paddleocr import PaddleOCR

        try:
            _engine_instance = PaddleOCR(
                use_textline_orientation=True,
                lang="en",
            )
        except TypeError:
            _engine_instance = PaddleOCR(use_angle_cls=True, lang="en")
    return _engine_instance

def _bbox_from_poly(poly: Any) ->list[list[float]]:
    '''normalise a polygon to a plain list of [x,y] pairs'''
    return [[float(x), float(y)] for x, y in poly]

def _parse_predict_result(raw_result: list[Any]) -> list[OCRResult]:

    results: list[OCRResult] = []
    for page in raw_result:
        texts = page.get("rec_texts",[]) #detct text 
        scores = page.get("rec_scores", []) # detct conf score
        polys = page.get("rec_polys", page.get("dt_polys",[])) #detect bounding boxes

        for text, score, poly in zip(texts, scores, polys):
            text = str(text).strip()
            if not text:
                continue
            results.append(
                {
                    "text":text,
                    "bbox": _bbox_from_poly(poly),
                    "confidence": float(score),
                }
            )
    return results

def _parse_legacy_result(raw_result: list[Any]) -> list[OCRResult]:

    results: list[OCRResult] = []
    for page in raw_result:
        if not page:
            continue
        for line in page:
            bbox, (text, confidence) = line
            text = str(text).strip()
            if not text:
                continue
            results.append(
                {
                    "text":text,
                    "bbox": _bbox_from_poly(bbox),
                    "confidence": float(confidence),
                }
            )
    return results

# the mian OCR function that detects text with its location and confidence
def extract_text(image: np.ndarray) -> list[OCRResult]:
    '''
    input = image
    returns:
        list of dict text,bbox,confidence
    '''
    engine = _get_engine()

    if image.ndim ==2:
        import cv2

        image_for_ocr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_for_ocr = image

    try:
        if hasattr(engine, "predict"):
            raw_result = engine.predict(image_for_ocr)
            parsed = _parse_predict_result(raw_result)
        else:
            raw_result = engine.ocr(image_for_ocr)
            parsed = _parse_legacy_result(raw_result)
    except Exception as exc:
        raise RuntimeError(f'paddleocr failed to process the image: {exc}') from exc

    def sort_key(item: OCRResult) -> tuple[float, float]:
        xs = [p[0] for p in item["bbox"]]
        ys = [p[1] for p in item["bbox"]]
        return (min(ys), min(xs)) if ys and xs else (0.0,0.0)

    parsed.sort(key=sort_key)
    return parsed