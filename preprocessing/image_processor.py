
"""
input = image
and we want to preprocess it:
- resize
- grayscle
- denoise
- contrast enhancement
- sharpen
- threshold
"""

from __future__ import annotations

from pathlib import Path

import cv2 #comp-vision
import numpy as np

TARGET_WIDTH = 1000

class ImageLoadError(Exception):
    '''used when the image is not loaded'''

def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f'Image path does not exist: {path}')
    if not path.is_file():
        raise ImageLoadError(f'image path is not a file: {path}')

    image = cv2.imread(str(path))
    if image is None:
        raise ImageLoadError(
            f"could not decode image: {path}"
        )
    return image

def resize_if_needed(image: np.ndarray, target_width: int = TARGET_WIDTH) -> np.ndarray:
    '''
    resize the image so the image size is equla to the target width - will be skip if already at the size of the target width
    '''
    height, width = image.shape[:2]
    if width == 0:
        return image

    #if nearly same then do nothing eat 5 star
    if abs(width - target_width) / target_width < 0.05:
        return image

    scale = target_width / width
    new_size = (target_width, max(1, int(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interpolation)

def to_grayscale(image: np.ndarray) -> np.ndarray:
    '''convert a bgr image to single channel grayscale'''

    if len(image.shape) ==2:
        return image # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(gray: np.ndarray) -> np.ndarray:
    '''
    reduce sensor/jpeg noise while trying to keep text edges sharp-
    fastNlMeansDenoising works well for photo taken with phone cameras
    '''
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened

def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    # it handles the uneven lightening
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

def preprocess_image(image_path: str | Path, save_dir: str | Path ="data/processed", apply_threshold: bool = False,) -> tuple[np.ndarray, Path]:
    '''main working function'''

    image = load_image(image_path)
    image = resize_if_needed(image)

    gray = to_grayscale(image)
    gray = denoise(gray)
    gray = enhance_contrast(gray)
    gray = sharpen(gray)

    processed = adaptive_threshold(gray) if apply_threshold else gray

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    input_name = Path(image_path).stem
    suffix = "_threshold" if apply_threshold else "_processed"
    saved_path = save_dir_path / f"{input_name}{suffix}.png"
    cv2.imwrite(str(saved_path), processed)

    return processed, saved_path


# ---------------------------------------------------------------------------
# Step 2: optional card-region detection / perspective correction.
# Kept isolated from preprocess_image() on purpose (see module docstring).
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    return rect


def find_card_contour(image: np.ndarray) -> np.ndarray | None:
    """
    Try to find a 4-point rectangular contour that plausibly represents the
    card's edges. Returns None if no reliable candidate is found -- this is
    a normal, expected outcome (e.g. the image is already a tight crop of
    the card), not an error.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    best_candidate = None
    best_area = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # A card should approximate to a quadrilateral.
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        # Reject contours that are implausibly small (noise) or that cover
        # almost the whole frame (likely the photo border, not the card).
        if area < 0.2 * image_area or area > 0.98 * image_area:
            continue

        if not cv2.isContourConvex(approx):
            continue

        if area > best_area:
            best_area = area
            best_candidate = approx

    if best_candidate is None:
        return None

    return best_candidate.reshape(4, 2).astype("float32")


def correct_perspective(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Attempt to detect the PAN card's rectangular boundary and warp it to a
    flat, top-down view.

    Returns:
        (image, corrected) where `corrected` is False and `image` is the
        original (unmodified) input whenever a reliable contour could not
        be found -- callers should treat this as a normal fallback path,
        never as an error.
    """
    try:
        pts = find_card_contour(image)
        if pts is None:
            return image, False

        rect = _order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        # Guard against degenerate warps (near-zero width/height).
        if max_width < 50 or max_height < 50:
            return image, False

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped, True

    except cv2.error:
        # Any OpenCV-level failure during detection/warping falls back to
        # the original image rather than crashing the pipeline.
        return image, False

