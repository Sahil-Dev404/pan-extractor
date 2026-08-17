
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
    clahe = cv2.createCLAHE(clipLimit=2.0, titleGridSize=(8,8))
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


