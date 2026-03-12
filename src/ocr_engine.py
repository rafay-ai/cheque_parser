"""
ocr_engine.py
Wraps PaddleOCR: image preprocessing + running OCR + returning
a clean list of detection dicts.
"""

import os

import cv2
import numpy as np
from PIL import ExifTags, Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# ─── Image preprocessing ──────────────────────────────────────────────────────


def fix_exif_rotation(image_path: str) -> str:
    """Correct camera rotation from EXIF metadata. Returns corrected path."""
    try:
        pil = Image.open(image_path)
        exif = pil._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if value == 3:
                        pil = pil.rotate(180, expand=True)
                    elif value == 6:
                        pil = pil.rotate(270, expand=True)
                    elif value == 8:
                        pil = pil.rotate(90, expand=True)
                    break
        out = "exif_corrected.jpg"
        pil.save(out)
        print("[EXIF] rotation corrected")
        return out
    except Exception as e:
        print(f"[EXIF] no correction needed ({e})")
        return image_path


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Resize to at least 1000px wide, denoise, CLAHE, deskew.
    Returns BGR numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Upscale if too small
    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Deskew
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray_inv > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) <= 10:
            hh, ww = img.shape[:2]
            M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1.0)
            img = cv2.warpAffine(
                img, M, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    return img


# ─── OCR engine ───────────────────────────────────────────────────────────────


class OCREngine:
    """
    Thin wrapper around PaddleOCR.
    Lazy-loads PaddleOCR on first call to avoid slow import at startup.

    Usage:
        engine = OCREngine(use_gpu=False)
        detections = engine.run("processed.jpg")
    """

    def __init__(self, use_gpu: bool = False, lang: str = "en"):
        self._ocr = None
        self._use_gpu = use_gpu
        self._lang = lang

    def _load(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=False,
                show_log=False,
            )
            print("[OCREngine] PaddleOCR loaded (CPU mode)")

    def run(self, image_path: str) -> list[dict]:
        """
        Run OCR on image_path.
        Returns list of dicts sorted top→bottom, left→right:
            {"text", "confidence", "bbox", "top_y", "left_x"}
        """
        self._load()
        result = self._ocr.ocr(image_path, cls=True)
        detections = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                top_y = min(pt[1] for pt in bbox)
                left_x = min(pt[0] for pt in bbox)
                detections.append(
                    {
                        "text": text.strip(),
                        "confidence": round(float(conf), 4),
                        "bbox": bbox,
                        "top_y": int(top_y),
                        "left_x": int(left_x),
                    }
                )
        detections.sort(key=lambda d: (round(d["top_y"] / 20) * 20, d["left_x"]))
        print(f"[OCREngine] {len(detections)} regions detected")
        return detections
