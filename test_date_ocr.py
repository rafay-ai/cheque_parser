import sys
from src.ocr_engine import OCREngine
from src.extractor import _remove_grid_lines, _DATE_CHAR_MAP
import cv2, re, json, numpy as np

ocr = OCREngine()

# Load date crop from the debug test
date_crop = cv2.imread('debug_test/date_crop.jpg')
if date_crop is None:
    print("ERROR: Could not read date_crop.jpg")
    sys.exit(1)

print(f"Date crop shape: {date_crop.shape}")

results = {}

# Strategy 1: RemoveLines+Otsu
scaled = cv2.resize(date_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
clean = _remove_grid_lines(gray)
_, out = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
proc = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
cv2.imwrite("_test_removelines_otsu.jpg", proc)
crop_dets = ocr.run("_test_removelines_otsu.jpg")
results['RemoveLines+Otsu'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets]

# Strategy 2: RemoveLines+Adaptive
clean2 = _remove_grid_lines(gray)
out2 = cv2.adaptiveThreshold(clean2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
proc2 = cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)
cv2.imwrite("_test_removelines_adaptive.jpg", proc2)
crop_dets2 = ocr.run("_test_removelines_adaptive.jpg")
results['RemoveLines+Adaptive'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets2]

# Strategy 3: CLAHE + Sharpen
scaled3 = cv2.resize(date_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
gray3 = cv2.cvtColor(scaled3, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
enhanced = clahe.apply(gray3)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(enhanced, -1, kernel)
proc3 = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
cv2.imwrite("_test_clahe_sharpen.jpg", proc3)
crop_dets3 = ocr.run("_test_clahe_sharpen.jpg")
results['CLAHE+Sharpen'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets3]

# Strategy 4: 3x plain
scaled4 = cv2.resize(date_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("_test_3x_plain.jpg", scaled4)
crop_dets4 = ocr.run("_test_3x_plain.jpg")
results['3x+Plain'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets4]

# Strategy 5: Just the clean (no gridlines) image directly
cv2.imwrite("_test_clean_only.jpg", cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))
crop_dets5 = ocr.run("_test_clean_only.jpg")
results['CleanOnly'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets5]

# Strategy 6: 2x scale (might work better for some OCR)
scaled6 = cv2.resize(date_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("_test_2x_plain.jpg", scaled6)
crop_dets6 = ocr.run("_test_2x_plain.jpg")
results['2x+Plain'] = [{'text': d['text'], 'conf': round(d['confidence'],2)} for d in crop_dets6]

# Collect digit strings
for name, dets in results.items():
    all_digits = ""
    for d in dets:
        all_digits += re.sub(r'\D', '', d['text'].translate(_DATE_CHAR_MAP))
    results[name] = {'dets': dets, 'digits': all_digits}
    print(f"{name}: digits='{all_digits}' dets={[d['text'] for d in dets]}")

with open('test_date_log.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nDone. Check _test_*.jpg images for visual inspection.")
