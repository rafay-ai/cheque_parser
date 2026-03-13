"""
extractor.py
Pure extraction logic — no ML models, no PaddleOCR imports here.
"""

import re

import cv2
import numpy as np

# ─── Spatial helpers ─────────────────────────────────────────────────────────


def _pts(d):
    return np.array(d["bbox"])


def bbox_y1(d):
    return _pts(d)[:, 1].min()


def bbox_y2(d):
    return _pts(d)[:, 1].max()


def bbox_x1(d):
    return _pts(d)[:, 0].min()


def bbox_x2(d):
    return _pts(d)[:, 0].max()


def bbox_center_y(d):
    return (_pts(d)[:, 1].min() + _pts(d)[:, 1].max()) / 2


def bbox_center_x(d):
    return (_pts(d)[:, 0].min() + _pts(d)[:, 0].max()) / 2


def bbox_height(d):
    return bbox_y2(d) - bbox_y1(d)


# ─── Region crop helpers ──────────────────────────────────────────────────────


def get_region_right_of_label(
    image, dets, label_pattern, pad_top=25, pad_bottom=35, right_bound=None
):
    """Crop the region to the right of a label detection.
    If right_bound is given, crop stops at that x-coordinate."""
    label_det = None
    for d in dets:
        if re.search(label_pattern, d["text"].strip(), re.IGNORECASE):
            label_det = d
            break
    if label_det is None:
        return None, None
    pts = _pts(label_det)
    y1 = int(pts[:, 1].min())
    y2 = int(pts[:, 1].max())
    x2 = int(pts[:, 0].max())
    h, w = image.shape[:2]
    x_end = right_bound if right_bound is not None else w
    crop = image[max(0, y1 - pad_top) : min(h, y2 + pad_bottom), x2:x_end]
    return (crop if crop.size > 0 else None), label_det


def _find_pkr_box_x(dets):
    """Find the left x-coordinate of the PKR amount box."""
    for d in dets:
        t = d["text"].strip()
        if re.match(r"^PKR$", t, re.IGNORECASE):
            return int(bbox_x1(d))
        # Sometimes PKR is merged with the amount, e.g. "PKR 34,351/-"
        if re.match(r"^PKR\s", t, re.IGNORECASE):
            return int(bbox_x1(d))
    return None


def _find_bearer_x(dets):
    """Find the left x-coordinate of 'or bearer' text."""
    for d in dets:
        t = d["text"].strip()
        if re.search(r"bearer", t, re.IGNORECASE):
            return int(bbox_x1(d))
        if re.search(r"^or\s*$", t, re.IGNORECASE):
            return int(bbox_x1(d))
    return None


def get_pay_region(image, dets):
    """Crop the Pay field — from right of 'Pay' label to before 'or bearer' or PKR box."""
    pay_det = None
    for d in dets:
        if re.search(r"^pa[ytv]$", d["text"].strip(), re.IGNORECASE):
            pay_det = d
            break
    if pay_det is None:
        return None, None

    pts = _pts(pay_det)
    y1 = int(pts[:, 1].min())
    y2 = int(pts[:, 1].max())
    x2 = int(pts[:, 0].max())
    h, w = image.shape[:2]

    # Find right boundary: stop at 'or bearer' or PKR box, whichever comes first
    right_bounds = [w]
    bearer_x = _find_bearer_x(dets)
    if bearer_x is not None and bearer_x > x2:
        right_bounds.append(bearer_x - 5)
    pkr_x = _find_pkr_box_x(dets)
    if pkr_x is not None and pkr_x > x2:
        right_bounds.append(pkr_x - 10)
    right_end = min(right_bounds)

    pad_top, pad_bottom = 20, 30
    crop = image[max(0, y1 - pad_top) : min(h, y2 + pad_bottom), x2:right_end]
    return (crop if crop.size > 0 else None), pay_det


def get_rupees_region(image, dets):
    """Crop the Rupees (amount in words) field.
    Includes continuation lines below the 'Rupees' label row,
    stops before the IBAN / account line."""
    rupees_det = None
    for d in dets:
        if re.search(r"^ru[pg]ees?", d["text"].strip(), re.IGNORECASE):
            rupees_det = d
            break
    if rupees_det is None:
        return None, None

    pts = _pts(rupees_det)
    label_y1 = int(pts[:, 1].min())
    label_y2 = int(pts[:, 1].max())
    label_x1 = int(pts[:, 0].min())  # left edge of "Rupees" label
    label_x2 = int(pts[:, 0].max())  # right edge of "Rupees" label
    row_height = label_y2 - label_y1
    h, w = image.shape[:2]

    # Find PKR box left edge as right boundary for the text area
    pkr_x = _find_pkr_box_x(dets)
    right_end = (pkr_x - 10) if (pkr_x is not None and pkr_x > label_x2) else w

    # Look for continuation lines (text that is below "Rupees" row but still part of amount in words)
    # Stop line markers: IBAN (PK...), account info, "Please do not", MICR digits at bottom
    stop_patterns = [
        r"PK\d{2}",
        r"HABB|HABIB|please",
        r"^\d{10,}",
        r"signature",
        r"MUHAMMAD|account",
    ]

    # Find the bottom extent: check detections up to 2 rows below
    bottom_y = label_y2 + 30  # minimum: include padded label row
    for d in dets:
        d_cy = bbox_center_y(d)
        d_y2_val = int(bbox_y2(d))
        # Must be below the label row
        if d_cy <= label_y2:
            continue
        # Must be within 2.5 row-heights below label
        if d_cy > label_y1 + row_height * 3.5:
            continue
        # Check if this is a stop-marker line
        t = d["text"].strip()
        is_stop = any(re.search(p, t, re.IGNORECASE) for p in stop_patterns)
        if is_stop:
            continue
        # This text is part of the continuation
        if d_y2_val + 25 > bottom_y:
            bottom_y = d_y2_val + 25

    pad_top = 15
    crop = image[max(0, label_y1 - pad_top) : min(h, bottom_y), label_x1:right_end]
    return (crop if crop.size > 0 else None), rupees_det


def get_pkr_region(image, dets):
    for d in dets:
        if re.match(r"PKR", d["text"].strip(), re.IGNORECASE):
            pts = _pts(d)
            y1 = int(pts[:, 1].min())
            y2 = int(pts[:, 1].max())
            x1 = int(pts[:, 0].min())
            h, w = image.shape[:2]
            crop = image[max(0, y1 - 15) : min(h, y2 + 20), max(0, x1 - 10) : w]
            return crop if crop.size > 0 else None
    return None


def get_signature_region(image, dets):
    sig_det = None
    for d in dets:
        if re.search(r"^signature", d["text"].strip(), re.IGNORECASE):
            sig_det = d
            break

    if sig_det is None:
        return None, None

    pts = _pts(sig_det)
    label_y1 = int(pts[:, 1].min())
    label_y2 = int(pts[:, 1].max())

    h, w = image.shape[:2]
    label_h = label_y2 - label_y1

    # Signature ink sits just above the label — tighten vertical range
    crop_y1 = max(0, label_y1 - label_h * 3)  # was 5, now 3
    crop_y2 = min(h, label_y2)
    crop_x1 = w // 2
    crop_x2 = w

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return (crop if crop.size > 0 else None), sig_det


# ─── Constants ───────────────────────────────────────────────────────────────

_BANK_PATTERNS = [
    r"HBL|Habib\s*Bank",
    r"UBL|United\s*Bank",
    r"MCB|Muslim\s*Commercial",
    r"Allied\s*Bank|ABL",
    r"Meezan\s*Bank",
    r"Bank\s*Alfalah",
    r"NBP|National\s*Bank",
    r"Standard\s*Chartered",
    r"Faysal\s*Bank",
    r"Bank\s*Al\s*Habib",
    r"Askari\s*Bank",
    r"BankIslami|Bank\s*Islami",
    r"Silk\s*Bank",
    r"Summit\s*Bank",
    r"JS\s*Bank",
    r"Habib\s*Metro",
    r"Soneri\s*Bank",
    r"Bank\s*of\s*Punjab|BOP",
    r"Bank\s*of\s*Khyber|BOK",
    r"Zarai\s*Taraqiati|ZTBL",
    r"First\s*Women\s*Bank",
    r"KASB\s*Bank",
    r"NIB\s*Bank",
    r"Citibank",
    r"Deutsche\s*Bank",
    r"HSBC",
]

_DATE_CHAR_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "Z": "2",
        "z": "2",
        "M": "9",
        "m": "9",
        "B": "8",
        "b": "6",
        "S": "5",
        "s": "5",
        "G": "6",
        "g": "9",
        "A": "4",
        "T": "7",
    }
)

_AMOUNT_CHAR_MAP = str.maketrans(
    {
        "I": "1",
        "i": "1",
        "|": "1",
        "O": "0",
        "o": "0",
        "C": "0",
        "c": "0",
        "D": "0",
        "Q": "0",
        "U": "0",
        "b": "0",
        "S": "5",
        "s": "5",
        "Z": "2",
        "z": "2",
        "G": "6",
        "L": "6",
        "l": "6",
        "B": "8",
        "g": "9",
        "Y": "1",
    }
)

_AMOUNT_BLACKLIST = [
    r"^\d{8}$",
    r"^\d{9,}$",
    r"\b(cheque|date|pay|branch|karachi|lahore|islamabad|account|islami|"
    r"islamic|limited|pakistan|clifton|shahrah|gulshan|please|bearer|"
    r"signature|main|pvt|ltd)\b",
]


# ─── Field extractors ────────────────────────────────────────────────────────


def extract_bank_name(dets):
    for d in dets:
        for pat in _BANK_PATTERNS:
            if re.search(pat, d["text"], re.IGNORECASE):
                return d["text"].strip()
    return None


def extract_date(dets):
    def _parse(t):
        m = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", t)
        if m:
            dd, mm, yy = m.group(1), m.group(2), m.group(3)
            yy = "20" + yy if len(yy) == 2 else yy
            if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                return f"{dd.zfill(2)}/{mm.zfill(2)}/{yy}"
        m = re.search(r"(?<!\d)(\d{8})(?!\d)", t)
        if m:
            r = m.group(1)
            dd, mm, yy = int(r[:2]), int(r[2:4]), int(r[4:])
            if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yy <= 2099:
                return f"{r[:2]}/{r[2:4]}/{r[4:]}"
        return None

    # Try each detection normally first
    for d in dets:
        t = d["text"].replace(" ", "")
        result = _parse(t) or _parse(t.translate(_DATE_CHAR_MAP))
        if result:
            return result

    # Fallback: some cheques (e.g. HBL) print the date as spaced individual digits
    # e.g. "0 5 1 1 5 2 0 2 5" — find detections near the "Date" label and concat digits
    date_det = None
    for d in dets:
        if re.search(r"^date$", d["text"].strip(), re.IGNORECASE):
            date_det = d
            break

    if date_det is not None:
        date_yc = bbox_center_y(date_det)
        date_x2 = bbox_x2(date_det)
        # Collect all detections on the same row to the right of "Date"
        row_tokens = []
        for d in dets:
            if d is date_det:
                continue
            if abs(bbox_center_y(d) - date_yc) < 40 and bbox_x1(d) > date_x2 - 10:
                row_tokens.append((bbox_x1(d), d["text"]))
        row_tokens.sort()
        combined = "".join(t for _, t in row_tokens).replace(" ", "")
        # Apply char map and try to parse
        combined = combined.translate(_DATE_CHAR_MAP)
        result = _parse(combined)
        if result:
            return result
        # Try extracting just digits
        digs = re.sub(r"\D", "", combined)
        result = _parse(digs)
        if result:
            return result

    return None


def extract_date_from_crop_ocr(date_crop, ocr_engine):
    if date_crop is None:
        return None

    def try_digits(s):
        m = re.search(r"(\d{8})", s)
        if m:
            r = m.group(1)
            dd, mm, yy = int(r[:2]), int(r[2:4]), int(r[4:])
            if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yy <= 2099:
                return f"{r[:2]}/{r[2:4]}/{r[4:]}"
        return None

    strategies = {
        "4x+Otsu": lambda img: cv2.cvtColor(
            cv2.threshold(
                cv2.cvtColor(
                    cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2GRAY,
                ),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1],
            cv2.COLOR_GRAY2BGR,
        ),
        "4x+Adaptive": lambda img: cv2.cvtColor(
            cv2.adaptiveThreshold(
                cv2.cvtColor(
                    cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2GRAY,
                ),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            ),
            cv2.COLOR_GRAY2BGR,
        ),
    }

    for name, transform_fn in strategies.items():
        try:
            proc = transform_fn(date_crop)
            cv2.imwrite("_date_crop_proc.jpg", proc)
            crop_dets = ocr_engine.run("_date_crop_proc.jpg")
            all_digits = ""
            for det in crop_dets:
                print(
                    f"  [date OCR {name}]: \"{det['text']}\" conf={det['confidence']:.2f}"
                )
                all_digits += re.sub(r"\D", "", det["text"].translate(_DATE_CHAR_MAP))
            result = try_digits(all_digits)
            if result:
                print(f"  [date found via {name}]: {result}")
                return result
        except Exception as e:
            print(f"  [date OCR {name} error]: {e}")
    return None


def extract_amount_ocr(dets, cheque_number=None):
    def _is_blacklisted(text):
        return any(re.search(p, text.strip(), re.IGNORECASE) for p in _AMOUNT_BLACKLIST)

    def _normalize(raw):
        # Strip common suffixes: /-  /=  /=-  =-  -  =
        raw = re.sub(r"[/\\][=\-]+$", "", raw)
        raw = re.sub(r"[=\-]+$", "", raw)
        raw = raw.split("/")[0].strip()
        raw = re.sub(r"(\d)-(\d)", r"\1\2", raw)
        raw = raw.translate(_AMOUNT_CHAR_MAP)
        raw = re.sub(r"[^\d,.]", "", raw)
        parts = raw.split(",")
        if len(parts) > 1 and len(parts[-1]) == 4:
            parts[-1] = parts[-1][:3]
        return ",".join(parts)

    best, best_val = None, 0

    # --- First pass: look for PKR-prefixed tokens ---
    for d in dets:
        t = d["text"].strip()
        if _is_blacklisted(t):
            continue

        m = re.match(r"PKR\s*([A-Z0-9,.\s\/\-=]+)$", t, re.IGNORECASE)
        if m:
            raw_pkr = m.group(1).strip()
            cleaned = _normalize(raw_pkr)
            ns = cleaned.replace(",", "")
            if ns:
                try:
                    val = float(ns)
                    if cheque_number and ns == re.sub(r"\D", "", str(cheque_number)):
                        continue
                    if 100 <= val <= 100_000_000 and val > best_val:
                        best_val = val
                        best = f"{int(val):,}" if val == int(val) else f"{val:,.2f}"
                        print(f"  [amount PKR token]: '{t}' → {best}")
                except ValueError:
                    pass
            continue

    if best is not None:
        return best

    # --- Second pass: look for standalone numeric tokens ---
    for d in dets:
        t = d["text"].strip()
        if _is_blacklisted(t):
            continue

        raw_letters = re.sub(r"[^A-Za-z]", "", t)
        misread_chars = set("COILSZGBgcoilszb|")
        if raw_letters and not all(c in misread_chars for c in raw_letters):
            continue
        t_clean = _normalize(t)
        if t_clean and len(t_clean.replace(",", "")) >= 3:
            try:
                val = float(t_clean.replace(",", ""))
                if cheque_number and t_clean.replace(",", "") == re.sub(
                    r"\D", "", str(cheque_number)
                ):
                    continue
                if 100 <= val <= 100_000_000 and val > best_val:
                    best_val = val
                    best = f"{int(val):,}" if val == int(val) else f"{val:,.2f}"
                    print(f"  [amount standalone]: '{t}' → {best}")
            except ValueError:
                pass
    return best


def extract_amount_from_crop_ocr(amount_crop, ocr_engine):
    if amount_crop is None:
        return None

    def _parse(text):
        # Strip PKR from beginning, and common cheque suffixes outside parsing
        t = re.sub(r"PKR\s*", "", text, flags=re.IGNORECASE)
        t = re.sub(r"[/\\][=\-]+$", "", t)
        t = re.sub(r"[=\-]+$", "", t)
        t = t.translate(_AMOUNT_CHAR_MAP)
        t = t.split("/")[0].strip()
        t = re.sub(r"[^\d,.]", "", t)
        parts = t.split(",")
        if len(parts) > 1 and len(parts[-1]) == 4:
            parts[-1] = parts[-1][:3]
        t = "".join(parts).replace(",", "")
        if not t:
            return None
        try:
            val = int(float(t))
            if 100 <= val <= 100_000_000:
                return f"{val:,}"
        except ValueError:
            pass
        return None

    strategies = {
        "4x+Otsu": lambda img: cv2.threshold(
            cv2.cvtColor(
                cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2GRAY,
            ),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1],
        "4x+Adaptive": lambda img: cv2.adaptiveThreshold(
            cv2.cvtColor(
                cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2GRAY,
            ),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            8,
        ),
        "3x+Sharpen": lambda img: cv2.filter2D(
            cv2.cvtColor(
                cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2GRAY,
            ),
            -1,
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        ),
        "2x+CLAHE": lambda img: cv2.createCLAHE(
            clipLimit=3.0, tileGridSize=(4, 4)
        ).apply(
            cv2.cvtColor(
                cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2GRAY,
            )
        ),
    }

    for name, transform_fn in strategies.items():
        try:
            processed = transform_fn(amount_crop)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("_amount_crop_proc.jpg", processed)
            crop_dets = ocr_engine.run("_amount_crop_proc.jpg")
            all_text = " ".join(d["text"] for d in crop_dets)
            print(f'  [amount crop OCR {name}]: "{all_text}"')

            # Try each detection individually (skip PKR label)
            for d in crop_dets:
                t = d["text"].strip()
                if re.match(r"^PKR$", t, re.IGNORECASE):
                    continue
                result = _parse(t)
                if result:
                    print(f"  [amount crop found via {name}]: {result}")
                    return result

            # Try concatenating all text (after removing PKR prefix)
            combined = re.sub(r"\bPKR\b", "", all_text, flags=re.IGNORECASE)
            combined = re.sub(r"\s+", "", combined)
            result = _parse(combined)
            if result:
                print(f"  [amount crop concat via {name}]: {result}")
                return result

            # Try extracting just digits from all detections
            all_digits = ""
            for d in crop_dets:
                t = d["text"].strip()
                if re.match(r"^PKR$", t, re.IGNORECASE):
                    continue
                t = re.sub(r"[/\\][=\-]+$", "", t)
                t = re.sub(r"[=\-]+$", "", t)
                t = t.translate(_AMOUNT_CHAR_MAP)
                all_digits += re.sub(r"[^\d]", "", t)
            if len(all_digits) >= 3:
                result = _parse(all_digits)
                if result:
                    print(f"  [amount crop digits via {name}]: {result}")
                    return result

        except Exception as e:
            print(f"  [amount crop OCR {name} error]: {e}")

    return None


def extract_cheque_number(dets, image_height=0):
    for d in dets:
        if re.search(r"cheque\s*no", d["text"], re.IGNORECASE):
            lx2 = bbox_x2(d)
            lyc = bbox_center_y(d)
            cands = []
            for o in dets:
                if o is d:
                    continue
                if abs(bbox_center_y(o) - lyc) < 40 and bbox_x1(o) > lx2 - 20:
                    digs = re.sub(r"\D", "", o["text"])
                    if 4 <= len(digs) <= 10:
                        cands.append((bbox_x1(o), digs))
            if cands:
                return sorted(cands)[0][1]

    for d in dets:
        m = re.search(r"cheque\s*no\.?\s*[:\-]?\s*(\d{4,10})", d["text"], re.IGNORECASE)
        if m:
            return m.group(1)

    mid_y = image_height // 2 if image_height else 99999
    cands = []
    for d in dets:
        if bbox_y1(d) > mid_y:
            continue
        t = d["text"].strip()
        digs = re.sub(r"\D", "", t)
        if not digs or len(digs) < 4 or len(digs) > 10:
            continue
        if not re.fullmatch(r"\d+", t):
            continue
        if len(digs) == 8:
            dd, mm, yy = int(digs[:2]), int(digs[2:4]), int(digs[4:])
            if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yy <= 2099:
                continue
        cands.append((bbox_x1(d), digs))
    if cands:
        return sorted(cands, key=lambda x: -x[0])[0][1]
    return None


def extract_account_number(dets):
    for d in dets:
        t = d["text"].replace(" ", "")
        t_fixed = re.sub(
            r"(PK\d{2}[A-Z]{2})8([A-Z])", r"\g<1>B\2", t, flags=re.IGNORECASE
        )
        m = re.search(r"PK\d{2}[A-Z]{4}\d{16}", t_fixed, re.IGNORECASE)
        if m:
            return m.group(0).upper()
    for d in dets:
        m = re.search(r"\b(\d{14,20})\b", d["text"].replace(" ", ""))
        if m:
            return m.group(1)
    return None


def extract_account_number_from_micr(micr_str):
    """
    Extract account number from MICR line.
    MICR format: "CHEQUE_NO" BRANCH_CODE : ACCOUNT_NO "TRANSACTION"
    e.g. "00000004"0540056:0000567901558399"000"
    
    Structure:
      part[0] = cheque number   (~8 digits)
      part[1] = branch code     (~7 digits)  
      part[2] = account number  (variable, strip trailing padding zeros)
      part[3] = transaction     (~3 digits, padding — drop it)
    """
    if not micr_str:
        return None

    s = micr_str.replace(" ", "")
    parts = [p for p in re.split(r'\D+', s) if p]

    if len(parts) >= 3:
        # parts[0]=cheque, parts[1]=branch, parts[2]=account, parts[3+]=padding
        acc = parts[2]
        # Strip trailing zeros that are padding (last segment is usually "000")
        # Only strip if the last part looks like padding (all zeros, <=4 digits)
        if len(parts) >= 4:
            tail = parts[3]
            if re.fullmatch(r'0+', tail) and len(tail) <= 4:
                pass  # drop it — don't append
            else:
                acc = acc + tail  # it's real data
        return acc.rstrip('0') if acc.endswith('000') else acc

    elif len(parts) == 2:
        # e.g. "125137560210127:0113200460350001000"
        # first part covers cheque+branch (>=14 digits), second is account+padding
        if len(parts[0]) >= 14:
            acc = parts[1]
            return re.sub(r'0{1,4}$', '', acc) if acc.endswith('000') else acc
        else:
            return parts[1]

    elif len(parts) == 1:
        # No separators — cheque(8) + branch(7) + account
        if len(s) > 15:
            acc = s[15:]
            return re.sub(r'0{1,4}$', '', acc) if acc.endswith('000') else acc
        return s

    return None


def extract_payee(dets):
    def _fix_camel(name):
        if name == name.upper():
            return name
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    def _is_amount_token(t):
        raw_letters = re.sub(r"[^A-Za-z]", "", t)
        misread = set("COILSZGBgcoilszb|")
        if not raw_letters:
            return True
        if not all(c in misread for c in raw_letters):
            return False
        return len(re.sub(r"[^\d]", "", t.translate(_AMOUNT_CHAR_MAP))) >= 3

    skip = r"or\s*bearer|rupees?|^\d{4,}|please|signature|date|cheque|^\-+$"

    for d in dets:
        m = re.match(r"^pa[ytv]\s+(.+)$", d["text"].strip(), re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if not re.search(r"or\s*bearer|rupees?|^\d{4,}", name, re.IGNORECASE):
                if len(name) >= 2:
                    return _fix_camel(name)

    pay_det = None
    for d in dets:
        if re.search(r"^pa[ytv]$", d["text"].strip(), re.IGNORECASE):
            pay_det = d
            break

    if pay_det is None:
        return None

    pay_x2 = bbox_x2(pay_det)
    pay_yc = bbox_center_y(pay_det)
    row_h = bbox_height(pay_det) * 1.2

    cands = []
    for d in dets:
        if d is pay_det:
            continue
        if abs(bbox_center_y(d) - pay_yc) < row_h and bbox_x1(d) > pay_x2 - 10:
            t = d["text"].strip()
            if re.search(skip, t, re.IGNORECASE):
                continue
            if len(t) < 2 or _is_amount_token(t):
                continue
            if re.match(r"PKR", t, re.IGNORECASE):
                continue
            cands.append((bbox_x1(d), t))

    if not cands:
        return None

    cands.sort(key=lambda x: x[0])
    parts = []
    for _, token in cands:
        if re.search(r"PKR|\bRupees\b|\b\d{4,}\b", token, re.IGNORECASE):
            break
        parts.append(_fix_camel(token))
    return " ".join(parts) or None


def extract_micr(dets, image_height):
    """
    Extract the MICR line from the bottom of the cheque.
    Format: "CHEQUE_NO"BRANCH_CODE:ACCOUNT_NO"TRANSACTION"
    """
    cands = []

    # Lower threshold to 40% — some cheques have MICR higher up
    min_y = image_height * 0.40 if image_height else 0

    for d in dets:
        if bbox_y1(d) < min_y:
            continue

        t = d["text"].strip()
        digs = re.sub(r"\D", "", t)
        # Relax to 12 digits to catch partial reads
        if len(digs) >= 12:
            cands.append((bbox_y1(d), t))

    if not cands:
        return None

    # Return the lowest (bottommost) match
    return sorted(cands, key=lambda x: -x[0])[0][1]