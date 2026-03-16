"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Run with:
    python inference.py --image path/to/cheque.jpg
    python inference.py --image path/to/cheque.jpg --json
    python inference.py --image path/to/cheque.jpg --output out.json
    python inference.py --batch images/        # process all images

Output JSON structure:
{
  "bank_name": "...",
  "date": "DD/MM/YYYY",
  "pay_to": "...",
  "amount_figures": "16,780",
  "amount_numeric": 16780.0,
  "Iban number": "PK23...",
  "cheque_number": "...",
  "crops": {
    "date":   "<base64 JPEG>",
    "amount": "<base64 JPEG>",
    "pay":    "<base64 JPEG>",
    "rupees": "<base64 JPEG>"
  }
}
"""

import argparse
import base64
import glob
import json
import os
import re

import cv2
import numpy as np
from num2words import num2words

from src.extractor import (extract_account_number,
                           extract_amount_from_crop_ocr, extract_amount_ocr,
                           extract_bank_name, extract_cheque_number,
                           extract_date, extract_date_from_crop_ocr,
                           extract_payee, get_date_region, get_pay_region,
                           get_pkr_region, get_region_right_of_label,
                           get_rupees_region, get_signature_region,
                           extract_micr, extract_account_number_from_micr)
from src.models.amount_net import AmountReader
from src.models.date_net import DateReader
from src.ocr_engine import OCREngine, fix_exif_rotation, preprocess_image

DEBUG_DIR = "debug_output"
OUTPUT_DIR = "output"


def _crop_to_b64(crop_bgr) -> str | None:
    """Encode a BGR numpy crop as a base64 JPEG string."""
    if crop_bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _save_crop(crop_bgr, filename: str, debug_dir: str = None):
    if crop_bgr is None:
        return
    out_dir = debug_dir or DEBUG_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, crop_bgr)
    print(f"  [saved] {path}")


def _save_detections(image_bgr, detections: list, debug_dir: str = None):
    out_dir = debug_dir or DEBUG_DIR
    os.makedirs(out_dir, exist_ok=True)
    vis = image_bgr.copy()
    for d in detections:
        pts = np.array(d["bbox"], dtype=np.int32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{d['text'][:20]} ({d['confidence']:.2f})",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    path = os.path.join(out_dir, "ocr_detections.jpg")
    cv2.imwrite(path, vis)
    print(f"  [saved] {path}")


def run_pipeline(
    image_path: str,
    date_model_path: str = "weights/date_model.pt",
    amount_model_path: str = "weights/amount_model.pt",
    use_gpu: bool = False,
    output_dir: str = None,
) -> dict:
    """Run the full cheque parsing pipeline on a single image.
    If output_dir is given, debug images and JSON are saved there."""
    cheque_debug_dir = output_dir or DEBUG_DIR

    print("\n" + "=" * 60)
    print("  CHEQUE PARSER")
    print("=" * 60)

    # ── 1. Load models
    date_reader = DateReader(date_model_path)
    amount_reader = AmountReader(amount_model_path)
    ocr_engine = OCREngine(use_gpu=use_gpu)

    # ── 2. Preprocess
    print(f"\n[Step 1] Loading: {image_path}")
    image_path = fix_exif_rotation(image_path)
    processed = preprocess_image(image_path)
    cv2.imwrite("_processed.jpg", processed)
    print(f"  Shape after preprocessing: {processed.shape}")

    # ── 3. PaddleOCR
    print("\n[Step 2] Running PaddleOCR...")
    detections = ocr_engine.run("_processed.jpg")
    print("\n─── Raw OCR output ─────────────────────────────────────")
    for i, d in enumerate(detections):
        print(f"  [{i:02d}] conf={d['confidence']:.2f}  \"{d['text']}\"")
    print("────────────────────────────────────────────────────────")

    # ── 4. Crop all 4 regions
    print("\n[Step 3] Cropping regions of interest...")

    # DATE — generous crop using date label + cheque no as anchors
    date_crop, _ = get_date_region(processed, detections)
    _save_crop(date_crop, "date_crop.jpg", cheque_debug_dir)

    # AMOUNT — PKR box or right of "Rupees"
    amount_crop = get_pkr_region(processed, detections)
    if amount_crop is None:
        amount_crop, _ = get_region_right_of_label(
            processed, detections, r"^rupees?$", pad_top=20, pad_bottom=30
        )
    _save_crop(amount_crop, "amount_crop.jpg", cheque_debug_dir)

    # PAY — smart crop: right of "Pay" label to before "or bearer" / PKR box
    pay_crop, _ = get_pay_region(processed, detections)
    _save_crop(pay_crop, "pay_crop.jpg", cheque_debug_dir)

    # RUPEES — smart crop: includes continuation lines below "Rupees" label
    rupees_crop, _ = get_rupees_region(processed, detections)
    _save_crop(rupees_crop, "rupees_crop.jpg", cheque_debug_dir)
    
    # SIGNATURE
    signature_crop, _ = get_signature_region(processed, detections)
    _save_crop(signature_crop, "signature_crop.jpg", cheque_debug_dir)

    # ── 5. Extract DATE
    print("\n[Step 4] Extracting DATE...")

    date_val = extract_date(detections)
    if date_val:
        print(f"  [date] PaddleOCR raw: {date_val}")

    if date_val is None and date_crop is not None:
        print("  [date] trying DateNet model...")
        from PIL import Image as PILImage

        pil_crop = PILImage.fromarray(cv2.cvtColor(date_crop, cv2.COLOR_BGR2RGB))
        dr = date_reader.predict(pil_crop)
        if dr["valid"]:
            date_val = dr["date"]
            print(f"  [date] DateNet: {date_val} (conf={dr['confidence']:.3f})")

    if date_val is None:
        print("  [date] trying PaddleOCR crop fallback...")
        date_val = extract_date_from_crop_ocr(date_crop, ocr_engine)

    # ── 6. Extract AMOUNT
    print("\n[Step 5] Extracting AMOUNT...")

    amount_val = extract_amount_ocr(detections, cheque_number=extract_cheque_number(detections, processed.shape[0]))
    if amount_val:
        print(f"  [amount] full-image PaddleOCR: {amount_val}")

    if amount_val is None and amount_crop is not None:
        print("  [amount] trying crop OCR...")
        amount_val = extract_amount_from_crop_ocr(amount_crop, ocr_engine)

    if amount_val is None and amount_crop is not None:
        print("  [amount] trying AmountNet model...")
        ar = amount_reader.predict(amount_crop, min_conf=0.85)
        if ar["valid"]:
            amount_val = ar["amount"]
            print(f"  [amount] AmountNet: {amount_val} (conf={ar['confidence']:.3f})")

    # ── 7. Extract other fields
    print("\n[Step 6] Extracting other fields...")
    cheque_no = extract_cheque_number(detections, processed.shape[0])
    account_no = extract_account_number(detections)
    bank_name = extract_bank_name(detections)
    micr = extract_micr(detections, processed.shape[0])
    acc_from_micr = extract_account_number_from_micr(micr)
    print(f"  [MICR raw]: {micr}")
    print(f"  [MICR account]: {acc_from_micr}")

    numeric_amount = None
    amount_in_alphabets = None
    if amount_val:
        try:
            numeric_amount = float(amount_val.replace(",", ""))
            rupees = int(numeric_amount)
            paisas = int(round((numeric_amount - rupees) * 100))
            r_str = num2words(rupees).title().replace(",", "") + " Rupees"
            if paisas > 0:
                p_str = num2words(paisas).title() + " Paisas"
                amount_in_alphabets = f"{r_str} And {p_str}"
            else:
                amount_in_alphabets = f"{r_str} Only"
        except ValueError:
            pass
    # ── 8. Build result with base64 crops
    result = {
        "bank_name": bank_name,
        "date": date_val,
        "amount_figures": amount_in_alphabets,
        "amount_numeric": numeric_amount,
        "amount_in_alphabets": amount_in_alphabets,
        "Iban number": account_no,
        "account_number": acc_from_micr,
        "cheque_number": cheque_no,
        "micr": micr,
        "crops": {
            "date": _crop_to_b64(date_crop),
            "amount": _crop_to_b64(amount_crop),
            "pay": _crop_to_b64(pay_crop),
            "rupees": _crop_to_b64(rupees_crop),
            "signature": _crop_to_b64(signature_crop),
        },
    }

    # ── 9. Print summary
    print("\n" + "=" * 55)
    print("         EXTRACTED CHEQUE FIELDS")
    print("=" * 55)
    for k, v in result.items():
        if k == "crops":
            for ck, cv_ in v.items():
                status = f"<base64 {len(cv_)} chars>" if cv_ else "None"
                print(f"  {'crop_' + ck:<22}: {status}")
        else:
            print(f"  {k.replace('_', ' ').title():<22}: {v}")
    print("=" * 55)

    if date_val is None:
        print("\n  ⚠  Date could not be parsed.")
    if amount_val is None:
        print("  ⚠  Amount could not be parsed.")

    # ── 10. Save debug images
    print("\n[Step 7] Saving debug images...")
    _save_detections(processed, detections, cheque_debug_dir)
    os.makedirs(cheque_debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(cheque_debug_dir, "preprocessed.jpg"), processed)
    print(f"  [saved] {cheque_debug_dir}/preprocessed.jpg")
    print(
        f"\n  📁 '{cheque_debug_dir}/' → preprocessed, date_crop, amount_crop, pay_crop, rupees_crop, ocr_detections"
    )

    # ── 11. Save JSON result to the output directory
    if output_dir:
        json_path = os.path.join(output_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [saved] {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cheque field extractor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single cheque image")
    group.add_argument(
        "--batch", help="Path to directory of cheque images (processes all)"
    )
    parser.add_argument("--date-model", default="weights/date_model.pt")
    parser.add_argument("--amount-model", default="weights/amount_model.pt")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON (including base64 crops) to stdout",
    )
    parser.add_argument(
        "--output", default=None, help="Save JSON to this file (single image mode only)"
    )
    args = parser.parse_args()

    # ── Batch mode: process all images in a directory
    if args.batch:
        img_dir = args.batch
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        image_files.sort()

        if not image_files:
            print(f"No images found in '{img_dir}'")
            return

        print(f"\n{'='*60}")
        print(f"  BATCH MODE: {len(image_files)} images found")
        print(f"{'='*60}")

        results = []
        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            cheque_output_dir = os.path.join(OUTPUT_DIR, basename)
            os.makedirs(cheque_output_dir, exist_ok=True)

            print(f"\n{'─'*60}")
            print(f"  Processing: {img_path}")
            print(f"  Output dir: {cheque_output_dir}")
            print(f"{'─'*60}")

            try:
                result = run_pipeline(
                    image_path=img_path,
                    date_model_path=args.date_model,
                    amount_model_path=args.amount_model,
                    use_gpu=args.gpu,
                    output_dir=cheque_output_dir,
                )
                result["source_file"] = os.path.basename(img_path)
                results.append(result)
            except Exception as e:
                print(f"  ⚠ ERROR processing {img_path}: {e}")
                results.append(
                    {"source_file": os.path.basename(img_path), "error": str(e)}
                )

        # Save a combined summary
        summary_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
        summary = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "crops"}
            summary.append(entry)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE: {len(results)} cheques processed")
        print(f"  Summary: {summary_path}")
        print(f"{'='*60}")
        return

    # ── Single image mode
    result = run_pipeline(
        image_path=args.image,
        date_model_path=args.date_model,
        amount_model_path=args.amount_model,
        use_gpu=args.gpu,
    )

    if args.json or args.output:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        if args.json:
            print("\n─── JSON OUTPUT ─────────────────────────────────────────")
            print(json_str)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"\n  [saved] JSON → {args.output}")


if __name__ == "__main__":
    main()