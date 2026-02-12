#!/usr/bin/env python3
"""
Prefill regression manifest with model-based label suggestions.

For each case in a manifest:
- Calls /predict with the image
- Stores model probabilities
- Stores suggested_present / suggested_absent

By default, this does NOT overwrite expected_* labels unless they are empty.
"""
import argparse
import json
from pathlib import Path

import requests

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Input regression manifest JSON")
    ap.add_argument("--api", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument(
        "--suggest-threshold",
        type=float,
        default=0.50,
        help="Probability threshold for suggested_present",
    )
    ap.add_argument(
        "--apply-to-expected",
        action="store_true",
        help="If set, fill expected_present/expected_absent when those lists are empty",
    )
    ap.add_argument("--out", default="", help="Output path (default: overwrite --manifest)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r") as f:
        cases = json.load(f)

    updated = []
    for i, case in enumerate(cases, start=1):
        image_path = case.get("image_path")
        if not image_path:
            case.setdefault("prefill_error", "missing image_path")
            updated.append(case)
            continue

        try:
            with open(image_path, "rb") as imgf:
                files = {"file": (Path(image_path).name, imgf, "image/jpeg")}
                resp = requests.post(args.api, files=files, timeout=args.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            case["prefill_error"] = str(e)
            updated.append(case)
            continue

        if not payload.get("ok", False):
            case["prefill_error"] = payload.get("message", "predict failed")
            updated.append(case)
            continue

        pred = payload.get("results", {})
        probs = {l: float(pred.get(l, {}).get("probability", 0.0)) for l in LABELS}
        suggested_present = [l for l, p in probs.items() if p >= args.suggest_threshold]
        suggested_absent = [l for l in LABELS if l not in suggested_present]

        case["model_probabilities"] = probs
        case["suggested_present"] = suggested_present
        case["suggested_absent"] = suggested_absent
        case["prefill_error"] = None

        if args.apply_to_expected:
            if not case.get("expected_present"):
                case["expected_present"] = list(suggested_present)
            if not case.get("expected_absent"):
                case["expected_absent"] = list(suggested_absent)

        updated.append(case)
        if i % 10 == 0:
            print(f"Processed {i}/{len(cases)}")

    out_path = Path(args.out) if args.out else manifest_path
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)

    print("Wrote:", out_path.resolve())


if __name__ == "__main__":
    main()

