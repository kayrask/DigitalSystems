#!/usr/bin/env python3
"""
Regression checker for known false-positive patterns (shadows, facial hair, makeup).

Manifest format (JSON array):
[
  {
    "id": "shadow_case_01",
    "image_path": "path/to/image.jpg",
    "expected_absent": ["hyperpigmentation", "acne"],
    "expected_present": [],
    "expect_quality_reject": false
  }
]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON file with regression cases")
    ap.add_argument("--api", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out", default="models/regression_report.json")
    args = ap.parse_args()

    with open(args.manifest, "r") as f:
        cases = json.load(f)

    report_cases = []
    total = len(cases)
    passed = 0

    for case in cases:
        cid = case.get("id", "unknown_case")
        image_path = case["image_path"]
        exp_absent = case.get("expected_absent", [])
        exp_present = case.get("expected_present", [])
        expect_quality_reject = bool(case.get("expect_quality_reject", False))

        result = {
            "id": cid,
            "image_path": image_path,
            "ok": False,
            "errors": [],
            "violations": [],
            "results": None,
            "uncertainty": None,
            "region_consistency": None,
        }

        try:
            with open(image_path, "rb") as imgf:
                files = {"file": (Path(image_path).name, imgf, "image/jpeg")}
                resp = requests.post(args.api, files=files, timeout=args.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            result["errors"].append(str(e))
            report_cases.append(result)
            continue

        if not payload.get("ok", False):
            msg = payload.get("message", "predict failed")
            result["errors"].append(msg)
            result["quality"] = payload.get("quality")
            if expect_quality_reject and "quality" in msg.lower():
                result["ok"] = True
                passed += 1
            report_cases.append(result)
            continue

        pred = payload.get("results", {})
        result["results"] = pred
        result["uncertainty"] = payload.get("uncertainty")
        result["region_consistency"] = payload.get("region_consistency")

        for label in exp_absent:
            if int(pred.get(label, {}).get("prediction", 0)) == 1:
                result["violations"].append(f"{label}: expected absent but predicted present")
        for label in exp_present:
            if int(pred.get(label, {}).get("prediction", 0)) == 0:
                result["violations"].append(f"{label}: expected present but predicted absent")

        result["ok"] = len(result["violations"]) == 0
        if result["ok"]:
            passed += 1
        report_cases.append(result)

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "api": args.api,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
        },
        "cases": report_cases,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Regression: {passed}/{total} passed")
    print("Wrote:", out.resolve())
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
