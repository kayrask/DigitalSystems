#!/usr/bin/env python3
"""
Week 4 release hardening gate.

Fails with non-zero exit code if release criteria are not met.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with p.open("r") as f:
        return json.load(f)


def _macro_f1(metrics_block: dict) -> float:
    vals = []
    for _, m in (metrics_block or {}).items():
        if isinstance(m, dict) and "f1" in m and m["f1"] is not None:
            vals.append(float(m["f1"]))
    return float(mean(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression-report", default="models/regression_report.json")
    ap.add_argument("--calibration-report", default="models/calibration_threshold_report_by_tone.json")
    ap.add_argument("--require-calibration", action="store_true")
    ap.add_argument("--min-regression-pass-rate", type=float, default=1.0)
    ap.add_argument("--max-regression-failed", type=int, default=0)
    ap.add_argument("--min-default-tuned-macro-f1", type=float, default=0.75)
    ap.add_argument("--max-default-macro-f1-drop", type=float, default=0.03)
    args = ap.parse_args()

    failures: list[str] = []

    # 1) Regression gate
    reg = _load_json(args.regression_report)
    summary = reg.get("summary", {})
    total = int(summary.get("total", 0))
    passed = int(summary.get("passed", 0))
    failed = int(summary.get("failed", max(0, total - passed)))
    pass_rate = float(summary.get("pass_rate", (passed / total if total else 0.0)))

    print("[ReleaseGate] Regression summary")
    print(f"  total={total} passed={passed} failed={failed} pass_rate={pass_rate:.4f}")

    if failed > int(args.max_regression_failed):
        failures.append(f"regression failed cases {failed} > max {args.max_regression_failed}")
    if pass_rate < float(args.min_regression_pass_rate):
        failures.append(
            f"regression pass_rate {pass_rate:.4f} < min {float(args.min_regression_pass_rate):.4f}"
        )

    # 2) Calibration / threshold quality gate (optional but recommended)
    cal_path = Path(args.calibration_report)
    if cal_path.exists():
        cal = _load_json(str(cal_path))
        cur_default = (((cal.get("metrics_current") or {}).get("default")) or {})
        tuned_default = (((cal.get("metrics_tuned") or {}).get("default")) or {})
        cur_macro = _macro_f1(cur_default)
        tuned_macro = _macro_f1(tuned_default)
        delta = tuned_macro - cur_macro

        print("[ReleaseGate] Calibration summary (default group)")
        print(f"  current_macro_f1={cur_macro:.4f}")
        print(f"  tuned_macro_f1={tuned_macro:.4f}")
        print(f"  tuned_minus_current={delta:+.4f}")

        if tuned_macro < float(args.min_default_tuned_macro_f1):
            failures.append(
                f"tuned default macro_f1 {tuned_macro:.4f} < min {float(args.min_default_tuned_macro_f1):.4f}"
            )
        if delta < -float(args.max_default_macro_f1_drop):
            failures.append(
                f"default macro_f1 drop {-delta:.4f} exceeds max {float(args.max_default_macro_f1_drop):.4f}"
            )
    elif args.require_calibration:
        failures.append(f"missing calibration report: {cal_path}")
    else:
        print(f"[ReleaseGate] Calibration report not found at {cal_path}, skipping.")

    if failures:
        print("[ReleaseGate] FAIL")
        for i, msg in enumerate(failures, start=1):
            print(f"  {i}. {msg}")
        return 1

    print("[ReleaseGate] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

