#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-data/regression_cases.json}"
API="${API:-http://127.0.0.1:8000/predict}"
REG_REPORT="${REG_REPORT:-models/regression_report.json}"
CAL_REPORT="${CAL_REPORT:-models/calibration_threshold_report_by_tone.json}"
MIN_REG_PASS="${MIN_REG_PASS:-1.0}"
MAX_REG_FAILED="${MAX_REG_FAILED:-0}"
MIN_TUNED_MACRO_F1="${MIN_TUNED_MACRO_F1:-0.73}"
MAX_MACRO_DROP="${MAX_MACRO_DROP:-0.03}"

echo "[ReleaseCheck] Running regression check..."
python scripts/regression_check.py \
  --manifest "$MANIFEST" \
  --api "$API" \
  --out "$REG_REPORT"

echo "[ReleaseCheck] Running release gate..."
python scripts/release_gate.py \
  --regression-report "$REG_REPORT" \
  --calibration-report "$CAL_REPORT" \
  --min-regression-pass-rate "$MIN_REG_PASS" \
  --max-regression-failed "$MAX_REG_FAILED" \
  --min-default-tuned-macro-f1 "$MIN_TUNED_MACRO_F1" \
  --max-default-macro-f1-drop "$MAX_MACRO_DROP"

echo "[ReleaseCheck] PASS"
