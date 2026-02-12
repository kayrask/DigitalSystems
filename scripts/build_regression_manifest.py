#!/usr/bin/env python3
"""
Build a regression manifest from recent face crops in uploads/.

This creates placeholder cases you can quickly label for
expected_absent/expected_present before running regression_check.py.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads_dir", default="uploads")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", default="data/regression_cases.json")
    args = ap.parse_args()

    up = Path(args.uploads_dir)
    files = sorted(up.glob("*_face_raw.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[: args.count]

    if not files:
        raise SystemExit(f"No *_face_raw.jpg files found in {up.resolve()}")

    cases = []
    for i, p in enumerate(files, start=1):
        cases.append(
            {
                "id": f"case_{i:03d}",
                "image_path": str(p).replace("\\", "/"),
                "expected_absent": [],
                "expected_present": [],
                "notes": "Fill expected labels before strict regression checks.",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Wrote {len(cases)} cases to {out.resolve()}")


if __name__ == "__main__":
    main()

