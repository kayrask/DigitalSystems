#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_metadata_unid_csv.py

Generate a unified metadata CSV for the Skin Defects dataset laid out as:
data/raw_b/
  ├─ 1/
  │   ├─ front.jpg
  │   ├─ left-side.jpg
  │   └─ right-side.jpg
  ├─ 2/
  │   ├─ front.jpg
  │   ├─ left-side.jpg
  │   └─ right-side.jpg
  └─ ...

Labels are provided either:
  A) from a column in the main CSV (rare in your case), or
  B) from a separate label map CSV with headers: id,label

Outputs CSV with columns: filepath,label,subject_id,split
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

# pandas is required
try:
    import pandas as pd
except Exception:
    print("This script requires pandas. Please `pip install pandas`.", file=sys.stderr)
    raise

# Optional: sklearn for stratified group split
_USE_SK = True
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    _USE_SK = False


# ---------------------------
# Utilities for CSV handling
# ---------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    return df

def read_main_csv(csv_path: str) -> pd.DataFrame:
    """Read the dataset's main CSV. Must contain an 'id' column (subject ID)."""
    df = pd.read_csv(csv_path)
    df = _norm_cols(df)
    if "id" not in df.columns:
        raise KeyError(f"Expected an 'id' column in {csv_path}, found {list(df.columns)}")
    df["id"] = df["id"].astype(str).str.strip()
    return df

def read_label_map(label_map_csv: str) -> dict:
    """Read an id→label mapping CSV with headers: id,label."""
    df = pd.read_csv(label_map_csv)
    df = _norm_cols(df)
    if not {"id", "label"}.issubset(df.columns):
        raise KeyError(f"{label_map_csv} must have columns: id,label")
    df["id"] = df["id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return dict(zip(df["id"], df["label"]))


# ---------------------------
# Image collection
# ---------------------------

def collect_images(root: str, label_map: dict, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """
    Collect only the expected views per subject.
    Expected filenames (stem, case-insensitive): front, left-side, right-side
    """
    rows = []
    root_path = Path(root)
    expected = {"front", "left-side", "right-side"}

    for subj_dir in sorted([p for p in root_path.iterdir() if p.is_dir()], key=lambda p: p.name):
        sid = subj_dir.name.strip()
        if sid not in label_map:
            # Skip subjects with no label
            continue
        label = label_map[sid]
        for f in sorted(subj_dir.iterdir(), key=lambda x: x.name.lower()):
            if f.is_file() and f.suffix.lower() in exts:
                stem = f.stem.strip().lower()
                if stem in expected:
                    rows.append({
                        "filepath": str(f.as_posix()),
                        "label": label,
                        "subject_id": sid
                    })
    return rows


# ---------------------------
# Grouped, (optionally) stratified split
# ---------------------------

def stratified_group_split(rows, val_frac: float, test_frac: float, seed: int = 42):
    """
    Returns a list of split tags ('train'/'val'/'test'), aligned with `rows`.
    Ensures group-wise split by subject_id. Uses StratifiedGroupKFold if available.
    """
    rng = random.Random(seed)

    # Derive one label per subject (majority among that subject's images)
    subj_counts = {}  # sid -> Counter of labels
    for r in rows:
        sid = r["subject_id"]
        lbl = r["label"]
        subj_counts.setdefault(sid, Counter())
        subj_counts[sid][lbl] += 1

    subj_labels = {sid: counts.most_common(1)[0][0] for sid, counts in subj_counts.items()}
    uniq_subjects = list(subj_labels.keys())
    uniq_labels = [subj_labels[s] for s in uniq_subjects]

    n = len(uniq_subjects)
    if n == 0:
        return []

    # Helper to assign by subject sets
    def assign(rows, test_set, val_set):
        subj_to_split = {}
        for i, sid in enumerate(uniq_subjects):
            if i in test_set:
                subj_to_split[sid] = "test"
            elif i in val_set:
                subj_to_split[sid] = "val"
            else:
                subj_to_split[sid] = "train"
        return [subj_to_split[r["subject_id"]] for r in rows]

    # If sklearn is available, try a 2-pass SGKF selection to approximate the desired fractions
    if _USE_SK and (test_frac > 0 or val_frac > 0) and len(set(uniq_labels)) > 1 and len(uniq_subjects) >= 4:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)

        # Pick test subjects
        best_test_idx = set()
        best_gap = 1e9
        for tr, te in sgkf.split(uniq_subjects, uniq_labels, groups=uniq_subjects):
            for candidate in (te, tr):  # try either fold as test to get closer to target fraction
                gap = abs(len(candidate) / n - test_frac)
                if gap < best_gap:
                    best_gap = gap
                    best_test_idx = set(candidate.tolist())

        # Pick val subjects from remaining
        remaining = sorted(set(range(n)) - best_test_idx)
        if remaining and val_frac > 0:
            rem_subjects = [uniq_subjects[i] for i in remaining]
            rem_labels = [subj_labels[s] for s in rem_subjects]
            # Adjust target: val fraction of the remaining pool
            target_val = val_frac / max(1e-9, (1 - test_frac))
            if len(set(rem_labels)) > 1 and len(rem_subjects) >= 2:
                sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
                best_val_rel = set()
                best_gap2 = 1e9
                for tr2, te2 in sgkf2.split(rem_subjects, rem_labels, groups=rem_subjects):
                    for candidate in (te2, tr2):
                        gap2 = abs(len(candidate) / len(rem_subjects) - target_val)
                        if gap2 < best_gap2:
                            best_gap2 = gap2
                            best_val_rel = set(candidate.tolist())
                best_val_idx = set(remaining[i] for i in sorted(best_val_rel))
            else:
                best_val_idx = set()
        else:
            best_val_idx = set()

        return assign(rows, best_test_idx, best_val_idx)

    # Fallback: simple group-aware random split (approx stratified by label)
    by_label = {}
    for sid, lbl in subj_labels.items():
        by_label.setdefault(lbl, []).append(sid)
    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    test_subjects = set()
    val_subjects = set()
    train_subjects = set()

    for lbl, sids in by_label.items():
        k = len(sids)
        k_test = int(round(k * test_frac))
        k_val = int(round(k * val_frac))
        test_subjects.update(sids[:k_test])
        val_subjects.update(sids[k_test:k_test + k_val])
        train_subjects.update(sids[k_test + k_val:])

    # Map subject IDs to global indices
    subj_index = {sid: i for i, sid in enumerate(uniq_subjects)}
    test_idx = {subj_index[s] for s in test_subjects}
    val_idx = {subj_index[s] for s in val_subjects}
    return assign(rows, test_idx, val_idx)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of raw_b (folders 1,2,3,...)")
    ap.add_argument("--csv", required=True, help="Main CSV (e.g., 'Facial Skin Condition Dataset.csv')")
    ap.add_argument("--out", required=True, help="Output metadata CSV path")
    ap.add_argument("--val", type=float, default=0.2, help="Validation fraction (by subject)")
    ap.add_argument("--test", type=float, default=0.1, help="Test fraction (by subject)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp",
                    help="Comma-separated image extensions")
    ap.add_argument("--label_map", type=str, default=None,
                    help="Optional CSV with columns id,label (subject → class)")
    args = ap.parse_args()

    # Load main CSV (for subject IDs, etc.)
    main_df = read_main_csv(args.csv)

    # Try to discover a label column in the main CSV
    label_col_candidates = ["label", "class", "target", "defect", "condition", "category"]
    available_label_col = next((c for c in label_col_candidates if c in main_df.columns), None)

    # Build label_map (id -> label)
    if available_label_col:
        label_map = dict(zip(
            main_df["id"],
            main_df[available_label_col].astype(str).str.strip().str.lower()
        ))
    elif args.label_map:
        label_map = read_label_map(args.label_map)
    else:
        raise SystemExit(
            "No label column found in the main CSV and --label_map was not provided.\n"
            "→ Create/fill a CSV with headers 'id,label' (values like acne/redness/bags) and run with:\n"
            "   --label_map path/to/labels.csv"
        )

    # Collect images
    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    rows = collect_images(args.root, label_map, exts=exts)
    if not rows:
        print("No images collected. Ensure subject folders exist and filenames are front/left-side/right-side.",
              file=sys.stderr)
        sys.exit(1)

    # Split (group-aware)
    splits = stratified_group_split(rows, args.val, args.test, args.seed)
    for r, s in zip(rows, splits):
        r["split"] = s

    # Write output CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "subject_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Print quick summary
    n_total = len(rows)
    per_split = Counter([r["split"] for r in rows])
    per_label = Counter([r["label"] for r in rows])
    print(f"Wrote {n_total} rows to {out_path}")
    print("Splits:", dict(per_split))
    print("Labels:", dict(per_label))


if __name__ == "__main__":
    main()
