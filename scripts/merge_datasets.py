#!/usr/bin/env python3
"""
Merge one or more new dataset CSVs with the existing training CSV,
re-run stratified group split (no subject leakage), and output a
training-ready CSV for train_multilabel.py.

Usage — single new dataset:
  python scripts/merge_datasets.py \
      --new_csv data/raw_hf/trainingdatapro/metadata.csv \
      --existing_csv data/processed/all_multilabel_onehot.csv \
      --out data/processed/all_multilabel_merged.csv

Usage — multiple new datasets at once:
  python scripts/merge_datasets.py \
      --new_csv data/raw_hf/trainingdatapro/metadata.csv data/raw_hf/skincare/metadata.csv \
      --existing_csv data/processed/all_multilabel_onehot.csv \
      --out data/processed/all_multilabel_merged.csv

Flags:
  --no_resplit     Keep existing train/val/test splits for the old data intact;
                   only assign splits to the new data. Use this if you want the
                   original val/test sets to stay comparable across runs.
  --val / --test   Fractions for val / test (default 0.20 / 0.10)
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rename 'filepath' → 'image_path' and ensure one-hot columns exist."""
    if "filepath" in df.columns and "image_path" not in df.columns:
        df = df.rename(columns={"filepath": "image_path"})
    for l in LABELS:
        if l not in df.columns:
            # Try to derive from text 'labels' column
            if "labels" in df.columns:
                df[l] = df["labels"].apply(lambda s: 1 if l in str(s).split(",") else 0)
            else:
                df[l] = 0
    return df


def _ensure_cols(df: "pd.DataFrame", cols: list) -> "pd.DataFrame":
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def _stratified_group_split(df: "pd.DataFrame", val_frac: float, test_frac: float, seed: int = 42) -> "pd.DataFrame":
    """
    Assign 'split' column (train/val/test) respecting group_id boundaries
    so subjects don't leak between splits.
    """
    rng = random.Random(seed)

    groups: dict = {}
    for _, row in df.iterrows():
        g = str(row["group_id"])
        labs = [l for l in LABELS if row.get(l, 0) == 1]
        lab = labs[0] if labs else "unknown"
        groups.setdefault(g, Counter())[lab] += 1

    gid = list(groups.keys())
    y = [groups[g].most_common(1)[0][0] for g in gid]
    n = len(gid)
    if n == 0:
        return df

    def _assign(test_set, val_set):
        split_map = {}
        for i, g in enumerate(gid):
            if i in test_set:
                split_map[g] = "test"
            elif i in val_set:
                split_map[g] = "val"
            else:
                split_map[g] = "train"
        df_out = df.copy()
        df_out["split"] = df_out["group_id"].map(split_map).fillna("train")
        return df_out

    if _HAS_SK and len(set(y)) > 1 and n >= 4:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        best_test, gap = set(), 1e9
        for tr, te in sgkf.split(gid, y, groups=gid):
            for cand in (te, tr):
                g = abs(len(cand) / n - test_frac)
                if g < gap:
                    best_test, gap = set(cand.tolist()), g

        remaining = sorted(set(range(n)) - best_test)
        best_val: set = set()
        if remaining and val_frac > 0:
            rem_g = [gid[i] for i in remaining]
            rem_y = [y[i] for i in remaining]
            target_val = val_frac / max(1e-9, 1 - test_frac)
            if len(set(rem_y)) > 1 and len(rem_g) >= 2:
                sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
                best_rel, gap2 = set(), 1e9
                for tr2, te2 in sgkf2.split(rem_g, rem_y, groups=rem_g):
                    for cand in (te2, tr2):
                        g2 = abs(len(cand) / len(rem_g) - target_val)
                        if g2 < gap2:
                            best_rel, gap2 = set(cand.tolist()), g2
                best_val = {remaining[i] for i in sorted(best_rel)}
        return _assign(best_test, best_val)

    # Fallback: shuffle by label bucket
    by_lbl: dict = {}
    for g, lab in zip(gid, y):
        by_lbl.setdefault(lab, []).append(g)
    for lab in by_lbl:
        rng.shuffle(by_lbl[lab])
    test_g, val_g = set(), set()
    for lab, gs in by_lbl.items():
        k = len(gs)
        k_test = int(round(k * test_frac))
        k_val = int(round(k * val_frac))
        test_g.update(gs[:k_test])
        val_g.update(gs[k_test:k_test + k_val])
    idx_map = {g: i for i, g in enumerate(gid)}
    return _assign({idx_map[g] for g in test_g}, {idx_map[g] for g in val_g})


def _print_summary(df: "pd.DataFrame", title: str = ""):
    if title:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
    print(f"  Total rows:    {len(df)}")
    split_counts = df["split"].value_counts().to_dict()
    for s in ("train", "val", "test"):
        print(f"  {s:8s}: {split_counts.get(s, 0)}")
    print(f"\n  Label counts  (train split):")
    train = df[df["split"] == "train"]
    for l in LABELS:
        pos = int(train[l].sum()) if l in train.columns else 0
        pct = 100 * pos / max(1, len(train))
        flag = "  ⚠  LOW" if pos < 30 else ""
        print(f"    {l:20s}: {pos:4d} / {len(train)}  ({pct:.1f}%){flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge new skin datasets into the AURAI training CSV.")
    ap.add_argument("--new_csv", nargs="+", required=True,
                    help="Path(s) to new dataset metadata CSVs (output of download_hf_dataset.py)")
    ap.add_argument("--existing_csv", required=True,
                    help="Existing training CSV, e.g. data/processed/all_multilabel_onehot.csv")
    ap.add_argument("--out", required=True,
                    help="Output merged CSV path, e.g. data/processed/all_multilabel_merged.csv")
    ap.add_argument("--val", type=float, default=0.20, help="Validation fraction (default 0.20)")
    ap.add_argument("--test", type=float, default=0.10, help="Test fraction (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_resplit", action="store_true",
                    help="Keep existing splits for old data intact; only split the new data. "
                         "Makes old val/test sets comparable across data additions.")
    args = ap.parse_args()

    COLS = ["image_path", "labels", "group_id", "dataset", "region", "split"] + LABELS

    # -- Load existing --
    existing = pd.read_csv(args.existing_csv)
    existing = _normalise(existing)
    existing = _ensure_cols(existing, COLS)
    print(f"Existing: {len(existing)} rows  ({args.existing_csv})")

    # -- Load new CSVs --
    new_frames = []
    for path in args.new_csv:
        if not Path(path).exists():
            print(f"WARNING: {path} does not exist — skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df = _normalise(df)
        df = _ensure_cols(df, COLS)
        print(f"New:      {len(df)} rows  ({path})")
        new_frames.append(df)

    if not new_frames:
        print("ERROR: No valid new CSVs found.", file=sys.stderr)
        sys.exit(1)

    combined_new = pd.concat(new_frames, ignore_index=True)

    if args.no_resplit:
        # Keep old splits; only assign splits to new rows
        print("\n--no_resplit: preserving existing split assignments for old data.")
        combined_new_split = _stratified_group_split(combined_new, args.val, args.test, args.seed)
        merged = pd.concat([existing[COLS], combined_new_split[COLS]], ignore_index=True)
    else:
        # Pool everything, deduplicate (new data wins on same path), re-split all
        print("\nRe-splitting entire merged dataset (new + old together).")
        merged_raw = pd.concat([existing[COLS], combined_new[COLS]], ignore_index=True)
        # Remove duplicate image paths — last occurrence wins (= new data)
        merged_raw = merged_raw.drop_duplicates(subset=["image_path"], keep="last").reset_index(drop=True)
        merged = _stratified_group_split(merged_raw, args.val, args.test, args.seed)

    # -- Enforce int dtype on one-hot columns --
    for l in LABELS:
        merged[l] = pd.to_numeric(merged[l], errors="coerce").fillna(0).astype(int)

    # -- Print summary --
    _print_summary(existing, title="BEFORE (existing only)")
    _print_summary(merged, title="AFTER  (merged)")

    # -- Save --
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[COLS].to_csv(out_path, index=False)
    print(f"\nSaved merged CSV → {out_path}")

    # -- Next step hint --
    print(f"\n{'='*60}")
    print("NEXT STEP — retrain the multilabel model:")
    print(f"  python scripts/train_multilabel.py \\")
    print(f"      --csv {out_path} \\")
    print(f"      --out models/best_multilabel_v2.pt \\")
    print(f"      --epochs 30 --batch 32")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
