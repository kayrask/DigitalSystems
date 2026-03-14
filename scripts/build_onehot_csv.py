#!/usr/bin/env python3
"""
Convert a raw metadata CSV (filepath/labels as comma-separated text)
into one-hot format ready for train_multilabel.py.

This is useful if you have a CSV produced by make_metadata_all_multilabel.py
(which outputs a 'labels' text column) and want to convert it to the
one-hot format that the training script expects.

Usage:
  python scripts/build_onehot_csv.py \
      --in_csv data/processed/all_multilabel.csv \
      --out_csv data/processed/all_multilabel_onehot.csv

  # Optionally re-run stratified split
  python scripts/build_onehot_csv.py \
      --in_csv data/processed/all_multilabel.csv \
      --out_csv data/processed/all_multilabel_onehot.csv \
      --resplit --val 0.20 --test 0.10

Column mapping:
  Input 'filepath' column → Output 'image_path' column (required by trainer)
  Input 'labels' column   → Expanded to one-hot: acne, bags, blackheads,
                             hyperpigmentation, redness
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]

CANON = {
    "acne": "acne", "acnes": "acne",
    "redness": "redness", "erythema": "redness",
    "bags": "bags", "under_eyes_bags": "bags", "bags_under_eyes": "bags",
    "dark_circles": "bags", "dark_circle": "bags", "eye_bags": "bags", "puffy_eyes": "bags",
    "blackhead": "blackheads", "blackheads": "blackheads", "comedones": "blackheads",
    "hyperpigmentation": "hyperpigmentation", "melasma": "hyperpigmentation",
    "darkspots": "hyperpigmentation", "dark_spots": "hyperpigmentation",
    "pigmentation": "hyperpigmentation",
}


def canon(s: str):
    s = s.strip().lower().replace(" ", "_")
    return CANON.get(s, s)


try:
    import pandas as pd
except ImportError:
    print("pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def stratified_group_split(df, val_frac, test_frac, seed=42):
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
        split_map = {g: ("test" if i in test_set else "val" if i in val_set else "train")
                     for i, g in enumerate(gid)}
        df_out = df.copy()
        df_out["split"] = df_out["group_id"].map(split_map).fillna("train")
        return df_out

    if _HAS_SK and len(set(y)) > 1 and n >= 4:
        n_splits_test = max(2, min(20, round(1.0 / max(test_frac, 1e-9))))
        sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
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
            target = val_frac / max(1e-9, 1 - test_frac)
            n_splits_val = max(2, min(20, round(1.0 / max(target, 1e-9))))
            if len(set(rem_y)) > 1 and len(rem_g) >= n_splits_val:
                sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
                best_rel, gap2 = set(), 1e9
                for tr2, te2 in sgkf2.split(rem_g, rem_y, groups=rem_g):
                    for cand in (te2, tr2):
                        g2 = abs(len(cand) / len(rem_g) - target)
                        if g2 < gap2:
                            best_rel, gap2 = set(cand.tolist()), g2
                best_val = {remaining[i] for i in sorted(best_rel)}
        return _assign(best_test, best_val)

    # Fallback
    by_lbl: dict = {}
    for g, lab in zip(gid, y):
        by_lbl.setdefault(lab, []).append(g)
    for lab in by_lbl:
        rng.shuffle(by_lbl[lab])
    test_g, val_g = set(), set()
    for lab, gs in by_lbl.items():
        k = len(gs)
        test_g.update(gs[:int(round(k * test_frac))])
        val_g.update(gs[int(round(k * test_frac)):int(round(k * (test_frac + val_frac)))])
    idx_map = {g: i for i, g in enumerate(gid)}
    return _assign({idx_map[g] for g in test_g}, {idx_map[g] for g in val_g})


def main():
    ap = argparse.ArgumentParser(description="Convert raw multilabel CSV to one-hot format.")
    ap.add_argument("--in_csv", required=True, help="Input CSV (make_metadata output: filepath, labels, group_id…)")
    ap.add_argument("--out_csv", required=True, help="Output one-hot CSV (image_path, split, acne, bags, …)")
    ap.add_argument("--resplit", action="store_true", help="Re-run stratified group split (overwrites 'split' column)")
    ap.add_argument("--val", type=float, default=0.20)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} rows from {args.in_csv}")
    print(f"Columns: {df.columns.tolist()}")

    # Rename filepath → image_path
    if "filepath" in df.columns and "image_path" not in df.columns:
        df = df.rename(columns={"filepath": "image_path"})

    if "image_path" not in df.columns:
        print("ERROR: no 'image_path' or 'filepath' column found.", file=sys.stderr)
        sys.exit(1)

    # Ensure group_id exists
    if "group_id" not in df.columns:
        df["group_id"] = [f"auto:{i}" for i in range(len(df))]

    # Expand labels text → one-hot
    if "labels" in df.columns:
        for l in LABELS:
            df[l] = df["labels"].apply(
                lambda s: 1 if any(canon(t) == l for t in str(s).split(",") if t.strip()) else 0
            )
    else:
        # Check if one-hot columns already exist
        missing = [l for l in LABELS if l not in df.columns]
        if missing:
            print(f"ERROR: no 'labels' column and missing one-hot columns: {missing}", file=sys.stderr)
            sys.exit(1)
        # Convert to int
        for l in LABELS:
            df[l] = pd.to_numeric(df[l], errors="coerce").fillna(0).astype(int)

    # Add missing metadata columns with defaults
    for col, default in [("dataset", "unknown"), ("region", "full_face"), ("split", "")]:
        if col not in df.columns:
            df[col] = default
    if "labels" not in df.columns:
        df["labels"] = df[LABELS].apply(
            lambda row: ",".join([l for l in LABELS if row[l] == 1]), axis=1
        )

    # Re-split if requested
    if args.resplit:
        print("Running stratified group split …")
        df = stratified_group_split(df, args.val, args.test, args.seed)

    # Summary
    print(f"\nSplit distribution: {df['split'].value_counts().to_dict()}")
    print("Label counts:")
    train = df[df["split"] == "train"] if "train" in df["split"].values else df
    for l in LABELS:
        pos = int(df[l].sum())
        train_pos = int(train[l].sum()) if len(train) else 0
        flag = "  ⚠  LOW" if train_pos < 30 else ""
        print(f"  {l:20s}: {pos:4d} total  ({train_pos} train){flag}")

    COLS = ["image_path", "labels", "group_id", "dataset", "region", "split"] + LABELS
    for c in COLS:
        if c not in df.columns:
            df[c] = ""

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
