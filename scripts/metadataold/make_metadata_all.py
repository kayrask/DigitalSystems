#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, re, random, sys
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except Exception:
    print("This script requires pandas. `pip install pandas`", file=sys.stderr)
    raise

# Optional stratified group split if sklearn is available
_USE_SK = True
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    _USE_SK = False

def norm(s): return re.sub(r'\s+', '_', str(s).strip().lower())

def collect_raw_a(root_a, exts, enforce_three_views=True):
    """
    raw_a layout:
      root_a/<label>/<subject_id>/{front.jpg,left-side.jpg,right-side.jpg}
    Returns rows with:
      - filepath
      - label (from top-level class folder)
      - group_id = f"a:{label}:{subject_id}"
      - dataset = 'a'
    """
    rows = []
    root = Path(root_a)
    expected = {"front","left-side","right-side"} if enforce_three_views else None

    # class folders: acne, redness, bags
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name.lower()):
        label = norm(cls_dir.name)
        # subject folders (0,1,2,...)
        for subj_dir in sorted([p for p in cls_dir.iterdir() if p.is_dir()], key=lambda p:p.name):
            sid = subj_dir.name.strip()
            gid = f"a:{label}:{sid}"
            for f in sorted(subj_dir.iterdir(), key=lambda x:x.name.lower()):
                if f.is_file() and f.suffix.lower() in exts:
                    stem = f.stem.strip().lower()
                    if expected is None or stem in expected:
                        rows.append({
                            "filepath": f.as_posix(),
                            "label": label,
                            "group_id": gid,
                            "dataset": "a"
                        })
    return rows

def read_label_map_b(label_map_csv):
    df = pd.read_csv(label_map_csv)
    df.columns = [norm(c) for c in df.columns]
    if not {"id","label"}.issubset(df.columns):
        raise KeyError(f"{label_map_csv} must have columns: id,label")
    df["id"] = df["id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return dict(zip(df["id"], df["label"]))

def collect_raw_b(root_b, label_map, exts, enforce_three_views=True):
    """
    raw_b layout:
      root_b/<subject_id>/{front.jpg,left-side.jpg,right-side.jpg}
    Returns rows with:
      - label from label_map[id]
      - group_id = f"b:{id}"
      - dataset = 'b'
    """
    rows = []
    root = Path(root_b)
    expected = {"front","left-side","right-side"} if enforce_three_views else None
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name):
        sid = subj_dir.name.strip()
        if sid not in label_map:
            continue
        label = norm(label_map[sid])
        gid = f"b:{sid}"
        for f in sorted(subj_dir.iterdir(), key=lambda x:x.name.lower()):
            if f.is_file() and f.suffix.lower() in exts:
                stem = f.stem.strip().lower()
                if expected is None or stem in expected:
                    rows.append({
                        "filepath": f.as_posix(),
                        "label": label,
                        "group_id": gid,
                        "dataset": "b"
                    })
    return rows

def stratified_group_split(rows, val_frac, test_frac, seed=42):
    rng = random.Random(seed)
    # majority label per group
    groups = {}
    for r in rows:
        gid = r["group_id"]; lab = r["label"]
        groups.setdefault(gid, Counter())[lab] += 1
    gid_list = list(groups.keys())
    gid_label = [groups[g].most_common(1)[0][0] for g in gid_list]
    n = len(gid_list)
    if n == 0:
        return []

    def assign(test_idx, val_idx):
        test_set, val_set = set(test_idx), set(val_idx)
        split = {}
        for i, g in enumerate(gid_list):
            if i in test_set: split[g] = "test"
            elif i in val_set: split[g] = "val"
            else: split[g] = "train"
        return [split[r["group_id"]] for r in rows]

    # Try SGKF 2-pass selection to approximate fractions
    if _USE_SK and (test_frac>0 or val_frac>0) and len(set(gid_label))>1 and n>=4:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        best_test, gap = set(), 1e9
        for tr, te in sgkf.split(gid_list, gid_label, groups=gid_list):
            for cand in (te, tr):
                g = abs(len(cand)/n - test_frac)
                if g < gap: best_test, gap = set(cand.tolist()), g
        remaining = sorted(set(range(n)) - best_test)
        if remaining and val_frac>0:
            rem_g = [gid_list[i] for i in remaining]
            rem_y = [gid_label[i] for i in remaining]
            target_val = val_frac / max(1e-9, (1 - test_frac))
            if len(set(rem_y))>1 and len(rem_g)>=2:
                sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed+1)
                best_val_rel, gap2 = set(), 1e9
                for tr2, te2 in sgkf2.split(rem_g, rem_y, groups=rem_g):
                    for cand in (te2, tr2):
                        g2 = abs(len(cand)/len(rem_g) - target_val)
                        if g2 < gap2: best_val_rel, gap2 = set(cand.tolist()), g2
                best_val = set(remaining[i] for i in sorted(best_val_rel))
            else:
                best_val = set()
        else:
            best_val = set()
        return assign(best_test, best_val)

    # Fallback: label-wise shuffle
    by_lbl = {}
    for g, y in zip(gid_list, gid_label):
        by_lbl.setdefault(y, []).append(g)
    for y in by_lbl: rng.shuffle(by_lbl[y])

    test_g = set(); val_g = set()
    for y, gs in by_lbl.items():
        k = len(gs)
        k_test = int(round(k * test_frac))
        k_val  = int(round(k * val_frac))
        test_g.update(gs[:k_test])
        val_g.update(gs[k_test:k_test+k_val])

    idx = {g:i for i,g in enumerate(gid_list)}
    return assign({idx[g] for g in test_g}, {idx[g] for g in val_g})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_a", required=True, help="Root of raw_a (class/subject/views)")
    ap.add_argument("--root_b", required=True, help="Root of raw_b (subject/views)")
    ap.add_argument("--label_map_b", required=True, help="CSV with columns id,label for raw_b")
    ap.add_argument("--out", required=True, help="Output merged CSV, e.g., data/processed/all.csv")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp")
    ap.add_argument("--allow_any_views", action="store_true",
                    help="If set, do NOT restrict to front/left-side/right-side")
    args = ap.parse_args()

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    enforce = not args.allow_any_views

    # Collect from A (class → subject → views)
    rows_a = collect_raw_a(args.root_a, exts, enforce_three_views=enforce)

    # Collect from B (subject → views, labels from map)
    label_map_b = read_label_map_b(args.label_map_b)
    rows_b = collect_raw_b(args.root_b, label_map_b, exts, enforce_three_views=enforce)

    rows = rows_a + rows_b
    if not rows:
        print("No images found in A or B.", file=sys.stderr); sys.exit(1)

    # Normalize labels to canonical names
    for r in rows:
        r["label"] = norm(r["label"])
        if r["label"] in {"under_eyes_bags","bags_under_eyes","bag","dark_circles"}:
            r["label"] = "bags"

    # Split (group-aware: a:<label>:<sid> or b:<sid>)
    splits = stratified_group_split(rows, args.val, args.test, args.seed)
    for r, s in zip(rows, splits):
        r["split"] = s

    # Write merged CSV
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath","label","group_id","dataset","split"])
        writer.writeheader(); writer.writerows(rows)

    # Summary
    print(f"Wrote {len(rows)} rows to {out}")
    print("Splits:", dict(Counter([r["split"] for r in rows])))
    print("Labels:", dict(Counter([r["label"] for r in rows])))
    print("By-dataset:", dict(Counter([r["dataset"] for r in rows])))

if __name__ == "__main__":
    main()
