#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, re, random, sys
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except Exception:
    print("Requires pandas: pip install pandas", file=sys.stderr); raise

# Optional stratified group split
_USE_SK = True
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    _USE_SK = False

CANON = {
    "acne":"acne", "redness":"redness", "erythema":"redness",
    "bags":"bags", "under_eyes_bags":"bags", "bags_under_eyes":"bags", "dark_circles":"bags",
    "blackhead":"blackheads", "blackheads":"blackheads", "comedones":"blackheads",
    "hyperpigmentation":"hyperpigmentation", "melasma":"hyperpigmentation",
    "keloid":"keloids", "keloids":"keloids",
    "wrinkle":"wrinkles", "wrinkles":"wrinkles", "acnes": "acne",
    "darkspots": "hyperpigmentation",
    "dark_spots": "hyperpigmentation",
    "blackheads": "blackheads",
    "blackhead": "blackheads",
    # we'll skip wrinkles later
    }

SKIP_LABELS = {"wrinkles"}  # you said no wrinkles

def norm(s): return re.sub(r'\s+','_',str(s).strip().lower())
def canon_label(s):
    s = norm(s)
    return CANON.get(s, s)

def collect_raw_a(root_a, exts, enforce_three=True):
    rows = []
    root = Path(root_a)
    expected = {"front","left-side","right-side"} if enforce_three else None
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name.lower()):
        lab = canon_label(cls_dir.name)
        if lab in SKIP_LABELS:  # skip wrinkles
            continue
        for subj_dir in sorted([p for p in cls_dir.iterdir() if p.is_dir()], key=lambda p:p.name):
            sid = subj_dir.name.strip()
            gid = f"a:{lab}:{sid}"
            for f in sorted(subj_dir.iterdir(), key=lambda x:x.name.lower()):
                if f.is_file() and f.suffix.lower() in exts:
                    stem = norm(f.stem)
                    if expected is None or stem in expected:
                        rows.append({
                            "filepath": f.as_posix(),
                            "labels": lab,              # single tag still fine in multi-label
                            "group_id": gid,
                            "dataset": "a_fullface",
                            "region": "full_face",
                        })
    return rows

def read_label_map_b(label_map_csv):
    df = pd.read_csv(label_map_csv)
    df.columns = [norm(c) for c in df.columns]
    if not {"id","label"}.issubset(df.columns):
        raise KeyError(f"{label_map_csv} must have columns: id,label")
    df["id"] = df["id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    mapping = {}
    for _, r in df.iterrows():
        lab = canon_label(r["label"])
        if lab in SKIP_LABELS:  # skip wrinkles
            continue
        mapping[r["id"]] = lab
    return mapping

def collect_raw_b(root_b, label_map, exts, enforce_three=True):
    rows = []
    root = Path(root_b)
    expected = {"front","left-side","right-side"} if enforce_three else None
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name):
        sid = subj_dir.name.strip()
        if sid not in label_map:  # unlabeled -> skip
            continue
        lab = label_map[sid]
        if lab in SKIP_LABELS:
            continue
        gid = f"b:{sid}"
        for f in sorted(subj_dir.iterdir(), key=lambda x:x.name.lower()):
            if f.is_file() and f.suffix.lower() in exts:
                stem = norm(f.stem)
                if expected is None or stem in expected:
                    rows.append({
                        "filepath": f.as_posix(),
                        "labels": lab,
                        "group_id": gid,
                        "dataset": "b_fullface",
                        "region": "full_face",
                    })
    return rows

def collect_patch_dataset(root_patches, exts, region_hint=None):
    """
    Expects: root_patches/<label>/**/*.jpg (or a single label folder)
    Each file becomes one row (group_id per image).
    region_hint: set a region like 'nose' or 'cheek' for the whole tree.
    """
    rows = []
    if root_patches is None: return rows
    root = Path(root_patches)
    if not root.exists(): return rows

    # either label folders, or a flat folder with implied label from root name
    label_dirs = [p for p in root.iterdir() if p.is_dir()]
    if label_dirs:
        for ld in sorted(label_dirs, key=lambda p:p.name.lower()):
            lab = canon_label(ld.name)
            if lab in SKIP_LABELS: continue
            for f in sorted(ld.rglob("*"), key=lambda x:x.as_posix().lower()):
                if f.is_file() and f.suffix.lower() in exts:
                    gid = f"p:{lab}:{norm(f.stem)}"
                    rows.append({
                        "filepath": f.as_posix(),
                        "labels": lab,
                        "group_id": gid,
                        "dataset": "patches",
                        "region": region_hint or "unknown_region",
                    })
    else:
        # treat root name as label
        lab = canon_label(root.name)
        if lab not in SKIP_LABELS:
            for f in sorted(root.rglob("*"), key=lambda x:x.as_posix().lower()):
                if f.is_file() and f.suffix.lower() in exts:
                    gid = f"p:{lab}:{norm(f.stem)}"
                    rows.append({
                        "filepath": f.as_posix(),
                        "labels": lab,
                        "group_id": gid,
                        "dataset": "patches",
                        "region": region_hint or "unknown_region",
                    })
    return rows

def stratified_group_split(rows, val_frac, test_frac, seed=42):
    rng = random.Random(seed)
    # Majority label per group (multi-label rows use single label strings here)
    groups = {}
    for r in rows:
        g = r["group_id"]; labs = [s.strip() for s in str(r["labels"]).split(",") if s.strip()]
        lab = labs[0] if labs else "unknown"
        groups.setdefault(g, Counter())[lab] += 1
    gid = list(groups.keys())
    y   = [groups[g].most_common(1)[0][0] for g in gid]
    n = len(gid)
    if n == 0: return []

    def assign(test_idx, val_idx):
        test_set, val_set = set(test_idx), set(val_idx)
        split = {}
        for i, g in enumerate(gid):
            if i in test_set: split[g] = "test"
            elif i in val_set: split[g] = "val"
            else: split[g] = "train"
        return [split[r["group_id"]] for r in rows]

    if _USE_SK and (test_frac>0 or val_frac>0) and len(set(y))>1 and n>=4:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        best_test, gap = set(), 1e9
        for tr, te in sgkf.split(gid, y, groups=gid):
            for cand in (te, tr):
                g = abs(len(cand)/n - test_frac)
                if g < gap: best_test, gap = set(cand.tolist()), g
        remaining = sorted(set(range(n)) - best_test)
        if remaining and val_frac>0:
            rem_g = [gid[i] for i in remaining]
            rem_y = [y[i] for i in remaining]
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

    # fallback shuffle by label buckets
    by_lbl = {}
    for g, lab in zip(gid, y):
        by_lbl.setdefault(lab, []).append(g)
    for lab in by_lbl: rng.shuffle(by_lbl[lab])
    test_g = set(); val_g = set()
    for lab, gs in by_lbl.items():
        k = len(gs); k_test = int(round(k*test_frac)); k_val = int(round(k*val_frac))
        test_g.update(gs[:k_test]); val_g.update(gs[k_test:k_test+k_val])
    idx = {g:i for i,g in enumerate(gid)}
    return assign({idx[g] for g in test_g}, {idx[g] for g in val_g})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_a", required=True, help="data/raw_a")
    ap.add_argument("--root_b", required=True, help="data/raw_b")
    ap.add_argument("--label_map_b", required=True, help="CSV with id,label for raw_b")
    ap.add_argument("--patch_root", default=None, help="Root of patch dataset(s). Can contain label folders.")
    ap.add_argument("--patch_region", default=None, help="Region tag for patches (e.g., nose, cheek, under_eye).")
    ap.add_argument("--out", required=True, help="Output CSV, e.g., data/processed/all_multilabel.csv")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp")
    ap.add_argument("--allow_any_views", action="store_true", help="If set, do NOT restrict to front/left-side/right-side")
    args = ap.parse_args()

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    enforce = not args.allow_any_views

    rows = []
    rows += collect_raw_a(args.root_a, exts, enforce_three=enforce)
    label_map_b = read_label_map_b(args.label_map_b)
    rows += collect_raw_b(args.root_b, label_map_b, exts, enforce_three=enforce)
    rows += collect_patch_dataset(args.patch_root, exts, region_hint=args.patch_region)

    if not rows:
        print("No images found.", file=sys.stderr); sys.exit(1)

    # Split groups without leakage
    splits = stratified_group_split(rows, args.val, args.test, args.seed)
    for r, s in zip(rows, splits): r["split"] = s

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath","labels","group_id","dataset","region","split"])
        writer.writeheader(); writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")
    print("Splits:", dict(Counter([r["split"] for r in rows])))
    # quick label inventory
    labs = Counter()
    for r in rows:
        for t in [x for x in str(r["labels"]).split(",") if x.strip()]:
            labs[t.strip()] += 1
    print("Labels:", dict(labs))

if __name__ == "__main__":
    main()
