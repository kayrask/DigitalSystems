#!/usr/bin/env python3
"""
Download a HuggingFace image dataset and convert it to AURAI's format.

STEP 1 — Inspect: run without --label_map to see what labels exist:
  python scripts/download_hf_dataset.py \
      --dataset "UniqueData/dermatology-dataset-acne-redness-and-bags-under-the-eyes" \
      --label_col "type"

STEP 2 — Download with a label mapping:

  # UniqueData — acne, redness, bags (3 views per person: front, left_side, right_side)
  python scripts/download_hf_dataset.py \
      --dataset "UniqueData/dermatology-dataset-acne-redness-and-bags-under-the-eyes" \
      --label_col "type" \
      --image_cols "front,left_side,right_side" \
      --out_dir "data/raw_hf/uniquedata" \
      --label_map '{"acne":"acne","skin_redness":"redness","bags_under_eyes":"bags"}'

  # UniDataPro — broader 15-class dataset (inspect first to see class names)
  python scripts/download_hf_dataset.py \
      --dataset "UniDataPro/facial-skin-condition-dataset" \
      --label_col "label" \
      --out_dir "data/raw_hf/unidatapro" \
      --label_map '{"acne":"acne","redness":"redness","blackhead":"blackheads"}'

  # AICamp-2023 — 4,465 images, MIT license, 23 classes (inspect first)
  python scripts/download_hf_dataset.py \
      --dataset "notable12/AICamp-2023-Skin-Conditions-Dataset" \
      --label_col "label" \
      --out_dir "data/raw_hf/aicamp"

STEP 3 — Merge with existing training data:
  python scripts/merge_datasets.py \
      --new_csv data/raw_hf/uniquedata/metadata.csv \
      --existing_csv data/processed/all_multilabel_onehot.csv \
      --out data/processed/all_multilabel_merged.csv

Confirmed existing HuggingFace datasets for AURAI (bags & redness are the weak classes):
  UniqueData/dermatology-dataset-acne-redness-and-bags-under-the-eyes  — acne, redness, bags
  UniDataPro/facial-skin-condition-dataset                              — 15-class facial conditions
  notable12/AICamp-2023-Skin-Conditions-Dataset                        — 23-class, MIT license
  Nexdata/Human_Facial_Skin_Defects_Data                               — acne, stains, dark circles
  khanusa/facial-skin-conditions                                        — acne + pigmentation
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

AURAI_LABELS = {"acne", "bags", "blackheads", "hyperpigmentation", "redness"}


def inspect_dataset(ds, label_col, n_inspect=500):
    """Print sample label values so the user can build their label_map."""
    label_counts = Counter()
    for i, row in enumerate(ds):
        if i >= n_inspect:
            break
        val = row.get(label_col)
        if val is None:
            continue
        # Handle integer class indices (HF feature classes)
        label_counts[str(val)] += 1
    print(f"\nLabel values in first {n_inspect} rows:")
    for lab, cnt in label_counts.most_common():
        print(f"  {lab!r:30s}  ({cnt} samples)")

    # Try to get class names if label is an int feature
    try:
        feature = ds.features[label_col]
        if hasattr(feature, "names"):
            print(f"\nClass names from feature metadata:")
            for i, name in enumerate(feature.names):
                print(f"  {i}: {name}")
    except Exception:
        pass

    print(f"\nRe-run with --label_map to map these values to AURAI labels.")
    print(f"Valid AURAI labels: {sorted(AURAI_LABELS)}")
    print(f"Use null in the JSON map to skip a label class.")


def main():
    ap = argparse.ArgumentParser(description="Download and convert a HuggingFace skin dataset.")
    ap.add_argument("--dataset", required=True, help="HuggingFace dataset ID, e.g. 'trainingdatapro/skin-diseases-face-images'")
    ap.add_argument("--subset", default=None, help="Dataset config/subset (leave blank for default)")
    ap.add_argument("--hf_split", default="train", help="HF split to download: train / test / validation / all (default: train)")
    ap.add_argument("--image_col", default="image", help="Single image column name (default: 'image'). Use --image_cols for multi-view datasets.")
    ap.add_argument("--image_cols", default=None, help="Comma-separated image column names for multi-view datasets, e.g. 'front,left_side,right_side'. Each view becomes a separate row.")
    ap.add_argument("--label_col", default="label", help="Column containing labels (default: 'label')")
    ap.add_argument("--label_map", default=None,
                    help="JSON dict mapping dataset labels → AURAI labels. null values skip that class. "
                         "Omit this flag to run in inspect mode first.")
    ap.add_argument("--out_dir", default=None, help="Output dir for images + metadata.csv (required when --label_map provided)")
    ap.add_argument("--max_samples", type=int, default=0, help="Limit samples per label class (0 = no limit)")
    ap.add_argument("--min_size", type=int, default=80, help="Skip images smaller than this in either dimension (default: 80)")
    ap.add_argument("--jpg_quality", type=int, default=92, help="JPEG quality when saving images (default: 92)")
    args = ap.parse_args()

    # -- Check deps --
    try:
        from datasets import load_dataset, DatasetDict, concatenate_datasets
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pip install pandas", file=sys.stderr)
        sys.exit(1)
    try:
        from PIL import Image
        import io
    except ImportError:
        print("ERROR: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    # -- Parse label map --
    label_map = None
    if args.label_map:
        label_map = json.loads(args.label_map)
        # Normalise keys to lowercase+underscore
        label_map = {k.strip().lower().replace(" ", "_"): v for k, v in label_map.items()}
        for k, v in label_map.items():
            if v is not None and v not in AURAI_LABELS:
                print(f"WARNING: '{v}' is not a valid AURAI label. Valid: {sorted(AURAI_LABELS)}", file=sys.stderr)

    # -- Load dataset --
    # Resolve image columns
    image_cols = [c.strip() for c in args.image_cols.split(",")] if args.image_cols else [args.image_col]

    print(f"Loading '{args.dataset}' (split={args.hf_split}) …")
    try:
        load_kwargs = {}
        if args.subset:
            load_kwargs["name"] = args.subset
        if args.hf_split == "all":
            ds_raw = load_dataset(args.dataset, verification_mode="no_checks", **load_kwargs)
        else:
            ds_raw = load_dataset(args.dataset, split=args.hf_split, verification_mode="no_checks", **load_kwargs)
    except Exception as e:
        print(f"ERROR loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Concatenate all splits if "all" was requested
    if isinstance(ds_raw, DatasetDict):
        ds = concatenate_datasets(list(ds_raw.values()))
    else:
        ds = ds_raw

    print(f"Columns: {ds.column_names}")
    print(f"Total samples: {len(ds)}")

    if args.label_col not in ds.column_names:
        print(f"ERROR: label column '{args.label_col}' not found. Available: {ds.column_names}", file=sys.stderr)
        print("Use --label_col to specify the correct column name.", file=sys.stderr)
        sys.exit(1)

    # -- Inspect mode (no label_map) — show labels and exit before checking image cols --
    if label_map is None:
        print(f"\nAvailable columns: {ds.column_names}")
        inspect_dataset(ds, args.label_col)
        return

    # -- Download mode — validate image columns now --
    for ic in image_cols:
        if ic not in ds.column_names:
            print(f"ERROR: image column '{ic}' not found. Available: {ds.column_names}", file=sys.stderr)
            print("Use --image_col or --image_cols to specify the correct column name(s).", file=sys.stderr)
            sys.exit(1)

    if not args.out_dir:
        print("ERROR: --out_dir is required when --label_map is provided.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Per-class counter for max_samples enforcement
    class_counts: Counter = Counter()

    rows = []
    skipped_label = 0
    skipped_size = 0
    skipped_err = 0
    total = len(ds)

    # Get class names for integer labels
    int_to_name = {}
    try:
        feature = ds.features[args.label_col]
        if hasattr(feature, "names"):
            int_to_name = {i: name for i, name in enumerate(feature.names)}
    except Exception:
        pass

    print(f"Downloading and converting {total} samples …")
    for i in range(total):
        row = ds[i]

        # -- Resolve label --
        raw = row[args.label_col]
        # Handle integer class index
        if isinstance(raw, int) and int_to_name:
            raw = int_to_name.get(raw, str(raw))
        raw_norm = str(raw).strip().lower().replace(" ", "_")
        aurai_label = label_map.get(raw_norm)
        if aurai_label is None:
            skipped_label += 1
            continue

        # Enforce per-class limit
        if args.max_samples > 0 and class_counts[aurai_label] >= args.max_samples:
            continue

        # -- Process each image column (supports multi-view: front, left_side, right_side) --
        for col_idx, img_col in enumerate(image_cols):
            img_data = row.get(img_col)
            if img_data is None:
                skipped_err += 1
                continue
            try:
                if hasattr(img_data, "convert"):
                    img = img_data.convert("RGB")
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                else:
                    img = Image.fromarray(img_data).convert("RGB")
            except Exception:
                skipped_err += 1
                continue

            if img.width < args.min_size or img.height < args.min_size:
                skipped_size += 1
                continue

            fname = f"{i:07d}_{col_idx}.jpg"
            fpath = img_dir / fname
            try:
                img.save(fpath, "JPEG", quality=args.jpg_quality)
            except Exception:
                skipped_err += 1
                continue

            class_counts[aurai_label] += 1
            rows.append({
                "image_path": fpath.as_posix(),
                "labels": aurai_label,
                # Same group_id for all views of same subject → kept together in splits
                "group_id": f"hf:{out_dir.name}:{i}",
                "dataset": out_dir.name,
                "region": "full_face",
                "split": "",
                "acne":              1 if aurai_label == "acne" else 0,
                "bags":              1 if aurai_label == "bags" else 0,
                "blackheads":        1 if aurai_label == "blackheads" else 0,
                "hyperpigmentation": 1 if aurai_label == "hyperpigmentation" else 0,
                "redness":           1 if aurai_label == "redness" else 0,
            })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}  kept={len(rows)}  skipped_label={skipped_label}  skipped_size={skipped_size}  skipped_err={skipped_err}")

    # -- Report --
    print(f"\nFinished.")
    print(f"  Kept:             {len(rows)}")
    print(f"  Skipped (label):  {skipped_label}  (not in label_map or mapped to null)")
    print(f"  Skipped (size):   {skipped_size}  (image too small, < {args.min_size}px)")
    print(f"  Skipped (error):  {skipped_err}  (corrupt / unreadable)")
    print(f"\nLabel breakdown:")
    for lab in sorted(AURAI_LABELS):
        print(f"  {lab}: {class_counts[lab]}")

    if not rows:
        print("\nERROR: No rows saved. Check your --label_map values.", file=sys.stderr)
        sys.exit(1)

    # -- Save metadata CSV --
    import pandas as pd
    df = pd.DataFrame(rows)
    meta_path = out_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved to: {meta_path}")
    print(f"Images saved to:   {img_dir}/")

    print(f"\n{'='*60}")
    print(f"NEXT STEP — merge with existing training data:")
    print(f"  python scripts/merge_datasets.py \\")
    print(f"      --new_csv {meta_path} \\")
    print(f"      --existing_csv data/processed/all_multilabel_onehot.csv \\")
    print(f"      --out data/processed/all_multilabel_merged.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
