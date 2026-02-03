#!/bin/bash
set -e
# Activate venv
source .venv/bin/activate

# Go to the folder where this script lives
cd "$(dirname "$0")"



# Build unified metadata CSV
python make_metadata_all_multilabel.py \
  --root_a data/raw_a \
  --root_b data/raw_b \
  --label_map_b data/raw_b/labels_to_fill.csv \
  --patch_root data/patches \
  --out data/processed/all_multilabel.csv
