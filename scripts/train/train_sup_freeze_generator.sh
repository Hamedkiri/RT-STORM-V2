#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/path/to/train}"
SAVE_DIR="${SAVE_DIR:-/path/to/output_sup_freeze_generator}"
RESUME_DIR="${RESUME_DIR:-/path/to/pretrain_style_run}"
CLASSES_JSON="${CLASSES_JSON:-data/Tasks.json}"
python main.py \
  --mode sup_freeze \
  --data "$DATA_ROOT" \
  --save_dir "$SAVE_DIR" \
  --sup_feat_source generator \
  --sup_feat_type tok6 \
  --classes_json "$CLASSES_JSON" \
  --resume_dir "$RESUME_DIR"
