#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/path/to/train}"
SAVE_DIR="${SAVE_DIR:-/path/to/output_sup_freeze_sem}"
SEM_CKPT="${SEM_CKPT:-/path/to/SemBackbone_epoch199.pt}"
CLASSES_JSON="${CLASSES_JSON:-data/Tasks.json}"
python main.py \
  --mode sup_freeze \
  --data "$DATA_ROOT" \
  --save_dir "$SAVE_DIR" \
  --sup_feat_source sem_resnet50 \
  --classes_json "$CLASSES_JSON" \
  --sem_pretrained_path "$SEM_CKPT" \
  --sem_pretrained_strict 1
