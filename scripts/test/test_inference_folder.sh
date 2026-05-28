#!/usr/bin/env bash
set -euo pipefail
CFG="${CFG:-/path/to/train_cfg.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/path/to/fold_00}"
SUP_CKPT="${SUP_CKPT:-/path/to/fold_00/SupHeads_best_fold0.pth}"
DATA_DIR="${DATA_DIR:-/path/to/images}"
CLASSES_JSON="${CLASSES_JSON:-data/Tasks.json}"
python test.py \
  --mode inference \
  --cfg "$CFG" \
  --weights_dir "$WEIGHTS_DIR" \
  --sup_ckpt "$SUP_CKPT" \
  --data "$DATA_DIR" \
  --feature_mode style \
  --embed_type tok6 \
  --classes_json "$CLASSES_JSON" \
  --inference_save_csv
