#!/usr/bin/env bash
set -euo pipefail
CFG="${CFG:-/path/to/train_cfg.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/path/to/fold_00}"
DATA_DIR="${DATA_DIR:-/path/to/images_or_dataset}"
python test.py \
  --mode tsne_interactive \
  --cfg "$CFG" \
  --weights_dir "$WEIGHTS_DIR" \
  --data "$DATA_DIR" \
  --feature_mode style \
  --embed_type tok6
