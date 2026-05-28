#!/usr/bin/env bash
set -euo pipefail
CFG="${CFG:-/path/to/train_cfg.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/path/to/fold_00}"
SUP_CKPT="${SUP_CKPT:-/path/to/fold_00/SupHeads_best_fold0.pth}"
SEM_CKPT="${SEM_CKPT:-/path/to/fold_00/SemBackbone_best_fold0.pt}"
CLASSES_JSON="${CLASSES_JSON:-data/Tasks.json}"
python test.py \
  --mode backbone_camera \
  --cfg "$CFG" \
  --weights_dir "$WEIGHTS_DIR" \
  --sup_ckpt "$SUP_CKPT" \
  --sem_pretrained_path "$SEM_CKPT" \
  --sem_pretrained_strict 1 \
  --feature_mode fusion \
  --sup_feat_source fusion \
  --embed_type tok6 \
  --classes_json "$CLASSES_JSON"
