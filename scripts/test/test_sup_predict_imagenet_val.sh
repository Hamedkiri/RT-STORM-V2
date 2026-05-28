#!/usr/bin/env bash
set -euo pipefail
CFG="${CFG:-/path/to/train_cfg.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/path/to/fold_00}"
SUP_CKPT="${SUP_CKPT:-/path/to/fold_00/SupHeads_best_fold0.pth}"
DATA_VAL="${DATA_VAL:-/path/to/ILSVRC/Data/CLS-LOC/val}"
ANN_DIR="${ANN_DIR:-/path/to/ILSVRC/Annotations/CLS-LOC}"
SYNSET_MAP="${SYNSET_MAP:-/path/to/LOC_synset_mapping.txt}"
OUT_DIR="${OUT_DIR:-/path/to/fold_00/results_val}"
SEM_CKPT="${SEM_CKPT:-/path/to/fold_00/SemBackbone_best_fold0.pt}"
python test.py \
  --mode sup_predict \
  --cfg "$CFG" \
  --weights_dir "$WEIGHTS_DIR" \
  --sup_ckpt "$SUP_CKPT" \
  --data "$DATA_VAL" \
  --imagenet_split val \
  --imagenet_ann_dir "$ANN_DIR" \
  --imagenet_synset_mapping "$SYNSET_MAP" \
  --out_dir "$OUT_DIR" \
  --feature_mode sem_resnet50 \
  --sup_feat_source sem_resnet50 \
  --sem_pretrained_path "$SEM_CKPT" \
  --sem_pretrained_strict 1 \
  --dump_param_count_json
