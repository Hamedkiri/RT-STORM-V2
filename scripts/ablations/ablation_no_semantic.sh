#!/usr/bin/env bash
set -euo pipefail
SAVE_DIR="${SAVE_DIR:-/path/to/ablation_no_semantic}"
DATA_JSON="${DATA_JSON:-/path/to/train.json}"
python main.py --mode auto --data_json "$DATA_JSON" --save_dir "$SAVE_DIR" --epochs 41 --k_folds 2 --fold_epochs 4 --tb --tb_freq 1000 --nce_layers "bot,skip64,skip32" --nce_layer_weights "1,1,1" --nce_intra --nce_inter --nce_max_patches 100 --lambda_nce_a_adv 0.2 --lambda_reg_a_adv 0.05 --lambda_nce_a_mix 0 --lambda_reg_a_mix 0.0 --skip_amix --tex_enable --tex_apply_A --tex_use_fft --tex_use_swd --lambda_fft 0.1 --lambda_swd 0.05 --tex_sigma 2 --tex_gamma 3 --adv_enable_A --adv_type lsgan --jepa_tokens --jepa_on_style --jepa_on_content --lambda_jepa 0.5 --content_nce_enable --lambda_content_nce 10 --lambda_sem 0 --style_lambda 50 --style_lambda_min 5 --style_lambda_sched linear
