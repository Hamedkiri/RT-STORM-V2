#!/usr/bin/env bash
set -euo pipefail
SAVE_DIR="${SAVE_DIR:-/path/to/ablation_no_jepa}"
DATA_JSON="${DATA_JSON:-/path/to/train.json}"
SEM_CKPT="${SEM_CKPT:-/path/to/sem_pretrained.pth.tar}"
python main.py --mode auto --data_json "$DATA_JSON" --save_dir "$SAVE_DIR" --epochs 41 --k_folds 2 --fold_epochs 4 --tb --tb_freq 1000 --nce_layers "bot,skip64,skip32" --nce_layer_weights "1,1,1" --nce_intra --nce_inter --nce_max_patches 100 --lambda_nce_a_adv 0.2 --lambda_reg_a_adv 0.05 --lambda_nce_a_mix 0 --lambda_reg_a_mix 0.0 --skip_amix --tex_enable --tex_apply_A --tex_use_fft --tex_use_swd --lambda_fft 0.1 --lambda_swd 0.05 --tex_sigma 2 --tex_gamma 3 --adv_enable_A --adv_type lsgan --lambda_jepa 0 --lambda_jepa_style 0 --lambda_jepa_content 0 --lambda_jepa_var 0 --lambda_jepa_cov 0 --lambda_jepa_kd 0 --content_nce_enable --lambda_content_nce 10 --sem_content --lambda_sem 1 --sem_two_styles --sem_pretrained 1 --sem_pretrained_path "$SEM_CKPT" --sem_pretrained_strict 1 --sem_queue 6000 --sem_use_aug --sem_crop 224 --sem_min_scale 0.5 --sem_color_jitter 0.4 --sem_gray 0.2 --sem_blur 0.1 --style_lambda 50 --style_lambda_min 5 --style_lambda_sched linear
