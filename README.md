# RT-STORM-V2 / ST-STORM

This repository contains a unified pipeline for:
- self-supervised learning of style and content,
- supervised fine-tuning with frozen or unfrozen backbones,
- evaluation, inference, and webcam usage,
- multi-task classification,
- and object detection.

The general idea is to separate:
- **semantic content**: structure, shape, spatial layout,
- **style**: texture, contrast, frequency signatures, local and global appearance.

The project notably supports:
- a **generator / style** pathway,
- a **sem_resnet50 / semantic content** pathway,
- a **style + content fusion** mechanism with vector gating.

---

## 1. General structure

The main files are:
- `main.py`: training entry point
- `train_style_disentangle.py`: orchestration of the training modes
- `helpers.py`: core losses, supervised fine-tuning, and checkpoint saving
- `test.py`: evaluation, inference, webcam, t-SNE
- `config.py`: all CLI options
- `models/`: generator, discriminators, JEPA, heads, fusion
- `tests/functions_for_test.py`: model loading utilities for testing
- `scripts/`: ready-to-adapt examples

---

## 2. Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Depending on your use case, you may also need:
- a CUDA-compatible PyTorch installation for GPU training,
- OpenCV with a GUI backend if you want webcam display with OpenCV,
- Tkinter if OpenCV is headless but webcam display is still required.

---

## 3. Training modes

The central option is `--mode` in `main.py`.

### 3.1 `auto`

Full self-supervised learning.

This mode trains the style and content components without explicit class supervision.
Depending on the configuration, it combines:
- adversarial losses,
- token-based style consistency losses,
- FFT / SWD losses,
- Style-JEPA,
- MoCo content learning,
- PatchNCE,
- Content-NCE,
- guided reconstruction.

Use this mode to pretrain the model.

### 3.2 `hybrid`

Mixed learning:
- self-supervised A/B pretext phases,
- followed by a supervised phase C with `SupHeads`.

Use this mode when you want to combine self-supervision and supervised learning in the same run.

### 3.3 `sup_freeze`

Supervised fine-tuning with frozen feature extractors.

This mode mainly trains:
- `SupHeads`,
- `FusionHead` if `--sup_feat_source fusion` is used.

The feature extractors remain frozen. This mode is used to measure how reusable the learned representations are.

### 3.4 `sup_unfreeze`

End-to-end supervised fine-tuning.

This mode follows the same supervised logic as `sup_freeze`, but the feature extractors actually used to produce the features are unfrozen and updated using the supervised loss.

Depending on `--sup_feat_source`:
- `generator`: the style backbone used to produce `tok6`, `tokG`, `mapL`, etc. is fine-tuned;
- `sem_resnet50`: the semantic ResNet is fine-tuned;
- `fusion`: both feature extractors, plus `FusionHead`, are fine-tuned.

The fine-tuned extractors are explicitly saved so that they can be reloaded during testing.

### 3.5 `cls_tokens`

Supervised classification from multi-scale tokens.

### 3.6 `detect_transformer`

Object detection mode.

---

## 4. Supervised feature sources

The central option for supervised tasks is:

```bash
--sup_feat_source {generator,sem_resnet50,fusion}
```

### 4.1 `generator`

Features are extracted by the generator, usually through style representations.

Useful examples of `--sup_feat_type`:
- `tokG`
- `tok6`
- `tok6_mean`
- `tok6_w`
- `tokL`
- `tokL_mean`
- `tokL_w`
- `mapG`
- `mapL`
- `mapL_mean`
- `mapL_w`

### 4.2 `sem_resnet50`

Features come from the semantic ResNet backbone after spatial aggregation.

### 4.3 `fusion`

Style and semantic content features are fused through a lightweight module:
- normalization,
- projection into a common dimension,
- vector gating,
- followed by `SupHeads`.

---

## 5. Typical training commands

### 5.1 Self-supervised pretraining

```bash
python main.py \
  --mode auto \
  --data_json /path/to/train.json \
  --save_dir /path/to/output_auto \
  --epochs 40 \
  --k_folds 2 \
  --fold_epochs 4 \
  --tb --tb_freq 1000 \
  --jepa_tokens --jepa_on_style --jepa_on_content \
  --content_nce_enable \
  --sem_content \
  --tex_enable --tex_use_fft --tex_use_swd \
  --adv_enable_A
```

Strongly recommended:

```bash
python main.py --mode auto --data ""  --save_dir "/Baseline" --epochs 41 --k_folds 2 --fold_epochs 4 --tb --tb_freq 1000 --nce_layers "bot,skip64,skip32" --nce_layer_weights "1,1,1" --nce_intra --nce_inter --nce_max_patches 100 --lambda_nce_a_adv 0.2 --lambda_reg_a_adv 0.05 --lambda_nce_a_mix 0 --lambda_reg_a_mix 0.0 --skip_amix --tex_enable --tex_apply_A --tex_use_fft --tex_use_swd --lambda_fft 0.1 --lambda_swd 0.05 --tex_sigma 2 --tex_gamma 3 --adv_enable_A --adv_type lsgan --jepa_tokens --jepa_on_style --jepa_every 1  --lambda_jepa 2 --content_nce_enable --lambda_content_nce 10 --sem_content --lambda_sem 1 --sem_two_styles --sem_pretrained 1 --sem_pretrained_path r-50-1000ep.pth.tar --sem_pretrained_strict 1 --sem_queue 6000 --sem_use_aug --sem_crop 224 --sem_min_scale 0.5  --sem_color_jitter 0.4 --sem_gray 0.2 --sem_blur 0.1 --style_lambda 50 --style_lambda_min 5 --style_lambda_sched linear --sup_feat_type tok6

```

### 5.2 `sup_freeze` with style features

```bash
python main.py \
  --mode sup_freeze \
  --data /path/to/train \
  --save_dir /path/to/output_sup_freeze \
  --sup_feat_source generator \
  --sup_feat_type tok6 \
  --classes_json data/Tasks.json \
  --resume_dir /path/to/pretrain_style
```

### 5.3 `sup_freeze` with semantic ResNet

```bash
python main.py \
  --mode sup_freeze \
  --data /path/to/train \
  --save_dir /path/to/output_sup_freeze_sem \
  --sup_feat_source sem_resnet50 \
  --classes_json data/Tasks.json \
  --sem_pretrained_path /path/to/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

### 5.4 `sup_freeze` with fusion

```bash
python main.py \
  --mode sup_freeze \
  --data /path/to/train \
  --save_dir /path/to/output_sup_freeze_fusion \
  --sup_feat_source fusion \
  --sup_feat_type tok6 \
  --fusion_dim 1024 \
  --classes_json data/Tasks.json \
  --resume_dir /path/to/pretrain_style \
  --sem_pretrained_path /path/to/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

### 5.5 End-to-end `sup_unfreeze`

```bash
python main.py \
  --mode sup_unfreeze \
  --data /path/to/train \
  --save_dir /path/to/output_sup_unfreeze \
  --sup_feat_source fusion \
  --sup_feat_type tok6 \
  --fusion_dim 1024 \
  --classes_json data/Tasks.json \
  --resume_dir /path/to/pretrain_style \
  --sem_pretrained_path /path/to/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

---

## 6. ImageNet: training and testing

The project supports ImageNet CLS-LOC through `--data`, `--imagenet_split`, `--imagenet_ann_dir`, and `--imagenet_synset_mapping`.

### 6.1 Example: supervised ImageNet training

```bash
python main.py \
  --mode sup_freeze \
  --data /path/to/ILSVRC/Data/CLS-LOC/train \
  --imagenet_split train \
  --imagenet_synset_mapping /path/to/LOC_synset_mapping.txt \
  --imagenet_ann_dir /path/to/ILSVRC/Annotations/CLS-LOC/train \
  --save_dir /path/to/output_imagenet \
  --sup_feat_source sem_resnet50 \
  --classes_json data/Tasks.json
```

### 6.2 Example: ImageNet validation

```bash
python test.py \
  --mode sup_predict \
  --data /path/to/ILSVRC/Data/CLS-LOC/val \
  --imagenet_split val \
  --imagenet_ann_dir /path/to/ILSVRC/Annotations/CLS-LOC \
  --imagenet_synset_mapping /path/to/LOC_synset_mapping.txt \
  --weights_dir /path/to/fold_00 \
  --out_dir /path/to/fold_00/results_val \
  --cfg /path/to/train_cfg.json \
  --sup_ckpt /path/to/fold_00/SupHeads_last_epoch99.pt \
  --feature_mode sem_resnet50 \
  --sup_feat_source sem_resnet50 \
  --sem_pretrained_path /path/to/fold_00/SemBackbone_best_fold0.pt \
  --sem_pretrained_strict 1
```

For `fusion`, you also need:
- `FusionHead_best_fold0.pth` or `FusionHead_last_fold0.pth`,
- the fine-tuned generator,
- the fine-tuned `SemBackbone`.

---

## 7. Testing and evaluation modes

The entry point is `test.py`.

### 7.1 `sup_predict`

Supervised evaluation.

Typical outputs:
- `accuracy`
- `top1_accuracy`
- `precision`
- `recall`
- `f1`
- `confusion_matrix` if ground truth is available
- `submission_*.csv` if the split has no annotations

Example:

```bash
python test.py \
  --mode sup_predict \
  --cfg /path/to/train_cfg.json \
  --weights_dir /path/to/fold_00 \
  --sup_ckpt /path/to/fold_00/SupHeads_best_fold0.pth \
  --data /path/to/val \
  --feature_mode style \
  --embed_type tok6
```

### 7.2 `inference`

Pure inference, without metrics.

This mode only produces model predictions.
It also supports a simple flat folder of images through `--data`.

Example:

```bash
python test.py \
  --mode inference \
  --cfg /path/to/train_cfg.json \
  --weights_dir /path/to/fold_00 \
  --sup_ckpt /path/to/fold_00/SupHeads_best_fold0.pth \
  --data /path/to/images \
  --feature_mode style \
  --embed_type tok6 \
  --inference_save_csv
```

### 7.3 `backbone_camera`

Webcam / camera mode with live prediction display.

Example:

```bash
python test.py \
  --mode backbone_camera \
  --cfg /path/to/train_cfg.json \
  --weights_dir /path/to/fold_00 \
  --sup_ckpt /path/to/fold_00/SupHeads_best_fold0.pth \
  --feature_mode fusion \
  --sup_feat_source fusion \
  --embed_type tok6 \
  --classes_json data/Tasks.json
```

### 7.4 `tsne_interactive`

t-SNE visualization of learned representations.

### 7.5 `passe_by_metrics`

Exploratory evaluation with clustering metrics, KNN, separability, etc.

### 7.6 `style_transfer`

Style transfer visualization.

### 7.7 `detect_transformer`

Object detection evaluation or inference.

---

## 8. Important checkpoints to reload

### 8.1 In `sup_freeze`

During testing, reload the following depending on the case:
- always: `SupHeads_*.pth`
- if `generator`: `G_A_*.pt` / `G_B_*.pt`, depending on the backbone used
- if `sem_resnet50`: `SemBackbone_*.pt`
- if `fusion`:
  - `SupHeads_*.pth`
  - `FusionHead_*.pth`
  - `SemBackbone_*.pt`
  - `G_A_*.pt` / `G_B_*.pt`

### 8.2 In `sup_unfreeze`

The same logic applies, but you must reload the **fine-tuned** feature extractors, not older pretraining checkpoints.

---

## 9. Recommended ablations

The recommended protocol is:
- define a complete baseline,
- then remove one component at a time.

Priority ablations:
- without FFT
- without SWD
- without the full texture block
- without JEPA
- style-only JEPA
- content-only JEPA
- without adversarial learning
- without PatchNCE
- without Content-NCE
- without the semantic branch
- without reconstruction B

The comparison is then performed using:
- F1
- accuracy
- top-1
- precision / recall
- task-level metrics in the multi-task setting

---

## 10. `scripts/` folder

The repository now contains example scripts in:
- `scripts/train/`
- `scripts/test/`
- `scripts/ablations/`

Each script is a starting point that should be adapted to your local paths.

---

## 11. Available scripts

### Training

- `scripts/train/train_auto_example.sh`
- `scripts/train/train_sup_freeze_generator.sh`
- `scripts/train/train_sup_freeze_sem.sh`
- `scripts/train/train_sup_freeze_fusion.sh`
- `scripts/train/train_sup_unfreeze_fusion.sh`

### Testing

- `scripts/test/test_sup_predict_imagenet_val.sh`
- `scripts/test/test_inference_folder.sh`
- `scripts/test/test_backbone_camera_fusion.sh`
- `scripts/test/test_tsne_style.sh`

### Ablations

- `scripts/ablations/ablation_full.sh`
- `scripts/ablations/ablation_no_fft.sh`
- `scripts/ablations/ablation_no_swd.sh`
- `scripts/ablations/ablation_no_jepa.sh`
- `scripts/ablations/ablation_jepa_style_only.sh`
- `scripts/ablations/ablation_jepa_content_only.sh`
- `scripts/ablations/ablation_no_adv.sh`
- `scripts/ablations/ablation_no_patchnce.sh`
- `scripts/ablations/ablation_no_content_nce.sh`
- `scripts/ablations/ablation_no_semantic.sh`
- `scripts/ablations/ablation_no_recon_b.sh`

---

## 12. Practical advice

- In `sup_unfreeze`, make sure you are testing the fine-tuned extractors from the fold directory.
- In `fusion`, make sure that `FusionHead_*.pth` is correctly found and loaded.
- For ImageNet `val`, if XML annotations exist, provide `--imagenet_ann_dir` or let auto-detection handle it.
- For splits without annotations, testing switches to CSV submission mode.
- For webcam usage, if OpenCV is headless, the code may switch to Tkinter depending on the environment.

---

## 13. Quick summary

- `auto`: self-supervised pretraining
- `hybrid`: self-supervision + supervision
- `sup_freeze`: supervision with frozen extractors
- `sup_unfreeze`: end-to-end supervision
- `generator`: style features
- `sem_resnet50`: semantic features
- `fusion`: style + content
- `sup_predict`: evaluation
- `inference`: prediction without metrics
- `backbone_camera`: webcam

This README provides an overview. For reproducible usage, the simplest approach is to start from the scripts in the `scripts/` folder.
