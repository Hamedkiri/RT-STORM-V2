# RT-STORM-V2 / ST-STORM

Ce dépôt contient un pipeline unifié pour :
- l'apprentissage auto-supervisé du style et du contenu,
- le fine-tuning supervisé avec backbones gelés ou dégelés,
- l'évaluation, l'inférence et la webcam,
- la classification multi-tâches,
- et la détection d'objets.

L'idée générale est de séparer :
- le **contenu sémantique** : structure, forme, disposition spatiale,
- le **style** : texture, contraste, signatures fréquentielles, apparence locale et globale.

Le projet supporte notamment :
- un chemin **générateur / style**,
- un chemin **sem_resnet50 / contenu sémantique**,
- une **fusion style + contenu** avec gating vectoriel.

---

## 1. Structure générale

Les fichiers principaux sont :
- `main.py` : point d'entrée entraînement
- `train_style_disentangle.py` : orchestration des modes d'entraînement
- `helpers.py` : cœur des losses, du fine-tuning supervisé et des sauvegardes
- `test.py` : évaluation, inférence, webcam, t-SNE
- `config.py` : ensemble des options CLI
- `models/` : générateur, discriminateurs, JEPA, heads, fusion
- `tests/functions_for_test.py` : chargement des modèles côté test
- `scripts/` : exemples prêts à adapter

---

## 2. Installation

Créer un environnement Python puis installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Selon les usages, il faut aussi disposer de :
- PyTorch compatible CUDA si entraînement GPU,
- OpenCV avec backend GUI si webcam avec affichage OpenCV,
- Tkinter si OpenCV est headless mais qu'un affichage webcam est souhaité.

---

## 3. Modes d'entraînement

L'option centrale est `--mode` dans `main.py`.

### 3.1 `auto`
Apprentissage auto-supervisé complet.

Ce mode entraîne les composants de style et de contenu sans supervision explicite de classes.
Il combine selon la configuration :
- pertes adversariales,
- pertes de cohérence de style par tokens,
- pertes FFT / SWD,
- Style-JEPA,
- MoCo contenu,
- PatchNCE,
- Content-NCE,
- reconstruction guidée.

À utiliser pour préentraîner le modèle.

### 3.2 `hybrid`
Apprentissage mixte :
- prétexte auto-supervisé A/B,
- puis phase supervisée C avec `SupHeads`.

À utiliser si l'on veut combiner self-supervision et apprentissage supervisé dans le même run.

### 3.3 `sup_freeze`
Fine-tuning supervisé avec extracteurs gelés.

Ce mode entraîne principalement :
- `SupHeads`,
- `FusionHead` si `--sup_feat_source fusion`.

Les extracteurs de features restent gelés. Ce mode sert à mesurer la réutilisabilité des représentations apprises.

### 3.4 `sup_unfreeze`
Fine-tuning supervisé end-to-end.

Même logique supervisée que `sup_freeze`, mais les extracteurs réellement utilisés pour produire les features sont dégelés et mis à jour par la perte supervisée.

Selon `--sup_feat_source` :
- `generator` : le backbone style utilisé pour produire `tok6`, `tokG`, `mapL`, etc. est finetuné ;
- `sem_resnet50` : le ResNet sémantique est finetuné ;
- `fusion` : les deux extracteurs, plus `FusionHead`, sont finetunés.

Les extracteurs finetunés sont sauvegardés explicitement pour être rechargés au test.

### 3.5 `cls_tokens`
Classification supervisée à partir de tokens multi-échelles.

### 3.6 `detect_transformer`
Mode détection d'objets.

---

## 4. Sources de features supervisées

L'option centrale pour les tâches supervisées est :

```bash
--sup_feat_source {generator,sem_resnet50,fusion}
```

### 4.1 `generator`
Les features sont extraites par le générateur, typiquement via des représentations de style.

Exemples de `--sup_feat_type` utiles :
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
Les features proviennent du backbone sémantique ResNet, après agrégation spatiale.

### 4.3 `fusion`
Les features de style et de contenu sémantique sont fusionnées via un module léger :
- normalisation,
- projection dans une dimension commune,
- gating vectoriel,
- `SupHeads` derrière.

---

## 5. Commandes d'entraînement typiques

### 5.1 Préentraînement auto-supervisé

```bash
python main.py \
  --mode auto \
  --data_json /chemin/train.json \
  --save_dir /chemin/output_auto \
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

Fortement recommandé : 

```bash
python main.py --mode auto --data ""  --save_dir "/Baseline" --epochs 41 --k_folds 2 --fold_epochs 4 --tb --tb_freq 1000 --nce_layers "bot,skip64,skip32" --nce_layer_weights "1,1,1" --nce_intra --nce_inter --nce_max_patches 100 --lambda_nce_a_adv 0.2 --lambda_reg_a_adv 0.05 --lambda_nce_a_mix 0 --lambda_reg_a_mix 0.0 --skip_amix --tex_enable --tex_apply_A --tex_use_fft --tex_use_swd --lambda_fft 0.1 --lambda_swd 0.05 --tex_sigma 2 --tex_gamma 3 --adv_enable_A --adv_type lsgan --jepa_tokens --jepa_on_style --jepa_every 1  --lambda_jepa 2 --content_nce_enable --lambda_content_nce 10 --sem_content --lambda_sem 1 --sem_two_styles --sem_pretrained 1 --sem_pretrained_path r-50-1000ep.pth.tar --sem_pretrained_strict 1 --sem_queue 6000 --sem_use_aug --sem_crop 224 --sem_min_scale 0.5  --sem_color_jitter 0.4 --sem_gray 0.2 --sem_blur 0.1 --style_lambda 50 --style_lambda_min 5 --style_lambda_sched linear --sup_feat_type tok6

```

### 5.2 `sup_freeze` avec style

```bash
python main.py \
  --mode sup_freeze \
  --data /chemin/train \
  --save_dir /chemin/output_sup_freeze \
  --sup_feat_source generator \
  --sup_feat_type tok6 \
  --classes_json data/Tasks.json \
  --resume_dir /chemin/pretrain_style
```

### 5.3 `sup_freeze` avec ResNet sémantique

```bash
python main.py \
  --mode sup_freeze \
  --data /chemin/train \
  --save_dir /chemin/output_sup_freeze_sem \
  --sup_feat_source sem_resnet50 \
  --classes_json data/Tasks.json \
  --sem_pretrained_path /chemin/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

### 5.4 `sup_freeze` avec fusion

```bash
python main.py \
  --mode sup_freeze \
  --data /chemin/train \
  --save_dir /chemin/output_sup_freeze_fusion \
  --sup_feat_source fusion \
  --sup_feat_type tok6 \
  --fusion_dim 1024 \
  --classes_json data/Tasks.json \
  --resume_dir /chemin/pretrain_style \
  --sem_pretrained_path /chemin/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

### 5.5 `sup_unfreeze` end-to-end

```bash
python main.py \
  --mode sup_unfreeze \
  --data /chemin/train \
  --save_dir /chemin/output_sup_unfreeze \
  --sup_feat_source fusion \
  --sup_feat_type tok6 \
  --fusion_dim 1024 \
  --classes_json data/Tasks.json \
  --resume_dir /chemin/pretrain_style \
  --sem_pretrained_path /chemin/SemBackbone_epoch199.pt \
  --sem_pretrained_strict 1
```

---

## 6. ImageNet : entraînement et test

Le projet gère ImageNet CLS-LOC via `--data`, `--imagenet_split`, `--imagenet_ann_dir`, `--imagenet_synset_mapping`.

### 6.1 Exemple entraînement supervisé ImageNet

```bash
python main.py \
  --mode sup_freeze \
  --data /chemin/ILSVRC/Data/CLS-LOC/train \
  --imagenet_split train \
  --imagenet_synset_mapping /chemin/LOC_synset_mapping.txt \
  --imagenet_ann_dir /chemin/ILSVRC/Annotations/CLS-LOC/train \
  --save_dir /chemin/output_imagenet \
  --sup_feat_source sem_resnet50 \
  --classes_json data/Tasks.json
```

### 6.2 Exemple validation ImageNet

```bash
python test.py \
  --mode sup_predict \
  --data /chemin/ILSVRC/Data/CLS-LOC/val \
  --imagenet_split val \
  --imagenet_ann_dir /chemin/ILSVRC/Annotations/CLS-LOC \
  --imagenet_synset_mapping /chemin/LOC_synset_mapping.txt \
  --weights_dir /chemin/fold_00 \
  --out_dir /chemin/fold_00/results_val \
  --cfg /chemin/train_cfg.json \
  --sup_ckpt /chemin/fold_00/SupHeads_last_epoch99.pt \
  --feature_mode sem_resnet50 \
  --sup_feat_source sem_resnet50 \
  --sem_pretrained_path /chemin/fold_00/SemBackbone_best_fold0.pt \
  --sem_pretrained_strict 1
```

Pour `fusion`, il faut en plus :
- `FusionHead_best_fold0.pth` ou `FusionHead_last_fold0.pth`,
- le générateur finetuné,
- le `SemBackbone` finetuné.

---

## 7. Modes de test et d'évaluation

Le point d'entrée est `test.py`.

### 7.1 `sup_predict`
Évaluation supervisée.

Sorties typiques :
- `accuracy`
- `top1_accuracy`
- `precision`
- `recall`
- `f1`
- `confusion_matrix` si GT disponibles
- `submission_*.csv` si split sans annotations

Exemple :

```bash
python test.py \
  --mode sup_predict \
  --cfg /chemin/train_cfg.json \
  --weights_dir /chemin/fold_00 \
  --sup_ckpt /chemin/fold_00/SupHeads_best_fold0.pth \
  --data /chemin/val \
  --feature_mode style \
  --embed_type tok6
```

### 7.2 `inference`
Inférence pure, sans métriques.

Ce mode produit seulement les prédictions du modèle.
Il supporte aussi un simple dossier d'images plat via `--data`.

Exemple :

```bash
python test.py \
  --mode inference \
  --cfg /chemin/train_cfg.json \
  --weights_dir /chemin/fold_00 \
  --sup_ckpt /chemin/fold_00/SupHeads_best_fold0.pth \
  --data /chemin/images \
  --feature_mode style \
  --embed_type tok6 \
  --inference_save_csv
```

### 7.3 `backbone_camera`
Webcam / caméra avec affichage des prédictions.

Exemple :

```bash
python test.py \
  --mode backbone_camera \
  --cfg /chemin/train_cfg.json \
  --weights_dir /chemin/fold_00 \
  --sup_ckpt /chemin/fold_00/SupHeads_best_fold0.pth \
  --feature_mode fusion \
  --sup_feat_source fusion \
  --embed_type tok6 \
  --classes_json data/Tasks.json
```

### 7.4 `tsne_interactive`
Visualisation t-SNE des représentations.

### 7.5 `passe_by_metrics`
Évaluation exploratoire avec métriques de clusters, KNN, séparabilité, etc.

### 7.6 `style_transfer`
Visualisation de transfert de style.

### 7.7 `detect_transformer`
Évaluation ou inférence en détection d'objets.

---

## 8. Checkpoints importants à recharger

### 8.1 En `sup_freeze`
À recharger au test selon le cas :
- toujours : `SupHeads_*.pth`
- si `generator` : `G_A_*.pt` / `G_B_*.pt` selon le backbone utilisé
- si `sem_resnet50` : `SemBackbone_*.pt`
- si `fusion` :
  - `SupHeads_*.pth`
  - `FusionHead_*.pth`
  - `SemBackbone_*.pt`
  - `G_A_*.pt` / `G_B_*.pt`

### 8.2 En `sup_unfreeze`
Même logique, mais il faut impérativement recharger les extracteurs **finetunés**, pas les checkpoints de préentraînement plus anciens.

---

## 9. Ablations recommandées

Le protocole conseillé est :
- définir une baseline complète,
- puis enlever un seul composant à la fois.

Ablations prioritaires :
- sans FFT
- sans SWD
- sans bloc texture complet
- sans JEPA
- JEPA style seul
- JEPA contenu seul
- sans adversarial
- sans PatchNCE
- sans Content-NCE
- sans branche sémantique
- sans reconstruction B

La comparaison se fait ensuite sur :
- F1
- accuracy
- top-1
- precision / recall
- métriques par tâche si multitâche

---

## 10. Dossier `scripts/`

Le dépôt contient désormais des scripts d'exemple dans :
- `scripts/train/`
- `scripts/test/`
- `scripts/ablations/`

Chaque script est un point de départ à adapter aux chemins locaux.

---

## 11. Scripts disponibles

### Entraînement
- `scripts/train/train_auto_example.sh`
- `scripts/train/train_sup_freeze_generator.sh`
- `scripts/train/train_sup_freeze_sem.sh`
- `scripts/train/train_sup_freeze_fusion.sh`
- `scripts/train/train_sup_unfreeze_fusion.sh`

### Test
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

## 12. Conseils pratiques

- En `sup_unfreeze`, vérifie que tu testes bien les extracteurs finetunés du dossier de fold.
- En `fusion`, assure-toi que `FusionHead_*.pth` est bien retrouvé et chargé.
- Pour ImageNet `val`, si les annotations XML existent, donne `--imagenet_ann_dir` ou laisse l'auto-détection fonctionner.
- Pour les splits sans annotations, le test bascule en mode soumission CSV.
- Pour la webcam, si OpenCV est headless, le code peut basculer vers Tkinter selon l'environnement.

---

## 13. Résumé rapide

- `auto` : préentraînement auto-supervisé
- `hybrid` : auto + supervision
- `sup_freeze` : supervision avec extracteurs gelés
- `sup_unfreeze` : supervision end-to-end
- `generator` : features de style
- `sem_resnet50` : features sémantiques
- `fusion` : style + contenu
- `sup_predict` : évaluer
- `inference` : prédire sans métriques
- `backbone_camera` : webcam

Ce README donne une vue d'ensemble. Pour un usage reproductible, partir des scripts du dossier `scripts/` est la méthode la plus simple.
