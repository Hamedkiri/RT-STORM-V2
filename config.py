# file: config.py
# -*- coding: utf-8 -*-

def get_opts():
    """
    Hyper-paramètres pour le schéma Double-GAN (G_A, D_A, G_B, D_B)
    + phase C supervisée (SupHeads intégrées au UNet)
    + modes classification (cls_tokens) et détection (fasterrcnn/detr/vitdet/fastrnn).

    Cette version intègre le nécessaire pour l'affichage:
      - terminal: --print_freq, --postfix_keys
      - TensorBoard: --tb, --tb_freq, --tb_freq_C, --tb_flush
      - checkpoints: --save_freq (string), --epoch_ckpt_interval (compat)
      - robustesse: --print_trace_on_error

    Ajout: options complètes et cohérentes pour la détection FastRNN
    entraînée sur les features de la branche sémantique ResNet50 (--sem_content),
    avec gel total/partiel et LR séparés backbone/head.
    """
    import argparse

    p = argparse.ArgumentParser("Style Perturb-Reconstr Training")

    # =========================================================================
    # 0) Général / runtime / affichage
    # =========================================================================
    p.add_argument("--seed", type=int, default=42, help="Graine RNG.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: 'cuda', 'cpu'. (Sinon auto).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Dataloader workers (si supporté par data.build_dataloader).")
    p.add_argument("--pin_memory", action="store_true",
                   help="Active pin_memory pour DataLoader (si supporté).")

    # --- Terminal logging ---
    p.add_argument("--print_freq", type=int, default=50,
                   help="Fréquence (en steps) d'affichage dans le terminal (hors tqdm).")
    p.add_argument("--postfix_keys", type=str,
                   default="loss_G,loss_D,loss_nce,loss_jepa,loss_idt,loss_reg",
                   help="Clés (séparées par ',') affichées dans le postfix tqdm.")

    # =========================================================================
    # 1) Données, IO & run
    # =========================================================================
    p.add_argument("--data", type=str, default=None,
                   help="Dossier d'images (ImageFolder) pour auto/hybrid/cls_tokens.")
    p.add_argument("--data_json", type=str, default=None,
                   help="Liste d'images au format JSON (optionnel).")
    p.add_argument("--classes_json", type=str, default=None,
                   help="JSON décrivant les classes (optionnel).")

    p.add_argument("--search_folder", type=str, default=None,
                   help="Répertoire racine où chercher les images (si data_json).")
    p.add_argument("--find_images_by_sub_folder", type=str, default=None,
                   help="Racine des sous-dossiers où se trouvent les images.")

    p.add_argument("--save_dir", type=str, required=True,
                   help="Répertoire où sauvegarder checkpoints + logs.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Taille de batch globale.")
    p.add_argument("--k_folds", type=int, default=2,
                   help="Nombre de folds pour alternance A/B (>=2).")
    p.add_argument("--epochs", type=int, default=25,
                   help="Nombre d'époques (mode auto/hybrid/sup_freeze/cls_tokens).")

    # (souvent utilisé dans les pipelines: resize/crop)
    p.add_argument("--crop_size", type=int, default=256,
                   help="Taille d'entrée (si utilisée dans les transforms).")

    # =========================================================================
    # 2) Mode d'entraînement
    # =========================================================================
    p.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "hybrid", "sup_freeze", "cls_tokens", "detect_transformer"],
        help=(
            "Mode d'entraînement :\n"
            "  - auto : A+B (self-supervised style + JEPA)\n"
            "  - hybrid : A+B puis C supervisé (SupHeads)\n"
            "  - sup_freeze : supervision seule, G/D gelés\n"
            "  - cls_tokens : classification à partir de tokens multi-échelles\n"
            "  - detect_transformer : entraînement détection (fasterrcnn/detr/vitdet/fastrnn)"
        ),
    )
    p.add_argument("--sup_from", choices=["GA", "GB"], default="GB",
                   help="Quel générateur fournit les features à SupHeads (C).")
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Nombre d'époques avant d'activer la supervision (hybrid).")
    p.add_argument("--sup_ratio", type=float, default=0.25,
                   help="Ratio d'itérations réservées à C en mode hybrid.")

    # =========================================================================
    # 3) Optimisation (générateurs / discriminateurs)
    # =========================================================================
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Taux d'apprentissage de base pour G_A/G_B.")
    p.add_argument("--adv_lrD_mult", type=float, default=0.5,
                   help="Facteur TTUR pour lr_D : lr_D = lr * adv_lrD_mult.")
    p.add_argument("--ema_tau", type=float, default=0.005,
                   help="EMA sur D_B ← D_A par epoch (0 = off).")

    # Replay buffer
    p.add_argument("--replay_size", type=int, default=50,
                   help="Taille du buffer de rejeu pour les paires (x,y).")
    p.add_argument("--replay_ratio", type=float, default=0.3,
                   help="Proportion de samples issus du buffer de rejeu.")

    # =========================================================================
    # 4) Scheduling des phases A / mix / B (CycleScheduler)
    # =========================================================================
    p.add_argument("--adv_only_epochs", type=int, default=2,
                   help="Nombre d'époques en phase A-adv (stylisation pure).")
    p.add_argument("--adv_mix_epochs", type=int, default=0,
                   help="Nombre d'époques en phase A-mix (si activée).")
    p.add_argument("--recon_epochs", type=int, default=2,
                   help="Nombre d'époques en phase B (reconstruction).")

    p.add_argument("--adv_boost", type=int, default=0,
                   help="Allonge la durée des phases A (adv/mix) par cycle.")
    p.add_argument("--b_boost", type=int, default=0,
                   help="Allonge la durée de la phase B par cycle.")
    p.add_argument("--skip_amix", action="store_true",
                   help="Saute complètement la phase A-mix.")
    p.add_argument("--feat_switch_epoch", type=int, default=50,
                   help="Époque de switch du backbone T_B pour NCE(A).")
    p.add_argument("--fold_epochs", type=int, default=4,
                   help="Nombre d'époques par fold avant switch A/B.")

    # =========================================================================
    # 5) Pertes de base (NCE / L1 / style / supervision)
    # =========================================================================
    p.add_argument("--lambda_nce_a_adv", type=float, default=1.0,
                   help="Poids PatchNCE en phase A-adv.")
    p.add_argument("--lambda_reg_a_adv", type=float, default=1.0,
                   help="Poids L1 (ou régularisation) en phase A-adv.")

    p.add_argument("--lambda_nce_a_mix", type=float, default=1.0,
                   help="Poids PatchNCE en phase A-mix.")
    p.add_argument("--lambda_reg_a_mix", type=float, default=0.5,
                   help="Poids L1 (ou régularisation) en phase A-mix.")
    p.add_argument("--amix_ramp", action="store_true",
                   help="Rampe linéaire des λ en phase A-mix.")

    p.add_argument("--lambda_nce_b", type=float, default=1.0,
                   help="Poids PatchNCE en phase B.")
    p.add_argument("--lambda_idt_b", type=float, default=1.0,
                   help="Poids L1 identité (recon x -> x) en phase B.")
    p.add_argument("--lambda_reg_b", type=float, default=10.0,
                   help="Poids L1 global en phase B (fallback si lambda_idt_b non utilisé).")

    # --- Style lambda scheduler ---
    p.add_argument("--style_lambda", type=float, default=10.0,
                   help="Valeur cible (max) du lambda de style pendant l'entraînement.")
    p.add_argument("--style_lambda_min", type=float, default=0.0,
                   help="Valeur de départ du lambda de style (avant warmup).")
    p.add_argument("--style_lambda_sched", type=str, default="none",
                   choices=["none", "linear", "cosine", "exp", "piecewise"],
                   help="Planning de λ_style : none/linear/cosine/exp/piecewise.")
    p.add_argument("--style_lambda_warmup", type=int, default=20,
                   help="Warmup epochs pour passer de style_lambda_min à style_lambda.")

    p.add_argument("--lambda_sup", type=float, default=1.0,
                   help="Poids de la perte supervisée (phase C, SupHeads).")

    # =========================================================================
    # 6) PatchNCE : configuration
    # =========================================================================
    p.add_argument("--nce_t", type=float, default=0.07,
                   help="Température de la loss PatchNCE.")
    p.add_argument("--nce_layers", type=str, default="bot,skip64,skip32",
                   help="Couches utilisées pour NCE (ex: 'bot,skip64,skip32').")
    p.add_argument("--nce_layer_weights", type=str, default=None,
                   help="Poids relatifs par couche NCE (ex: '2,1,1').")

    # NOTE: store_true + default=True => impossible à désactiver proprement
    # On conserve ton comportement mais on post-process via --no_nce_intra/inter.
    p.add_argument("--nce_intra", action="store_true", default=True,
                   help="Utiliser des négatifs intra-image (par défaut ON).")
    p.add_argument("--no_nce_intra", action="store_true",
                   help="Désactive les négatifs intra-image.")
    p.add_argument("--nce_inter", action="store_true", default=True,
                   help="Utiliser des négatifs inter-image (par défaut ON).")
    p.add_argument("--no_nce_inter", action="store_true",
                   help="Désactive les négatifs inter-image.")

    p.add_argument("--nce_max_patches", type=int, default=None,
                   help="Nombre max de patches NCE par image (None = tous).")
    p.add_argument("--nce_gate", type=float, default=2000.0,
                   help="Seuil NCE pour activer A-mix (0 = toujours actif).")

    # =========================================================================
    # 7) Texture (FFT / SWD)
    # =========================================================================
    p.add_argument("--tex_enable", action="store_true",
                   help="Active tout le bloc texture (FFT/SWD).")
    p.add_argument("--tex_apply_A", action="store_true",
                   help="Appliquer les pertes texture aussi en phase A (sinon seulement B).")

    p.add_argument("--tex_sigma", type=float, default=2.0,
                   help="Intensité du bruit spectral injecté (prétexte texture) sur far_mix en B.")
    p.add_argument("--tex_gamma", type=float, default=1.0,
                   help="Exposant de pondération fréquentielle du bruit (≈ 1/f^gamma).")

    p.add_argument("--tex_use_fft", action="store_true",
                   help="Calcule la perte FFT (distance L1 des amplitudes spectrales).")
    p.add_argument("--tex_use_swd", action="store_true",
                   help="Calcule la SWD multi-échelles.")

    p.add_argument("--lambda_fft", type=float, default=0.1, help="Poids de la perte FFT.")
    p.add_argument("--lambda_swd", type=float, default=0.05, help="Poids de la perte SWD.")
    p.add_argument("--swd_levels", type=str, default="64",
                   help="Niveaux de SWD (ex: '64,32,16' ou '3').")
    p.add_argument("--swd_patch", type=int, default=64, help="Taille de patch (carré) pour SWD.")
    p.add_argument("--swd_proj", type=int, default=128, help="Nb de projections 1D par niveau de SWD.")
    p.add_argument("--swd_max_patches", type=int, default=64,
                   help="Nb max de patches aléatoires par image et par niveau.")

    # =========================================================================
    # 8) Style tokens A / B (gains + balance dynamique)
    # =========================================================================
    p.add_argument("--style_gain_A", type=float, default=10.0,
                   help="Gain multiplicatif sur tokens (multi-échelles+global) en phase A.")
    p.add_argument("--style_gain_B", type=float, default=1.5,
                   help="Gain appliqué aux tokens dans le chemin style G_B (phase B).")

    p.add_argument("--lambda_style_b", type=float, default=0.005,
                   help="Poids initial de la perte de cohérence de style en B.")
    p.add_argument("--lambda_style_b_min", type=float, default=0.0001,
                   help="Plancher pour le λ_style_B_dyn.")
    p.add_argument("--lambda_style_b_max", type=float, default=2.0,
                   help="Plafond pour le λ_style_B_dyn.")
    p.add_argument("--style_b_warmup_epochs", type=int, default=1,
                   help="Warmup (époques) avant d'activer la perte de style en B.")
    p.add_argument("--style_balance_target", type=float, default=0.06,
                   help="Cible de distance de style recon ↔ x_mix.")
    p.add_argument("--style_balance_alpha", type=float, default=0.10,
                   help="Agressivité du contrôleur multiplicatif pour λ_style_B_dyn.")

    p.add_argument("--lambda_spade_gate", type=float, default=0.05,
                   help="Poids de la régularisation des portes SPADE (évite ws ≫ wg).")
    p.add_argument("--spade_gate_margin", type=float, default=0.75,
                   help="Marge de la régul SPADE: pénalise ReLU(margin*ws - wg).")

    p.add_argument("--token_ablate_eval_every", type=int, default=400,
                   help="Steps entre deux évaluations par ablation (gain=0).")

    # =========================================================================
    # 9) JEPA (auto-supervisé : style / contenu)
    # =========================================================================
    p.add_argument("--jepa_tokens", action="store_true",
                   help="Active la perte JEPA sur tokens (mode auto uniquement).")
    p.add_argument("--lambda_jepa", type=float, default=0.15,
                   help="Poids global de la perte JEPA (si pas séparée style/contenu).")
    p.add_argument("--jepa_every", type=int, default=2,
                   help="Calcul de la perte JEPA toutes les N itérations.")
    p.add_argument("--jepa_mask_ratio", type=float, default=0.60,
                   help="Ratio de positions masquées pour JEPA.")
    p.add_argument("--jepa_mask_bias_high", type=float, default=2.0,
                   help="Biais de masque vers les échelles hautes (tokG/t5).")
    p.add_argument("--jepa_scale_weights", type=str, default="2,2,1.5,1,0.75,0.5",
                   help="Poids par échelle JEPA (tokG,t5,t4,t3,t2,t1).")

    p.add_argument("--jepa_on_style", action="store_true", help="JEPA sur tokens de style.")
    p.add_argument("--jepa_on_content", action="store_true", help="JEPA sur features de contenu multi-échelles.")
    p.add_argument("--lambda_jepa_style", type=float, default=0.15, help="Poids JEPA sur style.")
    p.add_argument("--lambda_jepa_content", type=float, default=0.15, help="Poids JEPA sur contenu.")

    p.add_argument("--jepa_hidden_mult", type=int, default=2, help="Multiplicateur de largeur du MLP JEPA.")
    p.add_argument("--jepa_heads", type=int, default=4, help="Nombre de têtes d'attention JEPA.")
    p.add_argument("--jepa_norm", type=int, default=1, help="1: normalisation interne JEPA, 0: pas de norm.")

    p.add_argument("--lambda_jepa_var", type=float, default=0.05, help="Poids variance (anti-collapse).")
    p.add_argument("--lambda_jepa_cov", type=float, default=0.05, help="Poids covariance (anti-redondance).")
    p.add_argument("--lambda_jepa_kd", type=float, default=0.05, help="Poids distillation depuis SupHeads.")
    p.add_argument("--jepa_use_teacher", type=int, default=1,
                   help="1: utiliser teachers EMA comme cibles, 0: student lui-même.")

    p.add_argument("--ema_update_every", type=int, default=1,
                   help="Fréquence (en itérations) d'update de T_A/T_B.")
    p.add_argument("--nce_m", type=float, default=0.999,
                   help="Momentum EMA pour T_A/T_B (NCE/JEPA).")

    # =========================================================================
    # 10) Mix-Swap (tokens / FFT)
    # =========================================================================
    p.add_argument("--mixswap_enable", action="store_true", help="Active le Mix-Swap.")
    p.add_argument("--mixswap_alpha", type=str, default="0.3,0.7",
                   help="Intervalle [lo,hi] pour le mix des tokens (ex: '0.3,0.7').")
    p.add_argument("--mixswap_token_p", type=float, default=1.0,
                   help="Proba d'appliquer le mix de tokens par batch.")
    p.add_argument("--mixswap_fft_p", type=float, default=0.0,
                   help="Proba d'un mix d'amplitude FFT sur y (visuel).")

    # =========================================================================
    # 11) Adversarial GAN (phases A/B)
    # =========================================================================
    p.add_argument("--adv_enable_A", action="store_true", help="Active la loss adversariale en phase A.")
    p.add_argument("--adv_enable_B", action="store_true", help="Active la loss adversariale en phase B.")
    p.add_argument("--adv_type", type=str, default="lsgan",
                   choices=["hinge", "lsgan"], help="Type de loss adversariale.")
    p.add_argument("--adv_r1_gamma", type=float, default=10.0, help="Poids pénalité R1 (0 = off).")
    p.add_argument("--adv_r1_every", type=int, default=16, help="Appliquer R1 toutes les N itérations.")
    p.add_argument("--adv_highpass", action="store_true", help="Focaliser D sur les hautes fréquences.")

    # =========================================================================
    # 12) Supervision multi-tâches (SupHeads)
    # =========================================================================
    p.add_argument(
        "--sup_feat_type",
        choices=[
            "tokG", "tok6", "tok6_mean", "tok6_w",
            "cont_tok", "cont_tok_vit",
            "style_tok", "bot", "bot+tok", "tok+delta", "mgap", "mgap+tok",
        ],
        default="tok6",
        help=(
            "Type de features fournis aux têtes supervisées (SupHeads).\n"
            "  - tok*: tokens de style/multi-échelles\n"
            "  - cont_tok*: tokens de contenu via encode_content(x)\n"
        ),
    )
    p.add_argument("--delta_weights", type=str, default="1,1,1,1,1",
                   help="Poids par échelle pour 'tok+delta' (Δ[s5],Δ[s4],Δ[s3],Δ[s2],Δ[s1]).")
    p.add_argument("--sup_tasks_json", type=str, default=None,
                   help="JSON décrivant les tâches : {task: n_classes ou [class_names...]}")

    # =========================================================================
    # 13) Backbone : option globale + overrides par mode
    # =========================================================================
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Gèle backbone pour cls_tokens/detect_transformer (peut être overridé).")
    p.add_argument("--cls_freeze_backbone", type=int, default=None,
                   help="Override freeze pour classification (None => global).")
    # NB: det_freeze_backbone (override) est défini dans le bloc Détection ci-dessous.

    # =========================================================================
    # 14) Détection (dataset COCO-like + choix head + FastRNN + freeze sémantique)
    # =========================================================================
    # Dataset/IO
    p.add_argument("--det_train_img_root", type=str, default=None,
                   help="Dossier images train pour détection (COCO-like).")
    p.add_argument("--det_train_ann", type=str, default=None,
                   help="Annotations train au format COCO (JSON).")
    p.add_argument("--det_val_img_root", type=str, default=None,
                   help="Dossier images val pour détection.")
    p.add_argument("--det_val_ann", type=str, default=None,
                   help="Annotations val au format COCO (JSON).")
    p.add_argument("--det_img_h", type=int, default=256,
                   help="Hauteur images d'entrée détection (resize).")
    p.add_argument("--det_img_w", type=int, default=256,
                   help="Largeur images d'entrée détection (resize).")

    # Classes / mapping
    p.add_argument("--det_num_classes", type=int, default=91,
                   help=("Nombre total de classes détection INCLUANT background "
                         "(torchvision style)."))
    p.add_argument("--det_classes_file", type=str, default="",
                   help=("JSON listant classes autorisées (ids ou noms). "
                         "Si fourni: mapping old_id->new_id (1..K) ; bg=0."))
    p.add_argument("--det_debug_merge", action="store_true",
                   help="Debug filtrage/merge des annotations détection.")

    # Head choisi
    p.add_argument("--det_head", type=str,
                   choices=["fasterrcnn", "detr", "vitdet", "fastrnn"],
                   default="fasterrcnn",
                   help="Type de tête de détection: fasterrcnn/detr/vitdet/fastrnn.")

    # Action de détection (appelée via --mode detect_transformer)
    p.add_argument("--det_run", type=str, default="train",
                   choices=["train", "eval", "camera"],
                   help=("Action en mode détection :\n"
                         "  - train  : entraîne le détecteur\n"
                         "  - eval   : évalue un checkpoint sur COCO (mAP)\n"
                         "  - camera : démo caméra (webcam/RTSP via OpenCV)"))

    # Checkpoints détection
    p.add_argument("--det_ckpt", type=str, default="",
                   help=("Checkpoint détecteur (.pth) à charger pour eval/camera, "
                         "ou pour fine-tuning. Si vide: tente detector_best.pth puis detector_last.pth dans --save_dir."))
    p.add_argument("--det_resume", type=str, default="",
                   help=("Checkpoint détecteur (.pth) à reprendre pour continuer l'entraînement "
                         "(reprend epoch/optimizer)."))

    # Source de features
    p.add_argument("--det_feat_source", type=str, default="sem_resnet50",
                   choices=["unet_content", "unet_style", "unet_concat", "sem_resnet50"],
                   help=("Source de features pour détection.\n"
                         "  - sem_resnet50 : utilise la branche --sem_content (ResNet50)\n"
                         "  - unet_* : utilise les features internes UNet/ST-STORM"))

    # Si det_feat_source=sem_resnet50, on choisit la couche exportée
    p.add_argument("--det_sem_return_layer", type=str, default="layer4",
                   choices=["layer2", "layer3", "layer4"],
                   help=("Couche ResNet sémantique fournie au head.\n"
                         "layer2(stride~8), layer3(~16), layer4(~32)"))

    # Gel backbone pour détection (override du global)
    p.add_argument("--det_freeze_backbone", type=int, default=None,
                   help=("Override gel backbone en détection: "
                         "None=>--freeze_backbone ; 1=>gel ; 0=>trainable."))

    # Gel fin du ResNet sémantique (si utilisé en détection)
    p.add_argument("--det_sem_freeze_at", type=int, default=-1,
                   choices=[-1, 0, 1, 2, 3, 4],
                   help=("Gel progressif du ResNet sémantique (si det_feat_source=sem_resnet50):\n"
                         " -1: rien gelé\n"
                         "  0: conv1+bn1\n"
                         "  1: +layer1\n"
                         "  2: +layer2\n"
                         "  3: +layer3\n"
                         "  4: +layer4 (tout)"))

    # Option: pendant les epochs détection, couper les pertes sem (MoCo/JEPA-content)
    p.add_argument("--det_disable_sem_losses", action="store_true",
                   help="En mode détection: ne calcule pas les losses sem (MoCo/JEPA-content).")

    # Optim détection
    p.add_argument("--det_epochs", type=int, default=20, help="Nb epochs détection.")
    p.add_argument("--det_lr_head", type=float, default=1e-4, help="LR tête de détection.")
    p.add_argument("--det_lr_backbone", type=float, default=1e-5,
                   help="LR backbone en détection (si non gelé).")
    p.add_argument("--det_weight_decay", type=float, default=1e-4, help="Weight decay détection.")
    p.add_argument("--det_grad_clip", type=float, default=0.0, help="Grad clip norm (0=off).")

    # DETR-like (si det_head='detr')
    p.add_argument("--det_num_queries", type=int, default=300, help="Nb queries DETR.")
    p.add_argument("--det_nheads", type=int, default=8, help="Nb têtes attention décodeur.")
    p.add_argument("--det_dec_layers", type=int, default=6, help="Nb couches décodeur DETR.")
    p.add_argument("--det_eos_coef", type=float, default=0.1, help="Poids 'no object' DETR.")

    # FasterRCNN (si det_head='fasterrcnn') - anchors
    p.add_argument("--fasterrcnn_anchor_sizes", type=str, default="32,64,128,256,512",
                   help="Anchor sizes (comma-separated).")
    p.add_argument("--fasterrcnn_aspect_ratios", type=str, default="0.5,1.0,2.0",
                   help="Anchor aspect ratios (comma-separated).")

    # FastRNN (si det_head='fastrnn')
    p.add_argument("--fastrnn_hidden", type=int, default=256, help="Dim interne FastRNN head.")
    p.add_argument("--fastrnn_dropout", type=float, default=0.0, help="Dropout FastRNN head.")

    # Bool propre: bidir par défaut ON, désactivable avec --fastrnn_no_bidir
    p.add_argument("--fastrnn_bidir", action="store_true",
                   help="Active FastRNN bidirectionnel.")
    p.add_argument("--fastrnn_no_bidir", action="store_true",
                   help="Force FastRNN unidirectionnel.")

    # Loss anchor-free (focal + reg + centerness)
    p.add_argument("--fastrnn_focal_alpha", type=float, default=0.25)
    p.add_argument("--fastrnn_focal_gamma", type=float, default=2.0)
    p.add_argument("--fastrnn_lambda_reg", type=float, default=1.0)
    p.add_argument("--fastrnn_lambda_ctr", type=float, default=1.0)
    p.add_argument("--fastrnn_lambda_cls", type=float, default=1.0)

    # Inference / post-process
    p.add_argument("--fastrnn_score_thresh", type=float, default=0.05)
    p.add_argument("--fastrnn_nms_thresh", type=float, default=0.5)
    p.add_argument("--fastrnn_topk", type=int, default=1000)
    p.add_argument("--fastrnn_size_divisible", type=int, default=32,
                   help="Padding images pour batch (multiple de).")
    p.add_argument("--fastrnn_force_stride", type=int, default=0,
                   help="0=auto, sinon impose un stride constant (ex 32).")

    # Évaluation COCO
    p.add_argument("--det_eval_iou_types", type=str, default="bbox",
                   help="Types IoU COCOeval (ex: 'bbox' ou 'bbox,segm').")
    p.add_argument("--det_eval_max_dets", type=int, default=100,
                   help="COCOeval maxDets (par image).")
    p.add_argument("--det_eval_limit", type=int, default=0,
                   help="Limite nb d'images en évaluation (0=val complet).")

    # Caméra (OpenCV)
    p.add_argument("--det_cam_id", type=int, default=0,
                   help="ID caméra OpenCV (0,1,...) ou index si webcam.")
    p.add_argument("--det_cam_url", type=str, default="",
                   help="URL caméra (RTSP/HTTP). Si défini, prioritaire sur --det_cam_id.")
    p.add_argument("--det_cam_threshold", type=float, default=0.35,
                   help="Seuil score pour afficher les boxes à la caméra.")
    p.add_argument("--det_cam_save", type=str, default="",
                   help="Chemin de sortie vidéo (.mp4) pour enregistrer la démo caméra (optionnel).")

    # =========================================================================
    # 15) Classification via tokens (cls_tokens)
    # =========================================================================
    p.add_argument("--cls_num_classes", type=int, default=10,
                   help="Fallback nb classes si non déduit du dataset.")
    p.add_argument("--cls_d_model", type=int, default=256, help="Dim tokens tête cls.")
    p.add_argument("--cls_nhead", type=int, default=4, help="Nb têtes TransformerEncoder cls.")
    p.add_argument("--cls_layers", type=int, default=2, help="Nb couches TransformerEncoder cls.")
    p.add_argument("--cls_dim_ff", type=int, default=1024, help="Dim feedforward cls.")
    p.add_argument("--cls_dropout", type=float, default=0.1, help="Dropout tête cls.")

    p.add_argument("--cls_lr_backbone", type=float, default=1e-4,
                   help="LR backbone en cls_tokens (si non gelé).")
    p.add_argument("--cls_lr_head", type=float, default=1e-3,
                   help="LR token_encoder + tête cls.")
    p.add_argument("--cls_epochs", type=int, default=50, help="Nb epochs classification.")
    p.add_argument("--cls_save_freq", type=int, default=10,
                   help="Save freq checkpoints cls (epochs).")

    # =========================================================================
    # 16) Logs & checkpoints
    # =========================================================================
    p.add_argument("--tb", action="store_true", help="Active TensorBoard.")
    p.add_argument("--tb_freq", type=int, default=100,
                   help="Itérations entre deux logs TensorBoard (A/B).")
    p.add_argument("--tb_freq_C", type=int, default=50,
                   help="Itérations entre logs TensorBoard pour C (sup).")
    p.add_argument("--tb_flush", type=int, default=100,
                   help="Flush TB tous les N steps (0 => jamais).")

    p.add_argument("--save_freq", type=str, default="epoch",
                   help="Fréquence ckpt: 'epoch', 'epoch:N', 'step', 'step:N', 'none' ou 'N' (step:N).")
    p.add_argument("--epoch_ckpt_interval", type=int, default=None,
                   help="Compat: intervalle en epochs si tu veux forcer un N (override save_freq epoch).")

    p.add_argument("--resume_dir", type=str, default=None,
                   help="Répertoire de reprise (checkpoint précédent).")
    p.add_argument("--print_trace_on_error", action="store_true",
                   help="Affiche le traceback complet en cas d'exception.")

    # =========================================================================
    # 17) PatchNCE contenu (invariance au style)
    # =========================================================================
    p.add_argument("--content_nce_enable", action="store_true",
                   help="Active un PatchNCE supplémentaire sur le contenu (invariance au style).")
    p.add_argument("--lambda_content_nce", type=float, default=0.0,
                   help="Poids de la NCE contenu (entre deux styles pour le même contenu).")

    # =========================================================================
    # 18) Contenu sémantique (ResNet50 + MoCo + JEPA-content)
    # =========================================================================
    p.add_argument("--sem_content", action="store_true",
                   help="Active une branche de contenu sémantique (ResNet50 + MoCo + JEPA-content).")
    p.add_argument("--lambda_sem", type=float, default=0.5,
                   help="Poids de la perte contrastive MoCo pour le contenu sémantique.")
    p.add_argument("--sem_every", type=int, default=1,
                   help="Optimise la branche sémantique toutes les N itérations.")
    p.add_argument("--sem_sym", action="store_true", help="Loss symétrique (x->far et far->x).")
    p.add_argument("--sem_two_styles", action="store_true",
                   help="Deux styles pour une même image comme positifs supplémentaires.")
    p.add_argument("--sem_detach_far", type=int, default=1,
                   help="1: contraste ne rétro-propage pas vers G_A (far.detach()).")
    p.add_argument("--lr_sem", type=float, default=None,
                   help="LR branche sémantique (None => --lr).")
    p.add_argument("--sem_pretrained", type=int, default=1,
                   help="1: ResNet50 ImageNet pré-entraîné pour content_sem.")
    p.add_argument("--sem_dim", type=int, default=256, help="Dim embedding global MoCo.")
    p.add_argument("--sem_tok_dim", type=int, default=256, help="Dim tokens sémantiques (JEPA-content).")
    p.add_argument("--sem_queue", type=int, default=65536, help="Taille queue MoCo.")
    p.add_argument("--sem_m", type=float, default=0.999, help="Momentum EMA encodeur MoCo.")
    p.add_argument("--sem_t", type=float, default=0.2, help="Température MoCo (InfoNCE).")

    p.add_argument("--sem_pretrained_path", type=str, default=None,
                   help="Chemin vers un checkpoint externe pour initialiser le backbone sémantique.")
    p.add_argument("--sem_pretrained_strict", type=int, default=0,
                   help="1: strict load_state_dict ; 0: permissif.")
    p.add_argument("--sem_pretrained_verbose", type=int, default=1,
                   help="1: logs chargement checkpoint ; 0: silencieux.")

    p.add_argument("--sem_use_aug", action="store_true",
                   help="Augmentations SimCLR-like pour le contraste (en plus de la vue far).")
    p.add_argument("--sem_crop", type=int, default=224, help="Taille crop branche sémantique.")
    p.add_argument("--sem_min_scale", type=float, default=0.5, help="Scale min RandomResizedCrop.")
    p.add_argument("--sem_color_jitter", type=float, default=0.4, help="Amplitude ColorJitter.")
    p.add_argument("--sem_gray", type=float, default=0.2, help="Probabilité RandomGrayscale.")
    p.add_argument("--sem_blur", type=float, default=0.1, help="Probabilité GaussianBlur.")

    # =========================================================================
    # 19) Petites options de sécurité / qualité de vie
    # =========================================================================
    p.add_argument("--strict_load", type=int, default=0,
                   help="Option générique si tu veux relayer strict=True dans certains load_checkpoint.")
    p.add_argument("--debug", action="store_true",
                   help="Mode debug: peut réduire nbatchs ailleurs si tu l'implémentes.")

    # =========================================================================
    # Parse
    # =========================================================================
    args = p.parse_args()

    # --- Post-process flags incohérents (nce_intra/inter)
    if getattr(args, "no_nce_intra", False):
        args.nce_intra = False
    if getattr(args, "no_nce_inter", False):
        args.nce_inter = False

    # --- FastRNN bidir : défaut ON, désactivable
    if getattr(args, "fastrnn_no_bidir", False):
        args.fastrnn_bidir = False
    else:
        # défaut = True (si tu ne passes aucun flag)
        args.fastrnn_bidir = True

    # --- Override freeze backbone détection : fallback sur freeze_backbone global
    if args.det_freeze_backbone is None:
        args.det_freeze_backbone = 1 if getattr(args, "freeze_backbone", False) else 0
    else:
        args.det_freeze_backbone = int(args.det_freeze_backbone)

    # --- Override freeze backbone classification : fallback sur freeze_backbone global
    if args.cls_freeze_backbone is None:
        args.cls_freeze_backbone = 1 if getattr(args, "freeze_backbone", False) else 0
    else:
        args.cls_freeze_backbone = int(args.cls_freeze_backbone)

    return args