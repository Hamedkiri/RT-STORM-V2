def get_opts():
    """
    Hyper-paramètres pour le schéma Double-GAN (G_A, D_A, G_B, D_B)
    + phase C supervisée (SupHeads intégrées au UNet)
    + modes classification (cls_tokens) et détection (DETR-like).
    """
    import argparse
    p = argparse.ArgumentParser("Style Perturb-Reconstr Training")

    # =========================================================================
    # 1) Données, IO & run
    # =========================================================================
    p.add_argument("--data", type=str,
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
    p.add_argument("--seed", type=int, default=42,
                   help="Graine RNG.")
    p.add_argument("--epochs", type=int, default=25,
                   help="Nombre d'époques (mode auto/hybrid/sup_freeze/cls_tokens).")

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
            "  - detect_transformer : tête DETR-like pour détection"
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
    # Phases A-adv / A-mix / B
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

    # ---------------------------------------------------------
    #  Style lambda scheduler (Phase A / B)
    # ---------------------------------------------------------
    p.add_argument(
        '--style_lambda',
        type=float,
        default=10.0,
        help="Valeur cible (max) du lambda de style pendant l'entraînement."
    )

    p.add_argument(
        '--style_lambda_min',
        type=float,
        default=0.0,
        help="Valeur de départ du lambda de style (avant warmup)."
    )

    p.add_argument(
        '--style_lambda_sched',
        type=str,
        default='none',
        choices=['none', 'linear', 'cosine', 'exp', 'piecewise'],
        help=(
            "Planning de λ_style au cours de l'entraînement : "
            "'none' (constant), 'linear', 'cosine', 'exp', ou 'piecewise'."
        )
    )

    p.add_argument(
        '--style_lambda_warmup',
        type=int,
        default=20,
        help=(
            "Nombre d'epochs de warmup pour passer de style_lambda_min "
            "à style_lambda (valeur cible/max)."
        )
    )

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

    p.add_argument("--nce_intra", action="store_true", default=True,
                   help="Utiliser des négatifs intra-image.")
    p.add_argument("--nce_inter", action="store_true", default=True,
                   help="Utiliser des négatifs inter-image.")
    p.add_argument("--nce_max_patches", type=int, default=None,
                   help="Nombre max de patches NCE par image (None = tous).")
    p.add_argument("--nce_gate", type=float, default=2000.0,
                   help="Seuil NCE pour activer A-mix (0 = toujours actif).")

    # =========================================================================
    # 7) Texture (FFT / SWD)
    # =========================================================================
    p.add_argument(
        "--tex_enable",
        action="store_true",
        help="Active tout le bloc texture (FFT/SWD)."
    )
    p.add_argument(
        "--tex_apply_A",
        action="store_true",
        help="Appliquer les pertes texture aussi en phase A (sinon seulement B)."
    )

    p.add_argument(
        "--tex_sigma", type=float, default=2.0,
        help="Intensité du bruit spectral injecté (prétexte texture) sur far_mix en B."
    )
    p.add_argument(
        "--tex_gamma", type=float, default=1.0,
        help="Exposant de pondération fréquentielle du bruit (≈ 1/f^gamma)."
    )

    p.add_argument(
        "--tex_use_fft", action="store_true",
        help="Calcule la perte FFT (distance L1 des amplitudes spectrales)."
    )
    p.add_argument(
        "--tex_use_swd", action="store_true",
        help="Calcule la SWD multi-échelles. Plus coûteux mais plus expressif."
    )

    p.add_argument("--lambda_fft", type=float, default=0.1,
                   help="Poids de la perte FFT.")
    p.add_argument("--lambda_swd", type=float, default=0.05,
                   help="Poids de la perte SWD.")

    p.add_argument("--swd_levels", type=str, default="64",
                   help="Niveaux de SWD (ex: '64,32,16' ou '3').")
    p.add_argument("--swd_patch", type=int, default=64,
                   help="Taille de patch (carré) pour SWD.")
    p.add_argument("--swd_proj", type=int, default=128,
                   help="Nb de projections 1D par niveau de SWD.")
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

    # Régularisation SPADE / gating
    p.add_argument("--lambda_spade_gate", type=float, default=0.05,
                   help="Poids de la régularisation des portes SPADE (évite ws ≫ wg).")
    p.add_argument("--spade_gate_margin", type=float, default=0.75,
                   help="Marge de la régul SPADE: pénalise ReLU(margin*ws - wg).")

    # Évaluations d'ablation de tokens
    p.add_argument("--token_ablate_eval_every", type=int, default=400,
                   help="Steps entre deux évaluations par ablation (gain=0).")

    # =========================================================================
    # 9) JEPA (auto-supervisé : style / contenu)
    # =========================================================================
    p.add_argument("--jepa_tokens", action="store_true",
                   help="Active la perte JEPA sur tokens de style (mode auto uniquement).")
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

    # Style vs contenu
    p.add_argument("--jepa_on_style", action="store_true",
                   help="JEPA sur tokens de style.")
    p.add_argument("--jepa_on_content", action="store_true",
                   help="JEPA sur features de contenu multi-échelles.")
    p.add_argument("--lambda_jepa_style", type=float, default=0.15,
                   help="Poids de la perte JEPA sur style.")
    p.add_argument("--lambda_jepa_content", type=float, default=0.15,
                   help="Poids de la perte JEPA sur contenu.")

    # Architecture JEPA
    p.add_argument("--jepa_hidden_mult", type=int, default=2,
                   help="Multiplicateur de largeur du MLP JEPA.")
    p.add_argument("--jepa_heads", type=int, default=4,
                   help="Nombre de têtes d'attention dans la tête JEPA.")
    p.add_argument("--jepa_norm", type=int, default=1,
                   help="1: normalisation interne JEPA, 0: pas de norm.")

    # Anti-collapse / distillation
    p.add_argument("--lambda_jepa_var", type=float, default=0.05,
                   help="Poids du terme variance (anti-collapse).")
    p.add_argument("--lambda_jepa_cov", type=float, default=0.05,
                   help="Poids du terme covariance (anti-redondance).")
    p.add_argument("--lambda_jepa_kd", type=float, default=0.05,
                   help="Poids de la distillation depuis SupHeads (0 = off).")
    p.add_argument("--jepa_use_teacher", type=int, default=1,
                   help="1: utiliser teachers EMA comme cibles, 0: student lui-même.")

    # EMA teachers pour JEPA
    p.add_argument("--ema_update_every", type=int, default=1,
                   help="Fréquence (en itérations) d'update de T_A/T_B.")
    p.add_argument("--nce_m", type=float, default=0.999,
                   help="Momentum EMA pour T_A/T_B (NCE/JEPA).")

    # =========================================================================
    # 10) Mix-Swap (tokens / FFT)
    # =========================================================================
    p.add_argument("--mixswap_enable", action="store_true",
                   help="Active le Mix-Swap.")
    p.add_argument("--mixswap_alpha", type=str, default="0.3,0.7",
                   help="Intervalle [lo,hi] pour le mix des tokens (ex: '0.3,0.7').")
    p.add_argument("--mixswap_token_p", type=float, default=1.0,
                   help="Proba d'appliquer le mix de tokens par batch.")
    p.add_argument("--mixswap_fft_p", type=float, default=0.0,
                   help="Proba d'un mix d'amplitude FFT sur y (visuel).")

    # =========================================================================
    # 11) Adversarial GAN (phases A/B)
    # =========================================================================
    p.add_argument("--adv_enable_A", action="store_true",
                   help="Active la loss adversariale en phase A.")
    p.add_argument("--adv_enable_B", action="store_true",
                   help="Active la loss adversariale en phase B.")
    p.add_argument("--adv_type", type=str, default="lsgan",
                   choices=["hinge", "lsgan"],
                   help="Type de loss adversariale.")
    p.add_argument("--adv_r1_gamma", type=float, default=10.0,
                   help="Poids de la pénalité R1 sur les réels (0 = off).")
    p.add_argument("--adv_r1_every", type=int, default=16,
                   help="Appliquer R1 toutes les N itérations.")
    p.add_argument("--adv_highpass", action="store_true",
                   help="Focaliser D sur les hautes fréquences.")

    # =========================================================================
    # 12) Supervision multi-tâches (SupHeads)
    # =========================================================================
    p.add_argument(
        "--sup_feat_type",
        choices=[
            # tokens multi-échelles "style"
            "tokG", "tok6", "tok6_mean", "tok6_w",
            # nouveaux : tokens de contenu + ViT-like
            "cont_tok",  # tokens de contenu + simple mean pooling
            "cont_tok_vit",  # tokens de contenu + petit Transformer (TokenClassifier sans head)
            # historiques (compat)
            "style_tok", "bot", "bot+tok", "tok+delta", "mgap", "mgap+tok"
        ],
        default="tok6",
        help=(
            "Type de features fournis aux têtes supervisées (SupHeads).\n"
            "  - tok*: tokens de style/multi-échelles\n"
            "  - cont_tok*: tokens de contenu via encode_content(x)\n"
        ),
    )

    p.add_argument(
        "--delta_weights", type=str, default="1,1,1,1,1",
        help=(
            "Poids par échelle pour 'tok+delta' (ordre Δ[s5],Δ[s4],Δ[s3],Δ[s2],Δ[s1]). "
            "Ex: '2,1.5,1,1,0.5'."
        )
    )
    p.add_argument(
        "--sup_tasks_json", type=str,
        help="JSON décrivant les tâches : {task: n_classes ou [class_names...]}"
    )

    # =========================================================================
    # 13) Backbone : option globale + overrides par mode
    # =========================================================================
    # Option commune : si flag présent → backbone gelé pour cls_tokens ET detect_transformer
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help=(
            "Si présent : gèle le backbone pour les modes cls_tokens et detect_transformer "
            "(linear probe / det-head only). "
            "Les options plus spécifiques --cls_freeze_backbone / --det_freeze_backbone "
            "peuvent surcharger ce réglage global (0/1)."
        ),
    )

    # Overrides spécifiques (pour compatibilité fine)
    p.add_argument("--cls_freeze_backbone", type=int, default=None,
                   help="Override de --freeze_backbone pour la classification (None => utiliser global).")
    p.add_argument("--det_freeze_backbone", type=int, default=None,
                   help="Override de --freeze_backbone pour la détection (None => utiliser global).")

    # =========================================================================
    # 14) Détection transformer (DETR-like)
    # =========================================================================
    # Données
    p.add_argument("--det_train_img_root", type=str, default=None,
                   help="Dossier images train pour détection (COCO-like).")
    p.add_argument("--det_train_ann", type=str, default=None,
                   help="Annotations train COCO-like (JSON).")
    p.add_argument("--det_val_img_root", type=str, default=None,
                   help="Dossier images val pour détection.")
    p.add_argument("--det_val_ann", type=str, default=None,
                   help="Annotations val COCO-like (JSON).")
    p.add_argument("--det_img_h", type=int, default=256,
                   help="Hauteur des images d'entrée pour la détection.")
    p.add_argument("--det_img_w", type=int, default=256,
                   help="Largeur des images d'entrée pour la détection.")

    # Backbone / branche features pour détection
    p.add_argument("--det_feat_branch", type=str, default="content",
                   choices=["content", "style", "concat"],
                   help="Type de features utilisés pour la détection.")
    p.add_argument("--det_backbone_ckpt", type=str, default=None,
                   help="Checkpoint pour initialiser le backbone (auto-supervisé, etc.).")

    # Hyperparamètres détection
    p.add_argument("--det_lr", type=float, default=1e-4,
                   help="LR de la tête de détection.")
    p.add_argument("--det_weight_decay", type=float, default=1e-4,
                   help="Weight decay pour la tête de détection.")
    p.add_argument("--det_num_classes", type=int, default=91,
                   help="Nombre de classes COCO-like (incluant background si besoin).")
    p.add_argument("--det_num_queries", type=int, default=300,
                   help="Nombre de queries DETR.")
    p.add_argument("--det_nheads", type=int, default=8,
                   help="Nombre de têtes d'attention dans le décodeur.")
    p.add_argument("--det_dec_layers", type=int, default=6,
                   help="Nombre de couches du décodeur DETR.")
    p.add_argument("--det_eos_coef", type=float, default=0.1,
                   help="Poids du token 'no object' dans la loss DETR.")
    p.add_argument("--det_epochs", type=int, default=20,
                   help="Nombre d'époques pour la détection (mode detect_transformer).")

    p.add_argument('--det_head_type', type=str, default='simple_unet',
                        help="Tête de détection: 'simple_unet' (UNet+SimpleDETRHead) ou 'detr_resnet50'")

    # =========================================================================
    # 15) Classification via tokens (cls_tokens)
    # =========================================================================
    p.add_argument("--cls_num_classes", type=int, default=10,
                   help="(Fallback) Nombre de classes si non déduit depuis le dataset.")
    p.add_argument("--cls_d_model", type=int, default=256,
                   help="Dimension des tokens pour la tête de classification.")
    p.add_argument("--cls_nhead", type=int, default=4,
                   help="Nombre de têtes de la TransformerEncoder pour cls.")
    p.add_argument("--cls_layers", type=int, default=2,
                   help="Nombre de couches TransformerEncoder pour cls.")
    p.add_argument("--cls_dim_ff", type=int, default=1024,
                   help="Dimension du feedforward interne de la tête cls.")
    p.add_argument("--cls_dropout", type=float, default=0.1,
                   help="Dropout dans la tête cls.")

    p.add_argument("--cls_lr_backbone", type=float, default=1e-4,
                   help="LR pour le backbone en mode cls_tokens (si non gelé).")
    p.add_argument("--cls_lr_head", type=float, default=1e-3,
                   help="LR pour le token_encoder + tête cls.")
    p.add_argument("--cls_epochs", type=int, default=50,
                   help="Nombre d'époques pour la classification.")
    p.add_argument("--cls_save_freq", type=int, default=10,
                   help="Fréquence de sauvegarde des checkpoints cls (en époques).")

    # =========================================================================
    # 16) Logs & checkpoints
    # =========================================================================
    p.add_argument("--tb", action="store_true",
                   help="Active TensorBoard.")
    p.add_argument("--tb_freq", type=int, default=100,
                   help="Itérations entre deux logs TensorBoard (A/B).")
    p.add_argument("--tb_freq_C", type=int, default=50,
                   help="Itérations entre logs TensorBoard pour C (sup).")

    p.add_argument("--save_freq", type=int, default=5,
                   help="Sauvegarde checkpoints tous les N epochs (auto/hybrid).")
    p.add_argument("--resume_dir", type=str, default=None,
                   help="Répertoire pour reprise (checkpoint précédent).")

    p.add_argument("--print_trace_on_error", action="store_true",
                   help="Affiche le traceback complet en cas d'exception.")

    p.add_argument(
        "--content_nce_enable",
        action="store_true",
        help="Active un PatchNCE supplémentaire sur le contenu (invariance au style).",
    )
    p.add_argument(
        "--lambda_content_nce",
        type=float,
        default=0.0,
        help="Poids de la NCE contenu (entre deux styles pour le même contenu).",
    )

    # =========================================================================
    # 17) Retour des options
    # =========================================================================

    # =========================================================================
    # 13) Contenu sémantique (ResNet50 + MoCo + JEPA-content)
    # =========================================================================
    p.add_argument("--sem_content", action="store_true",
                   help="Active une branche de contenu sémantique (ResNet50 + MoCo + JEPA-content) "
                        "pour obtenir des features de contenu invariantes au style via (x, far).")
    p.add_argument("--lambda_sem", type=float, default=0.5,
                   help="Poids de la perte contrastive (MoCo) pour le contenu sémantique.")
    p.add_argument("--sem_every", type=int, default=1,
                   help="Calcule/optimise la branche sémantique toutes les N itérations.")
    p.add_argument("--sem_sym", action="store_true",
                   help="Ajoute une loss symétrique (x->far et far->x).")
    p.add_argument("--sem_two_styles", action="store_true",
                   help="Utilise deux styles pour une même image (far1 & far2) comme positifs supplémentaires "
                        "(far2 = shuffle(y) dans le batch).")
    p.add_argument("--sem_detach_far", type=int, default=1,
                   help="Si 1, le contraste ne rétro-propage pas vers G_A (far.detach()). Recommandé.")
    p.add_argument("--lr_sem", type=float, default=None,
                   help="Learning rate de la branche sémantique. Si None, réutilise --lr.")
    p.add_argument("--sem_pretrained", type=int, default=1,
                   help="Si 1, ResNet50 ImageNet pré-entraîné pour content_sem.")
    p.add_argument("--sem_dim", type=int, default=256,
                   help="Dimension de l'embedding global MoCo (projection).")
    p.add_argument("--sem_tok_dim", type=int, default=256,
                   help="Dimension des tokens sémantiques (JEPA-content).")
    p.add_argument("--sem_queue", type=int, default=65536,
                   help="Taille de la queue MoCo (nombre de négatifs).")
    p.add_argument("--sem_m", type=float, default=0.999,
                   help="Momentum EMA pour l'encodeur MoCo (key encoder).")
    p.add_argument("--sem_t", type=float, default=0.2,
                   help="Température MoCo (InfoNCE).")

    # --- Transformations/augmentations pour la branche sémantique ---
    p.add_argument("--sem_use_aug", action="store_true",
                   help="Applique des augmentations SimCLR-like pour le contraste (en plus de la vue far).")
    p.add_argument("--sem_crop", type=int, default=224,
                   help="Taille de crop pour RandomResizedCrop (branche sémantique).")
    p.add_argument("--sem_min_scale", type=float, default=0.5,
                   help="Scale min pour RandomResizedCrop.")
    p.add_argument("--sem_color_jitter", type=float, default=0.4,
                   help="Amplitude du ColorJitter (brightness/contrast/saturation).")
    p.add_argument("--sem_gray", type=float, default=0.2,
                   help="Probabilité de RandomGrayscale.")
    p.add_argument("--sem_blur", type=float, default=0.1,
                   help="Probabilité de GaussianBlur.")

    return p.parse_args()
