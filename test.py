#!/usr/bin/env python
# -------------------------------------------------------------
#  test.py – t-SNE interactif, metrics, sup_predict,
#                       style transfer, cls_tokens, detect_transformer
# -------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset



from tests.functions_for_test import (
    compute_style_embeddings,
    compute_sem_embeddings,
    build_sem_backbone_for_eval,
    load_models,
    compute_metrics,
    compute_embeddings_with_paths,
    plot_tsne_interactive,
    build_test_dataloader,
    plot_tsne_cluster_knn,
    compute_gradcam_supheads,
    COLORMAP_DICT,
    _list_images_sorted,
    _infer_img_size_from_cfg,
    _make_transform,
    _load_img_tensor,
    _save_tensor_as_image,
    spectral_noise_like,
    # nouveaux helpers détection

)





def main() -> None:
    # ------------------------ ARGUMENTS ------------------------
    ap = argparse.ArgumentParser(
        "Exploration : t-SNE / métriques / sup_predict / style_transfer / detect_transformer"
    )

    # ckpts / modèles -------------------------------------------------------
    ap.add_argument("--cfg",         required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--sup_ckpt")
    ap.add_argument("--sup_in_dim", type=int)
    # ckpts séparés pour GA/GB (style GAN)
    ap.add_argument("--ckpt_GA")
    ap.add_argument("--ckpt_GB")

    # données --------------------------------------------------------------
    ap.add_argument("--data_json")
    ap.add_argument("--classes_json")
    ap.add_argument("--search_folder")
    ap.add_argument("--find_images_by_sub_folder", action="store_true")
    ap.add_argument("--data", help="Dossier racine au format ImageFolder (sous-dossiers = classes)")

    # modes principaux ------------------------------------------------------
    ap.add_argument(
        "--mode",
        choices=["tsne_interactive", "passe_by_metrics", "sup_predict", "style_transfer",
                 "detect_transformer"],
        default="tsne_interactive",
    )

    # source des features pour tsne / metrics / sup_predict
    ap.add_argument(
        "--feature_mode",
        choices=["style", "cls_tokens", "sem_resnet50"],
        default="style",
        help="Source des embeddings pour tsne/metrics/sup_predict : "
             "'style' (GAN), 'cls_tokens' (SupHeads/per_task) ou 'sem_resnet50' (backbone sémantique).",
    )


    # Source utilisée pour charger/entraîner SupHeads (utile pour auto-load du backbone sémantique)
    ap.add_argument(
        "--sup_feat_source",
        choices=["generator", "sem_resnet50"],
        default="generator",
        help="Source des features attendues par SupHeads. Si sem_resnet50, on peut auto-charger le backbone sémantique depuis weights_dir (SemBackbone_epoch*.pt).",
    )

    ap.add_argument(
        "--per_task",
        action="store_true",
        help="Utiliser les embeddings SupHeads par tâche (k-NN/cluster par tâche).",
    )

    # --- t-SNE : utiliser les embeddings produits par SupHeads ----------------
    # Par défaut, en feature_mode=style et per_task=False, la t-SNE utilise les
    # signatures de style extraites du générateur (tokens / maps) sans passer par
    # les têtes supervisées.
    #
    # Quand les poids SupHeads sont disponibles, on peut vouloir visualiser une
    # représentation *façonnée par SupHeads* (même espace que celui utilisé pour
    # la classification / multi-tâches). Cette option force alors la branche
    # per_task/composite : on passe par Wrap(G,Sup) et on récupère les embeddings
    # renvoyés par SupHeads(return_task_embeddings=True).
    ap.add_argument(
        "--tsne_use_supheads",
        action="store_true",
        help=(
            "En mode --mode tsne_interactive : utiliser les embeddings produits par SupHeads "
            "(return_task_embeddings=True) au lieu des signatures de style brutes. "
            "Requiert --sup_ckpt (SupHeads chargé)."
        ),
    )

    # --- Backbone sémantique (utilisé quand feature_mode=sem_resnet50) ---
    ap.add_argument(
        "--det_sem_backbone",
        type=str,
        default="resnet50",
        help="Architecture du backbone sémantique (resnet50/resnet101/resnet152).",
    )
    ap.add_argument(
        "--det_sem_return_layer",
        type=str,
        default="layer4",
        help="Couche à récupérer (layer2/layer3/layer4).",
    )
    ap.add_argument(
        "--sem_pretrained",
        type=int,
        default=1,
        help="1=utiliser des poids pré-entraînés (ImageNet ou --sem_pretrained_path).",
    )
    ap.add_argument(
        "--sem_pretrained_path",
        type=str,
        default="",
        help="Chemin optionnel vers un checkpoint pour initialiser le backbone sémantique.",
    )
    ap.add_argument(
        "--sem_pretrained_strict",
        type=int,
        default=0,
        help="Chargement strict du checkpoint sémantique (0/1).",
    )
    ap.add_argument(
        "--sem_pretrained_verbose",
        type=int,
        default=1,
        help="Affiche les infos de chargement du backbone sémantique (0/1).",
    )
    ap.add_argument(
        "--sem_imagenet_norm",
        type=int,
        default=1,
        help="1=convertit [-1,1]→[0,1] puis normalise ImageNet avant ResNet (recommandé).",
    )
    ap.add_argument("--num_samples", type=int)

    # embeddings -----------------------------------------------------------
    ap.add_argument("--token_pool", choices=["mean", "max", "none"], default="mean")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument(
        "--embed_type",
        choices=[
            "tokG",
            "tok6", "tok6_mean", "tok6_w",
            "style_tok",
            "bot", "bot+tok",
            "tok+delta",
            "mgap", "mgap+tok",
            "cont_tok",        # tokens de contenu + simple mean pooling
            "cont_tok_vit",    # tokens de contenu passés dans la tête ViT
        ],
        default="tok6",
    )

    ap.add_argument(
        "--delta_weights",
        type=str,
        default="1,1,1,1,1",
        help=(
            "Poids par échelle. "
            "Pour 'tok6_w' : 6 poids 'wG,w5,w4,w3,w2,w1'. "
            "Pour 'tok+delta' / 'mgap' / 'mgap+tok' : 5 poids 'w5,w4,w3,w2,w1'."
        ),
    )

    # métriques ------------------------------------------------------------
    ap.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Liste des métriques: knn,proto,sep,cluster,retrieval,cluster_mknn,"
             "cluster_spectral,cluster_dbscan,indices",
    )
    ap.add_argument("--pca_dim", type=int, default=None)
    ap.add_argument("--l2_norm", action="store_true")

    # divers --------------------------------------------------------------
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)

    # --- Options prédiction + Grad-CAM ---
    ap.add_argument(
        "--prob_threshold",
        type=float,
        default=-1.0,
        help="Si >0: probas < seuil -> étiquette -1 (Unknown)",
    )
    ap.add_argument(
        "--visualize_gradcam",
        action="store_true",
        help="Active le calcul/sauvegarde des Grad-CAM",
    )
    ap.add_argument(
        "--save_gradcam_images",
        action="store_true",
        help="Sauvegarde les images Grad-CAM sur disque (avec overlay).",
    )

    ap.add_argument(
        "--save_test_images",
        action="store_true",
        help="Sauvegarde une image annotée (GT/Pred/Prob) pour chaque échantillon.",
    )
    ap.add_argument(
        "--gradcam_task",
        type=str,
        default=None,
        help="Nom exact de la tâche pour Grad-CAM (par défaut: 1ère tâche des SupHeads).",
    )
    ap.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Colormap OpenCV pour Grad-CAM (jet, hot, inferno, magma, plasma, turbo)",
    )
    ap.add_argument(
        "--gradcam_source",
        choices=["style", "content"],
        default="style",
        help="Espace de la carte cible Grad-CAM : 'style' (t1..t5) ou 'content' (skip/bot).",
    )
    ap.add_argument(
        "--gradcam_level",
        type=str,
        default="t1",
        help="Si --gradcam_source=style : t1|t2|t3|t4|t5 ; si content : skip16|skip32|skip64|bot",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Dossier de sortie (par défaut: weights_dir/sup_predict)",
    )

    # --- Options style transfer ------------------------------------------
    ap.add_argument(
        "--style_A",
        type=str,
        default=None,
        help="Chemin vers une image ou un dossier d'images STYLE (A).",
    )
    ap.add_argument(
        "--content_B",
        type=str,
        default=None,
        help="Chemin vers une image ou un dossier d'images CONTENU (B).",
    )
    ap.add_argument(
        "--transfer_via",
        choices=["GA", "GB"],
        default="GA",
        help="Transférer le style via G_A (recommandé) ou via pipeline G_B (G_A puis G_B).",
    )
    ap.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Redimension à l'entrée (défaut: tente cfg → sinon 256).",
    )
    ap.add_argument(
        "--style_gain_A",
        type=float,
        default=3.5,
        help="Gain appliqué aux tokens quand on utilise G_A.",
    )
    ap.add_argument(
        "--style_gain_B",
        type=float,
        default=-5e-9,
        help="Gain appliqué aux tokens quand on utilise G_B.",
    )
    ap.add_argument(
        "--gb_spectral_noise",
        action="store_true",
        help="Ajoute un bruit spectral sur l'entrée de G_B (comme en phase B).",
    )
    ap.add_argument(
        "--gb_noise_sigma",
        type=float,
        default=0.02,
        help="Sigma du bruit spectral (amplitude FFT).",
    )
    ap.add_argument(
        "--gb_noise_gamma",
        type=float,
        default=1.0,
        help="Exposant sur le bruit spectral (adoucit/renforce).",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Dossier de sortie. Défaut: weights_dir/style_transfer_via_<GA|GB>",
    )

    # --- Options détection transformer -----------------------------------
    ap.add_argument(
        "--detect_ckpt",
        type=str,
        default=None,
        help="Checkpoint pour le modèle de détection (si --mode detect_transformer).",
    )
    ap.add_argument(
        "--detect_iou",
        type=float,
        default=0.5,
        help="Seuil IoU pour les métriques de détection.",
    )
    ap.add_argument(
        "--detect_score",
        type=float,
        default=0.5,
        help="Seuil de confiance pour les métriques / affichage des boxes.",
    )
    ap.add_argument(
        "--detect_max_dets",
        type=int,
        default=100,
        help="Nombre max de prédictions conservées par image pour les métriques.",
    )
    ap.add_argument(
        "--detect_camera",
        action="store_true",
        help="En mode detect_transformer : si activé, lit depuis la caméra au lieu du dataset.",
    )
    ap.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Index de la caméra OpenCV (0 par défaut).",
    )
    ap.add_argument('--det_head_type', type=str, default='simple_unet',
                   help="Tête de détection: 'simple_unet' (UNet+SimpleDETRHead) ou 'detr_resnet50'")

    # --- Options dataset de détection (test) -----------------------------
    ap.add_argument(
        "--det_dataset",
        type=str,
        default="coco",
        help="Nom du dataset de détection (ex: 'coco', 'bdd', ...). "
             "Utilisé par build_detection_dataloader.",
    )
    ap.add_argument(
        "--det_img_root",
        type=str,
        default=None,
        help=(
            "Racine des images pour le test de détection "
            "(ex: '/path/to/coco/val2017')."
        ),
    )
    ap.add_argument(
        "--det_ann_file",
        type=str,
        default=None,
        help=(
            "Fichier d’annotations pour le test de détection "
            "(ex: COCO JSON: 'instances_val2017.json')."
        ),
    )
    ap.add_argument(
        "--det_split",
        type=str,
        default="val",
        help="Nom du split de test (ex: 'val', 'test'). "
             "Utilisé par build_detection_dataloader si besoin.",
    )
    ap.add_argument(
        "--detect_label_map",
        type=str,
        default=None,
        help=(
            "Optionnel: JSON avec un mapping {id: nom_classe} pour l’affichage "
            "des catégories en détection (prioritaire sur la config)."
        ),
    )

    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = json.load(open(args.cfg))
    wdir = Path(args.weights_dir)

    # Si on veut explicitement travailler sur les cls_tokens,
    # on force per_task (embeddings par tâche via SupHeads).
    if args.feature_mode == "cls_tokens":
        args.per_task = True

    # Quand on travaille avec la source sémantique, on veut reproduire le comportement
    # de la branche style :
    #   - sans --tsne_use_supheads : t-SNE / metrics sur les features ResNet brutes
    #   - avec --tsne_use_supheads : on passe ces features dans SupHeads et on visualise
    #     l'espace "façonné" par SupHeads (per_task=True)
    if args.feature_mode == "sem_resnet50" and args.tsne_use_supheads:
        args.per_task = True

    # Option explicite : pour les modes t-SNE/metrics, utiliser les embeddings produits
    # par SupHeads (utile même si feature_mode=style, tant que SupHeads est chargé).
    # Cela s'applique à :
    #  - tsne_interactive : affichage interactif
    #  - passe_by_metrics : t-SNE + vues additionnelles (cluster/knn) basées sur des métriques
    if args.mode in ("tsne_interactive", "passe_by_metrics") and args.tsne_use_supheads:
        args.per_task = True

    # ---------------------- MODE: DETECT_TRANSFORMER ----------------------
    # ---------------------- MODE: DETECT_TRANSFORMER ----------------------
    if args.mode == "detect_transformer":
        # On peut forcer le head_type dans la cfg si on veut,
        # uniquement si le ckpt n'a pas déjà des hparams.
        # Cela aide pour des vieux checkpoints sans 'hparams'.
        cfg.setdefault("detector", {})
        if "head_type" not in cfg["detector"]:
            cfg["detector"]["head_type"] = args.det_head_type

        # Chargement du modèle de détection (backbone + tête)
        det_model, label_map = load_detection_model(
            weights_dir=wdir,
            device=device,
            cfg=cfg,
            ckpt=args.detect_ckpt,
        )

        # Surcharge éventuelle du label_map via un JSON externe
        if args.detect_label_map is not None:
            lm_path = Path(args.detect_label_map)
            if not lm_path.is_absolute():
                lm_path = (wdir / lm_path).resolve()
            if lm_path.exists():
                try:
                    with open(lm_path, "r") as f:
                        label_map = json.load(f)
                    print(f"[detect_transformer] label_map chargé depuis {lm_path}")
                except Exception as e:
                    print(f"[WARN] Impossible de charger detect_label_map={lm_path}: {e}")
            else:
                print(f"[WARN] detect_label_map inexistant: {lm_path}")

        if args.detect_camera:
            # Mode caméra : affichage temps réel
            run_detection_on_camera(
                det_model,
                device=device,
                label_map=label_map,
                score_thresh=args.detect_score,
                cam_index=args.camera_index,
            )
        else:
            # Mode évaluation offline sur un dataset de détection.
            #
            # build_detection_dataloader(args, cfg) doit utiliser :
            #   - args.det_dataset   (ex: 'coco')
            #   - args.det_img_root  (ex: '/path/to/coco/val2017')
            #   - args.det_ann_file  (ex: '/path/to/instances_val2017.json')
            #   - args.det_split     (ex: 'val')
            det_loader = build_detection_dataloader(args, cfg)
            stats = compute_detection_metrics(
                det_model,
                det_loader,
                device=device,
                iou_thresh=args.detect_iou,
                score_thresh=args.detect_score,
                max_dets=args.detect_max_dets,
            )
            print(json.dumps(stats, indent=2))

        return

    # ---------------------- MODE STYLE / CLS TOKENS -----------------------
    # contrôle des poids pour 'tok6_w'
    if args.embed_type == "tok6_w":
        try:
            n_w = len([t for t in args.delta_weights.split(",") if t.strip()])
            if n_w != 6:
                print(
                    f"[WARN] --delta_weights doit avoir 6 valeurs pour tok6_w "
                    f"(wG,w5,w4,w3,w2,w1) ; reçu {n_w}."
                )
        except Exception:
            print("[WARN] --delta_weights invalide pour tok6_w ; utilisé tel quel.")

    # ==============================================================
    # Chargement des modèles
    #
    # IMPORTANT:
    # - Quand --feature_mode sem_resnet50, les modes de test (tsne_interactive,
    #   passe_by_metrics, sup_predict) n'ont **pas** besoin de charger G_A/G_B.
    #   On charge uniquement le backbone sémantique + (optionnellement) SupHeads.
    # - Sinon (style / cls_tokens / style_transfer), on charge le générateur.
    # ==============================================================

    def _infer_tasks_and_in_dim_from_sup_state(sd: dict) -> tuple[dict[str, int] | None, int | None]:
        """Infère {task: n_classes} et in_dim depuis les poids SupHeads.
        Compatible avec:
          - keys préfixées 'sup_heads.' (bundle)
          - keys directes 'classifiers.<task>....'
        """
        tasks_auto: dict[str, int] = {}
        in_dim_auto: int | None = None

        # Normaliser: enlever 'sup_heads.' si présent
        for k, v in sd.items():
            kk = k[len("sup_heads."):] if k.startswith("sup_heads.") else k
            if not kk.startswith("classifiers."):
                continue
            if not isinstance(v, torch.Tensor) or v.dim() != 2:
                continue
            # classifiers.<task>.<...>.weight  OR  classifiers.<task>.weight
            rest = kk[len("classifiers."):]
            task, _, tail = rest.partition(".")
            if not task:
                continue
            if not (tail.endswith("weight") or tail == "weight"):
                continue
            n_cls, in_d = int(v.shape[0]), int(v.shape[1])
            # on prend le plus petit out (souvent la couche finale)
            if task not in tasks_auto or n_cls < tasks_auto[task]:
                tasks_auto[task] = n_cls
                if in_dim_auto is None:
                    in_dim_auto = in_d

        return (tasks_auto if tasks_auto else None), in_dim_auto

    def _read_state_any(path: Path, dev: torch.device) -> dict:
        path = Path(path)
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device=str(dev))
        obj = torch.load(path, map_location=dev)
        return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    def _find_latest_sem_ckpt(weights_dir: Path, sem_filename: str = "SemBackbone") -> Path | None:
        pats = list(weights_dir.glob(f"{sem_filename}_epoch*.pt")) + list(weights_dir.glob(f"{sem_filename}_epoch*.safetensors"))
        if not pats:
            return None
        pats.sort(key=lambda p: p.stat().st_mtime)
        return pats[-1]

    # ----------- Backbone sémantique (optionnel, pour tests) -----------
    sem_backbone = None
    sem_out_ch = None
    if args.feature_mode == "sem_resnet50":
        sem_backbone, sem_out_ch = build_sem_backbone_for_eval(
            device=device,
            arch=args.det_sem_backbone,
            return_layer=args.det_sem_return_layer,
            pretrained=bool(int(args.sem_pretrained)),
            pretrained_path=str(args.sem_pretrained_path or ""),
            strict=bool(int(args.sem_pretrained_strict)),
            verbose=bool(int(args.sem_pretrained_verbose)),
            weights_dir=(wdir if args.sup_feat_source == "sem_resnet50" else None),
            sem_filename="SemBackbone",
        )
        print(f"✓ sem_backbone prêt | arch={args.det_sem_backbone} | layer={args.det_sem_return_layer} | out_ch={sem_out_ch}")

    # ----------- Charger G/Sup selon le mode -----------
    G, Sup, task_cls = None, None, None

    sem_only_modes = {"tsne_interactive", "passe_by_metrics", "sup_predict"}
    if (args.feature_mode == "sem_resnet50") and (args.mode in sem_only_modes):
        # Pas besoin de G_A/G_B.
        # Charger SupHeads si requis (sup_predict) ou si demandé (tsne_use_supheads / per_task)
        need_sup = (args.mode == "sup_predict") or bool(args.tsne_use_supheads) or bool(args.per_task)
        if need_sup:
            if not args.sup_ckpt:
                raise RuntimeError("SupHeads requis mais --sup_ckpt manquant (utilise sup_predict/tsne_use_supheads/per_task).")
            from models.sup_heads import SupHeads

            sup_path = Path(args.sup_ckpt)
            if not sup_path.is_absolute():
                # Permet --sup_ckpt relatif à weights_dir
                cand = wdir / sup_path
                if cand.exists():
                    sup_path = cand
            if not sup_path.exists():
                raise FileNotFoundError(f"SupHeads introuvable: {sup_path}")

            sd = _read_state_any(sup_path, device)
            tasks_auto, in_dim_auto = _infer_tasks_and_in_dim_from_sup_state(sd)
            # Si l'utilisateur donne classes_json, on peut préférer cette structure.
            if (tasks_auto is None) and args.classes_json:
                with open(args.classes_json, "r", encoding="utf-8") as f:
                    cj = json.load(f)
                # attendu: {task: [class_names...]}
                if isinstance(cj, dict) and cj:
                    tasks_auto = {str(t): len(v) for t, v in cj.items() if isinstance(v, (list, tuple))}

            if tasks_auto is None:
                raise RuntimeError("Impossible d'inférer les tâches depuis SupHeads. Fournis --classes_json ou un sup_ckpt valide.")

            in_dim = int(args.sup_in_dim or (sem_out_ch or in_dim_auto or 2048))
            # Sur sem_resnet, on veut un mode 'flat' (pas de mixers multi6)
            Sup = SupHeads(tasks_auto, in_dim, token_mode="flat").to(device)

            # Charger uniquement les clés SupHeads (tolérant)
            # - sd peut contenir 'sup_heads.' prefix
            sd_clean = { (k[len("sup_heads."):] if k.startswith("sup_heads.") else k): v for k, v in sd.items() }
            missing, unexpected = Sup.load_state_dict(sd_clean, strict=False)
            if missing:
                print(f"[WARN] SupHeads: missing keys ({len(missing)})")
            if unexpected:
                print(f"[WARN] SupHeads: unexpected keys ({len(unexpected)})")
            Sup.eval()
            print(f"✓ SupHeads loaded (sem_resnet50) – {sup_path.name} | tasks={list(tasks_auto.keys())} | in_dim={in_dim}")

    else:
        # Chargement standard (style / cls_tokens / style_transfer)
        G, Sup, task_cls = load_models(
            weights_dir=wdir,
            device=device,
            cfg=cfg,
            ckpt_gen=args.ckpt,
            sup_ckpt=args.sup_ckpt,
            classes_json=args.classes_json,
            sup_in_dim=args.sup_in_dim,
            ckpt_GA=args.ckpt_GA,
            ckpt_GB=args.ckpt_GB,
        )

    # Récupération GA/GB uniquement si un générateur est chargé
    if G is not None:
        G_A = getattr(G, "GA", getattr(G, "G_A", G))
        G_B = getattr(G, "GB", getattr(G, "G_B", G))
    else:
        G_A = None
        G_B = None

    # DataLoader de test (gère --data_json / --data et renvoie ImageFolder si --data)
    loader, dataset, dataset_type = build_test_dataloader(args, cfg)

    # Subset optionnel
    if args.num_samples and args.num_samples < len(dataset):
        idx = random.sample(range(len(dataset)), args.num_samples)
        sub_ds = Subset(dataset, idx)
        loader = DataLoader(
            sub_ds, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False
        )
        print(f"✓ Subset : {len(idx)} images")

    # --------------------- Wrapper G + SupHeads ---------------------------
    class Wrap(nn.Module):
        """
        Wrapper 'léger' pour exploiter un générateur feuille (GA si dispo) + SupHeads
        sans créer de cycles nn.Module, et avec un eval()/train() non récursif.

        - G : peut être un conteneur avec attributs GA/G_A/GB/G_B ou un seul générateur.
        - Sup : module de têtes supervisées (optionnel).
        - embed_type / delta_weights : config des features (tok6_w, mgap, etc.)
        """

        @staticmethod
        def _deregister_child_as_plain(parent: nn.Module, attr: str):
            """Si parent.attr est un sous-module enregistré, on le retire de _modules et on remet l'attribut brut."""
            if not isinstance(parent, nn.Module) or not hasattr(parent, attr):
                return
            ch = getattr(parent, attr)
            if isinstance(ch, nn.Module):
                if attr in getattr(parent, "_modules", {}):
                    parent._modules.pop(attr, None)
                object.__setattr__(parent, attr, ch)

        @staticmethod
        def _pick_leaf_generator(G):
            """Retourne un 'leaf' generator : priorité à GA/G_A s'ils existent, sinon G."""
            cand_names = ("GA", "G_A")
            for n in cand_names:
                if hasattr(G, n) and isinstance(getattr(G, n), nn.Module):
                    return getattr(G, n)
            return G

        @staticmethod
        def _set_mode_no_recurse(mod: nn.Module, *, train: bool):
            """Change le flag .training de mod et de ses enfants **sans** appeler .train()/.eval() récursifs."""
            if not isinstance(mod, nn.Module):
                return
            seen = set()
            stack = [mod]
            while stack:
                x = stack.pop()
                idx = id(x)
                if idx in seen or not isinstance(x, nn.Module):
                    continue
                seen.add(idx)
                object.__setattr__(x, "training", bool(train))
                for ch in getattr(x, "_modules", {}).values():
                    if isinstance(ch, nn.Module):
                        stack.append(ch)

        def __init__(
            self,
            G,
            Sup=None,
            *,
            embed_type: str | None = None,
            delta_weights: str | list | tuple = "1,1,1,1,1",
        ):
            super().__init__()

            # On garde des références brutes, sans les enregistrer comme sous-modules
            object.__setattr__(self, "_G_ref", G)
            object.__setattr__(self, "_Sup_ref", Sup)

            # Si G est un conteneur, on dé-enregistre GA/GB
            for name in ("GA", "G_A", "GB", "G_B"):
                self._deregister_child_as_plain(self._G_ref, name)

            # Sélection d'un générateur feuille
            leaf = self._pick_leaf_generator(self._G_ref)
            object.__setattr__(self, "_G_leaf", leaf)

            # Gestion des tâches (si Sup fourni)
            self.tasks = (
                list(Sup.tasks.keys())
                if (Sup is not None and hasattr(Sup, "tasks"))
                else ["__DEFAULT__"]
            )

            self.feat_type_forced = (embed_type or "").strip() or None
            if isinstance(delta_weights, (list, tuple)):
                self.delta_weights = ",".join(str(float(x)) for x in delta_weights)
            else:
                self.delta_weights = str(delta_weights)

            # Dim d'entrée requise par Sup (si connue)
            self.required_dim = getattr(Sup, "in_dim", None) if Sup is not None else None
            if (
                self.required_dim is None
                and Sup is not None
                and hasattr(Sup, "classifiers")
                and len(Sup.classifiers) > 0
            ):
                first_clf = next(iter(Sup.classifiers.values()))
                self.required_dim = int(first_clf.weight.shape[1])

            # Choix feat_type
            if self.feat_type_forced:
                self.feat_type = self.feat_type_forced
                if self.feat_type == "tok6_w":
                    parts = [t for t in self.delta_weights.split(",") if t.strip()]
                    if len(parts) not in (0, 6):
                        self.delta_weights = "1,1,1,1,1,1"
            else:
                # On inclut cont_tok et cont_tok_vit dans les candidats possibles
                candidates = [
                    "tok6_w",
                    "tok6",
                    "tok6_mean",
                    "style_tok",
                    "tok+delta",
                    "mgap+tok",
                    "bot+tok",
                    "mgap",
                    "bot",
                    "cont_tok",
                    "cont_tok_vit",
                ]
                self.feat_type = self._pick_feat_type_by_dim(candidates, self.delta_weights)

            in_dim = self._probe_dim(self.feat_type, self.delta_weights)
            if (
                self.required_dim is not None
                and in_dim is not None
                and in_dim != self.required_dim
            ):
                print(
                    f"[WARN] Wrap: dim(features)={in_dim} != Sup.in_dim={self.required_dim} "
                    f"pour feat_type='{self.feat_type}'."
                )
            print(
                f"✓ Wrap : feat_type='{self.feat_type}'  (Sup.in_dim={self.required_dim})  "
                f"Δw={self.delta_weights}"
            )

        # ------------------------- Accès simples -------------------------
        @property
        def G(self):
            return self._G_ref

        @property
        def G_leaf(self):
            return self._G_leaf

        @property
        def Sup(self):
            return self._Sup_ref

        # ------------------------- Mode (non récursif) -------------------
        def eval(self):
            object.__setattr__(self, "training", False)
            self._set_mode_no_recurse(self._G_leaf, train=False)
            if self._G_ref is not self._G_leaf:
                self._set_mode_no_recurse(self._G_ref, train=False)
            if self.Sup is not None:
                self._set_mode_no_recurse(self.Sup, train=False)
            return self

        def train(self, mode: bool = True):
            object.__setattr__(self, "training", bool(mode))
            self._set_mode_no_recurse(self._G_leaf, train=bool(mode))
            if self._G_ref is not self._G_leaf:
                self._set_mode_no_recurse(self._G_ref, train=bool(mode))
            if self.Sup is not None:
                self._set_mode_no_recurse(self.Sup, train=bool(mode))
            return self

        # ------------------------- Sélection feat_type -------------------
        def _probe_dim(self, ft: str, dw: str | None) -> int | None:
            try:
                if hasattr(self.G_leaf, "sup_in_dim_for"):
                    d = self.G_leaf.sup_in_dim_for(ft)
                    if isinstance(d, int) and d > 0:
                        return d
            except Exception:
                pass
            try:
                dev = next(self.G_leaf.parameters()).device
            except Exception:
                dev = torch.device("cpu")
            x = torch.zeros(1, 3, 256, 256, device=dev)
            try:
                f = self._sup_features_from_G(x, feat_type=ft, delta_weights=dw)
                return int(f.shape[1])
            except Exception:
                return None

        def _pick_feat_type_by_dim(self, candidates, delta_w: str) -> str:
            if self.required_dim is None:
                for ft in candidates:
                    if self._probe_dim(ft, delta_w) is not None:
                        return ft
                return "style_tok"
            for ft in candidates:
                d = self._probe_dim(ft, delta_w)
                if d == self.required_dim:
                    return ft
            for ft in candidates:
                if self._probe_dim(ft, delta_w) is not None:
                    return ft
            return "style_tok"

        # ------------------------- Extraction de features ---------------
        def _sup_features_from_G(
            self,
            imgs: torch.Tensor,
            feat_type: str | None = None,
            delta_weights: str | None = None,
            return_intermediates: bool = False,
        ):
            Gnet = self.G_leaf
            ft = (feat_type or self.feat_type)
            dw = delta_weights if delta_weights is not None else self.delta_weights

            # IMPORTANT : pour cont_tok / cont_tok_vit on s'attend à ce que Gnet.sup_features
            # soit implémenté côté modèle. Si ce n'est pas le cas, le fallback ci-dessous
            # utilisera 'bot' (global content).
            if hasattr(Gnet, "sup_features") and not return_intermediates:
                return Gnet.sup_features(imgs, ft, delta_weights=dw)

            # Fallback générique (encode_content + style_enc)
            z, skips = Gnet.encode_content(imgs)
            se = Gnet.style_enc(imgs)

            maps, toks, tokG = None, None, None
            if isinstance(se, (list, tuple)) and len(se) == 3:
                maps, toks, tokG = se
            elif isinstance(se, (list, tuple)) and len(se) == 2:
                a, b = se
                if (
                    isinstance(a, (list, tuple))
                    and len(a)
                    and a[0].dim() == 4
                    and hasattr(b, "dim")
                    and b.dim() == 2
                ):
                    maps, tokG = a, b
                elif (
                    isinstance(a, (list, tuple))
                    and len(a)
                    and a[0].dim() == 2
                    and hasattr(b, "dim")
                    and b.dim() == 2
                ):
                    toks, tokG = a, b
            elif (
                isinstance(se, (list, tuple))
                and len(se) == 1
                and hasattr(se[0], "dim")
                and se[0].dim() == 2
            ):
                tokG = se[0]
            elif hasattr(se, "dim") and se.dim() == 2:
                tokG = se

            bot = F.adaptive_avg_pool2d(z, 1).flatten(1)

            def _l2(t):
                return t / (t.norm(dim=1, keepdim=True) + 1e-8)

            if ft in ("tok6_w", "tok6", "tok6_mean", "style_tok"):
                if tokG is None and (toks is None or len(toks) == 0):
                    feats = bot
                else:
                    if tokG is None and toks is not None and len(toks) > 0:
                        tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])
                    seq = [_l2(tokG)] + (
                        [_l2(t) for t in toks] if toks is not None else []
                    )
                    if ft == "style_tok":
                        feats = seq[0]
                    elif ft == "tok6":
                        feats = torch.cat(seq, dim=1)
                    elif ft == "tok6_mean":
                        feats = torch.stack(seq, dim=1).mean(1)
                    else:  # tok6_w
                        parts = [t for t in (dw or "").split(",") if t.strip()]
                        if len(parts) != 6:
                            parts = ["1", "1", "1", "1", "1", "1"]
                        w = torch.as_tensor(
                            [float(x) for x in parts],
                            device=imgs.device,
                            dtype=seq[0].dtype,
                        )
                        w = (w / (w.sum() + 1e-8)).view(1, -1, 1)
                        S = torch.stack(seq, dim=1)
                        feats = (S * w).sum(1)

            elif ft == "bot":
                feats = bot

            elif ft in ("bot+tok", "mgap+tok"):
                feats = bot if tokG is None else torch.cat([bot, tokG], dim=1)

            elif ft == "mgap":
                if maps is None:
                    feats = bot
                else:
                    pool = lambda m: F.adaptive_avg_pool2d(m, 1).flatten(1)
                    w5 = [
                        float(x)
                        for x in (dw.split(",") if dw else [])
                        if x.strip()
                    ]
                    if len(w5) != 5:
                        w5 = [1, 1, 1, 1, 1]
                    mg = [wi * pool(mi) for wi, mi in zip(w5, maps)]
                    feats = torch.cat(mg, dim=1)

            elif ft == "tok+delta":
                if maps is None or tokG is None:
                    feats = tokG if tokG is not None else bot
                else:
                    pool = lambda m: F.adaptive_avg_pool2d(m.abs(), 1).flatten(1)
                    w5 = [
                        float(x)
                        for x in (dw.split(",") if dw else [])
                        if x.strip()
                    ]
                    if len(w5) != 5:
                        w5 = [1, 1, 1, 1, 1]
                    dvec = torch.cat(
                        [wi * pool(mi) for wi, mi in zip(w5, maps)], dim=1
                    )
                    feats = torch.cat([tokG, dvec], dim=1)

            elif ft in ("cont_tok", "cont_tok_vit"):
                # Fallback minimal : on renvoie le 'bot' (content global).
                # Le cas idéal reste d'implémenter ces types dans G.sup_features.
                feats = bot

            else:
                feats = bot

            if return_intermediates:
                cache = {
                    "z": z,
                    "skips": skips,
                    "maps": maps,
                    "toks": toks,
                    "tokG": tokG,
                }
                return feats, cache
            return feats

        # ------------------------- Forward -------------------------
        def forward(
            self,
            imgs: torch.Tensor,
            *,
            return_task_embeddings: bool = False,
            return_embeddings: bool = False,
        ):
            feats = self._sup_features_from_G(imgs)

            if return_task_embeddings:
                if self.Sup is None:
                    return None, {"__DEFAULT__": feats}
                try:
                    _, embs = self.Sup(feats, return_task_embeddings=True)
                except TypeError:
                    logits = self.Sup(feats)
                    embs = (
                        {t: v for t, v in logits.items()}
                        if isinstance(logits, dict)
                        else {"__DEFAULT__": logits}
                    )
                return None, embs

            if return_embeddings or (self.Sup is None):
                return feats

            out = self.Sup(feats)
            return out

    composite = None
    if Sup is not None:
        if args.feature_mode == "sem_resnet50":
            # Wrapper léger : SupHeads consomme directement les features du backbone sémantique
            class _SemAdapter(nn.Module):
                """Expose des features sémantiques via une API G.sup_features compatible."""
                def __init__(self, sem_backbone: nn.Module, imagenet_norm: bool = True):
                    super().__init__()
                    self.sem_backbone = sem_backbone
                    self.imagenet_norm = bool(imagenet_norm)
                    self.register_buffer("_im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
                    self.register_buffer("_im_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

                def sup_features(self, imgs: torch.Tensor, feat_type: str = "sem_resnet50", delta_weights: str | None = None):
                    x = imgs
                    if x.dim() != 4:
                        raise ValueError(f"SemAdapter.sup_features: expected BCHW, got {tuple(x.shape)}")
                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)
                    if self.imagenet_norm:
                        # images du pipeline GAN: souvent [-1,1]
                        x = (x + 1.0) * 0.5
                        x = x.clamp(0.0, 1.0)
                        x = (x - self._im_mean) / self._im_std
                    out = self.sem_backbone(x)
                    if isinstance(out, dict):
                        feat = out.get("0", None)
                        if feat is None:
                            feat = next(iter(out.values()))
                    else:
                        feat = out
                    # GAP
                    feat = feat.mean(dim=(2, 3))
                    return feat

            class SemWrap(nn.Module):
                """Composite compatible avec compute_embeddings_with_paths (attend .G.sup_features + .Sup)."""
                def __init__(self, sem_backbone: nn.Module, Sup: nn.Module, imagenet_norm: bool = True):
                    super().__init__()
                    self.G = _SemAdapter(sem_backbone, imagenet_norm=imagenet_norm)
                    self.Sup = Sup
                    self.tasks = list(Sup.tasks.keys()) if hasattr(Sup, "tasks") else ["__DEFAULT__"]
                    self.feat_type = "sem_resnet50"
                    self.delta_weights = ""

                def sup_features(self, imgs: torch.Tensor):
                    return self.G.sup_features(imgs)

                def forward(self, imgs: torch.Tensor, *, return_task_embeddings: bool = False, return_embeddings: bool = False):
                    feats = self.G.sup_features(imgs)
                    if return_task_embeddings:
                        try:
                            _, embs = self.Sup(feats, return_task_embeddings=True)
                        except TypeError:
                            logits = self.Sup(feats)
                            embs = {t: v for t, v in logits.items()} if isinstance(logits, dict) else {"__DEFAULT__": logits}
                        return None, embs
                    if return_embeddings:
                        return feats
                    return self.Sup(feats)

            composite = SemWrap(sem_backbone, Sup, imagenet_norm=bool(int(args.sem_imagenet_norm))).to(device)
            print(f"✓ composite(sem) prêt | tasks={composite.tasks} | imagenet_norm={bool(int(args.sem_imagenet_norm))}")
        else:
            composite = Wrap(
                G, Sup, embed_type=args.embed_type, delta_weights=args.delta_weights
            ).to(device)
            print(
                f"✓ composite prêt | tasks={composite.tasks} | "
                f"feat_type={composite.feat_type} | Δw={composite.delta_weights}"
            )
    elif args.mode == "sup_predict":
        raise RuntimeError("Le mode 'sup_predict' requiert un SupHeads (--sup_ckpt).")

    # ========================= MODE: sup_predict ==========================
    if args.mode == "sup_predict":
        if Sup is None:
            raise RuntimeError(
                "Le mode 'sup_predict' requiert un SupHeads (fournis via --sup_ckpt)."
            )
        try:
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
            )

            _SK_OK = True
        except Exception:
            _SK_OK = False

        out_dir = Path(args.out_dir or (wdir / "sup_predict"))
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks_list = list(Sup.tasks.keys()) if hasattr(Sup, "tasks") else ["__DEFAULT__"]
        gradcam_task = args.gradcam_task or tasks_list[0]
        if gradcam_task not in tasks_list:
            raise ValueError(
                f"--gradcam_task '{gradcam_task}' n'existe pas. Tâches: {tasks_list}"
            )

        all_preds = {t: [] for t in tasks_list}
        all_probs = {t: [] for t in tasks_list}
        all_trues = {t: [] for t in tasks_list}
        times = []

        def _norm_key(s: str) -> str:
            return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")

        ds_for_names = (
            loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        )
        ds_task_cls = getattr(ds_for_names, "task_classes", {}) or {}
        nk_ds = {_norm_key(k): k for k in ds_task_cls.keys()}
        safe_task_cls_global = {}
        for t in tasks_list:
            ds_key = nk_ds.get(_norm_key(t), None)
            if ds_key is None and "__DEFAULT__" in ds_task_cls:
                ds_key = "__DEFAULT__"
            names = list(ds_task_cls.get(ds_key, [])) if ds_key is not None else []
            if not names:
                names = [f"class {i}" for i in range(32)]
            safe_task_cls_global[t] = names

        task_map = None
        idx_global = 0
        for b_idx, batch in enumerate(loader):
            t0 = time.time()
            if len(batch) == 3:
                imgs, raw, _paths = batch
            else:
                imgs, raw = batch
                _paths = None
            imgs = imgs.to(device)

            if task_map is None:
                if isinstance(raw, dict):
                    nk = {_norm_key(k): k for k in raw.keys()}
                    task_map = {t: nk.get(_norm_key(t), None) for t in tasks_list}
                else:
                    tdef = tasks_list[0]
                    task_map = {tdef: "__DEFAULT__"}
                print("   [sup_predict] task mapping →", task_map)

            with torch.no_grad():
                cam_feats_type = args.embed_type
                if args.feature_mode == "sem_resnet50":
                    # features sémantiques ResNet -> GAP
                    feats = composite.sup_features(imgs)
                else:
                    if hasattr(G, "sup_features"):
                        feats = G.sup_features(
                            imgs, cam_feats_type, delta_weights=args.delta_weights
                        )
                    else:
                        z, _ = G.encode_content(imgs)
                        feats = F.adaptive_avg_pool2d(z, 1).flatten(1)
                logits = Sup(feats)
                if not isinstance(logits, dict):
                    logits = {"__DEFAULT__": logits}

            B = imgs.size(0)
            lbls_list = []
            if isinstance(raw, dict):
                for i in range(B):
                    d = {}
                    for t in logits.keys():
                        ds_key = task_map.get(t)
                        if ds_key is None:
                            d[t] = -1
                        else:
                            v = raw.get(ds_key, None)
                            d[t] = -1 if (v is None or v[i] is None) else int(v[i])
                    lbls_list.append(d)
            else:
                t0_name = next(iter(logits.keys()))
                for i in range(B):
                    d = {
                        t0_name: int(
                            raw[i].item() if hasattr(raw[i], "item") else raw[i]
                        )
                    }
                    lbls_list.append(d)

            for t, out in logits.items():
                probs = out.softmax(1)
                confs, preds = probs.max(1)
                confs = confs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                trues = np.array([d.get(t, -1) for d in lbls_list], dtype=np.int64)
                if args.prob_threshold is not None and args.prob_threshold > 0:
                    mask_unk = confs < args.prob_threshold
                    preds = preds.copy()
                    preds[mask_unk] = -1
                all_preds[t].extend(list(preds))
                all_probs[t].extend(list(confs))
                all_trues[t].extend(list(trues))

            for i in range(B):
                if isinstance(loader.dataset, Subset):
                    img_index = loader.dataset.indices[idx_global + i]
                    img_path = loader.dataset.dataset.samples[img_index][0]
                else:
                    try:
                        img_path = loader.dataset.samples[idx_global + i][0]
                    except Exception:
                        img_path = None

                if img_path and os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB")
                    img_np = np.array(im)
                else:
                    x = imgs[i].detach().cpu()
                    x = (x * 0.5 + 0.5).clamp(0, 1)
                    img_np = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                y_true = all_trues[gradcam_task][-B + i]
                names_gc = safe_task_cls_global[gradcam_task]
                lab_name = (
                    "Unknown"
                    if (y_true == -1 or y_true >= len(names_gc))
                    else names_gc[y_true]
                )
                subdir = (
                    Path(args.out_dir) if args.out_dir else (wdir / "sup_predict")
                ) / lab_name
                subdir.mkdir(parents=True, exist_ok=True)

                if args.save_test_images:
                    annotated = img_np.copy()
                    y0, dy = 28, 24
                    for j, t in enumerate(tasks_list):
                        names = safe_task_cls_global[t]
                        p = all_preds[t][-B + i]
                        gt = all_trues[t][-B + i]
                        pr = (
                            "Unknown"
                            if p == -1
                            else (names[p] if p < len(names) else f"id{p}")
                        )
                        tr = (
                            "Unknown"
                            if gt == -1
                            else (names[gt] if gt < len(names) else f"id{gt}")
                        )
                        prob = all_probs[t][-B + i]
                        txt = f"{t} | GT: {tr} | Pred: {pr} | P={prob:.2f}"
                        cv2.putText(
                            annotated,
                            txt,
                            (8, y0 + j * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.imwrite(
                        str(subdir / f"pred_{idx_global + i:06d}.jpg"),
                        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                    )

                if args.visualize_gradcam and args.save_gradcam_images:
                    x1 = imgs[i : i + 1].clone().detach().to(device)
                    cam, tgt_cls = compute_gradcam_supheads(
                        G,
                        Sup,
                        x1,
                        feat_type=args.embed_type,
                        delta_weights=args.delta_weights,
                        task_name=gradcam_task,
                        use_source=args.gradcam_source,
                        level=args.gradcam_level,
                        target_class=(
                            None
                            if all_preds[gradcam_task][-B + i] == -1
                            else int(all_preds[gradcam_task][-B + i])
                        ),
                    )
                    H0, W0 = img_np.shape[:2]
                    if cam.shape[:2] != (H0, W0):
                        cam = cv2.resize(
                            cam.astype(np.float32),
                            (W0, H0),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    cm_code = COLORMAP_DICT.get(
                        args.colormap.lower(), cv2.COLORMAP_HOT
                    )
                    heat = cv2.applyColorMap(np.uint8(255 * cam), cm_code)
                    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                    vis = 0.5 * (img_np.astype(np.float32) / 255.0) + 0.5 * (
                        heat.astype(np.float32) / 255.0
                    )
                    vis = np.clip(vis, 0, 1)
                    vis = (vis * 255).astype(np.uint8)

                    p = all_preds[gradcam_task][-B + i]
                    gt = all_trues[gradcam_task][-B + i]
                    names = safe_task_cls_global[gradcam_task]
                    pr = (
                        "Unknown"
                        if p == -1
                        else (names[p] if p < len(names) else f"id{p}")
                    )
                    tr = (
                        "Unknown"
                        if gt == -1
                        else (names[gt] if gt < len(names) else f"id{gt}")
                    )
                    cv2.putText(
                        vis,
                        f"{gradcam_task} | GT:{tr} | Pred:{pr}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imwrite(
                        str(subdir / f"gradcam_{idx_global + i:06d}.jpg"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                    )

            times.append(time.time() - t0)
            idx_global += B

        metrics = {}
        for t in tasks_list:
            names = safe_task_cls_global[t]
            y_pred = np.array(all_preds[t], dtype=np.int64)
            y_true = np.array(all_trues[t], dtype=np.int64)
            mask = y_pred != -1
            if mask.sum() == 0:
                acc = prec = rec = f1 = 0.0
                cm = np.zeros((len(names), len(names)), dtype=int)
            else:
                acc = float((y_pred[mask] == y_true[mask]).mean())
                if _SK_OK:
                    prec = float(
                        precision_score(
                            y_true[mask],
                            y_pred[mask],
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    rec = float(
                        recall_score(
                            y_true[mask],
                            y_pred[mask],
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    f1 = float(
                        f1_score(
                            y_true[mask],
                            y_pred[mask],
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    cm = confusion_matrix(
                        y_true[mask],
                        y_pred[mask],
                        labels=list(range(len(names))),
                    )
                else:
                    prec = rec = f1 = 0.0
                    cm = np.zeros((len(names), len(names)), dtype=int)

            metrics[t] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
            }
            print(
                f"[{t}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}"
            )

        vals = list(metrics.values())
        avg = {
            "accuracy": float(np.mean([v["accuracy"] for v in vals] or [0.0])),
            "precision": float(np.mean([v["precision"] for v in vals] or [0.0])),
            "recall": float(np.mean([v["recall"] for v in vals] or [0.0])),
            "f1": float(np.mean([v["f1"] for v in vals] or [0.0])),
            "mean_batch_time_sec": float(
                np.mean(times) if len(times) > 0 else 0.0
            ),
        }
        metrics["average"] = avg
        print(
            f"[AVERAGE] acc={avg['accuracy']:.4f}  prec={avg['precision']:.4f}  "
            f"rec={avg['recall']:.4f}  f1={avg['f1']:.4f}"
        )

        metrics_path = (
            Path(args.out_dir) if args.out_dir else (wdir / "sup_predict")
        ) / "sup_predict_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ métriques sauvegardées → {metrics_path}")
        return

    # ========================= MODE: style_transfer =======================
    if args.mode == "style_transfer":
        if not args.style_A or not args.content_B:
            raise RuntimeError(
                "--style_A et --content_B sont requis en mode style_transfer."
            )

        import math

        # ----------------- hyper-params d'inférence -----------------
        train_side_for_style = int(
            getattr(args, "train_side_for_style", 256)
        )  # résol. style (≈ training)
        token_boost = float(
            getattr(args, "token_boost", 1.0)
        )  # >1 → tokens plus forts
        map_shrink = float(getattr(args, "map_shrink", 1.0))

        def _hann2d(h: int, w: int, device, dtype):
            def hann1(n):
                if n <= 1:
                    return torch.ones(n, dtype=dtype, device=device)
                x = torch.arange(n, dtype=dtype, device=device)
                return 0.5 * (1 - torch.cos(2 * math.pi * x / (n - 1)))

            wy = hann1(h).view(h, 1)
            wx = hann1(w).view(1, w)
            w2 = wy * wx
            return (w2 / w2.max().clamp_min(1e-6)).pow(0.7)

        @torch.no_grad()
        def _save_any_range_as_image(x: torch.Tensor, path: str):
            xm, xM = float(x.min().item()), float(x.max().item())
            x_for = x if (xm >= -1.01 and xM <= 1.01) else (x * 2.0 - 1.0)
            _save_tensor_as_image(x_for, path)

        @torch.no_grad()
        def _style_pairs_like_training(Gnet, style_img_1xCHW: torch.Tensor, gain: float):
            if (
                style_img_1xCHW.shape[-1] != train_side_for_style
                or style_img_1xCHW.shape[-2] != train_side_for_style
            ):
                s_small = F.interpolate(
                    style_img_1xCHW,
                    size=(train_side_for_style, train_side_for_style),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                s_small = style_img_1xCHW

            se = Gnet.style_enc(s_small)
            maps, toks, tokG = None, None, None
            if isinstance(se, (list, tuple)) and len(se) == 3:
                maps, toks, tokG = se
            elif isinstance(se, (list, tuple)) and len(se) == 2:
                a, b = se
                if (
                    isinstance(a, (list, tuple))
                    and len(a)
                    and a[0].dim() == 4
                    and hasattr(b, "dim")
                    and b.dim() == 2
                ):
                    maps, tokG = a, b
                elif (
                    isinstance(a, (list, tuple))
                    and len(a)
                    and a[0].dim() == 2
                    and hasattr(b, "dim")
                    and b.dim() == 2
                ):
                    toks, tokG = a, b
            elif (
                isinstance(se, (list, tuple))
                and len(se) == 1
                and hasattr(se[0], "dim")
                and se[0].dim() == 2
            ):
                tokG = se[0]
            elif hasattr(se, "dim") and se.dim() == 2:
                tokG = se

            if tokG is None and toks and len(toks) > 0:
                tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])

            g = float(gain) * float(token_boost)
            toks = tuple((t, g) for t in (toks or []))
            tokG = (tokG, g)
            return {"tokens": toks, "token": tokG}

        @torch.no_grad()
        def _forward_direct(Gnet, content_1xCHW: torch.Tensor, style_dict: dict):
            return Gnet(content_1xCHW, style=style_dict)

        @torch.no_grad()
        def _forward_tiled(
            Gnet,
            content_1xCHW: torch.Tensor,
            style_dict: dict,
            tile: int = 256,
            overlap: int = 64,
        ):
            B, C, H, W = content_1xCHW.shape
            assert B == 1
            device_t, dtype = content_1xCHW.device, content_1xCHW.dtype
            out = torch.zeros((1, C, H, W), device=device_t, dtype=dtype)
            wsum = torch.zeros((1, 1, H, W), device=device_t, dtype=dtype)
            stride = max(1, tile - overlap)

            for y0 in range(0, H, stride):
                for x0 in range(0, W, stride):
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)
                    ph, pw = y1 - y0, x1 - x0
                    patch = content_1xCHW[..., y0:y1, x0:x1]
                    pad_h, pad_w = tile - ph, tile - pw
                    if pad_h > 0 or pad_w > 0:
                        patch = F.pad(
                            patch, (0, pad_w, 0, pad_h), mode="reflect"
                        )
                    p = _forward_direct(Gnet, patch, style_dict)
                    p = p[..., :ph, :pw]
                    ww = _hann2d(ph, pw, device_t, dtype).view(1, 1, ph, pw)
                    out[..., y0:y1, x0:x1] += p * ww
                    wsum[..., y0:y1, x0:x1] += ww
            return out / wsum.clamp_min(1e-6)

        @torch.no_grad()
        def _forward_at_train_res(
            Gnet, content_1xCHW: torch.Tensor, style_dict: dict, train_side: int = 256
        ):
            B, C, H, W = content_1xCHW.shape
            x_low = F.interpolate(
                content_1xCHW,
                size=(train_side, train_side),
                mode="bilinear",
                align_corners=False,
            )
            y_low = _forward_direct(Gnet, x_low, style_dict)
            return F.interpolate(
                y_low, size=(H, W), mode="bilinear", align_corners=False
            )

        def _patch_spade_balance(
            Gnet, map_mult: float, token_gain_mult_already_applied: float
        ):
            import types

            for m in Gnet.modules():
                if m.__class__.__name__ == "SPADEResBlock":
                    orig = m.forward

                    def new_forward(
                        self,
                        x,
                        style_map,
                        style_token,
                        _orig=orig,
                        _mm=float(map_mult),
                        _tb=float(token_boost),
                    ):
                        sm = style_map * _mm
                        st = style_token
                        if isinstance(st, (tuple, list)) and len(st) == 2:
                            tok, g = st
                            st = (tok, float(g))
                        return _orig(x, sm, st)

                    m.forward = types.MethodType(new_forward, m)

        A_paths = _list_images_sorted(args.style_A)
        B_paths = _list_images_sorted(args.content_B)
        if len(A_paths) == 0 or len(B_paths) == 0:
            raise RuntimeError(
                "Aucune image trouvée dans --style_A ou --content_B."
            )

        if len(A_paths) == 1:
            pairs = [(A_paths[0], b) for b in B_paths]
        else:
            if len(A_paths) != len(B_paths):
                raise RuntimeError(
                    f"Lorsque |A|>1, |A| doit = |B|. Reçus: |A|={len(A_paths)} |B|={len(B_paths)}"
                )
            pairs = list(zip(A_paths, B_paths))

        side = int(args.resize) if args.resize else _infer_img_size_from_cfg(cfg, 256)
        tfm = _make_transform(side)

        via = args.transfer_via.upper()
        out_root = Path(args.save_dir) if args.save_dir else (wdir / f"style_transfer_via_{via}")
        out_root.mkdir(parents=True, exist_ok=True)

        for net in (G_A, G_B):
            try:
                net.eval()
            except Exception:
                pass

        _patch_spade_balance(G_A, map_shrink, token_boost)
        if via != "GA":
            _patch_spade_balance(G_B, map_shrink, token_boost)

        print(
            f"→ Style transfer via {via} | pairs={len(pairs)} | resize={side} | out={str(out_root)}"
        )
        print(
            f"[INF] train_side_for_style={train_side_for_style}  "
            f"token_boost={token_boost}  map_shrink={map_shrink}"
        )

        first_debug_saved = False
        with torch.no_grad():
            for a_path, b_path in pairs:
                A_img = _load_img_tensor(a_path, tfm, device)  # [-1,1]
                B_img = _load_img_tensor(b_path, tfm, device)  # [-1,1]

                styleGA = _style_pairs_like_training(
                    G_A, A_img, gain=float(args.style_gain_A)
                )
                styleGB = (
                    _style_pairs_like_training(
                        G_B, A_img, gain=float(args.style_gain_B)
                    )
                    if via != "GA"
                    else None
                )

                cand = []
                for tag, fun in (
                    ("direct", _forward_direct),
                    ("tiled", _forward_tiled),
                    ("train256", _forward_at_train_res),
                ):
                    try:
                        outB = fun(
                            G_A, B_img, _style_pairs_like_training(G_A, B_img, 1.0)
                        )
                        mse = torch.mean((outB - B_img).float() ** 2).item()
                        std = float(outB.std().item())
                        cand.append((mse, -std, tag))
                    except Exception:
                        pass
                cand.sort(key=lambda z: (z[0], z[1]))
                exec_GA = cand[0][2] if cand else "train256"
                print(f"[AUTO GA] exec={exec_GA}")

                def _apply_exec(Gnet, X, style, exec_kind):
                    if exec_kind == "direct":
                        return _forward_direct(Gnet, X, style)
                    if exec_kind == "tiled":
                        return _forward_tiled(Gnet, X, style, tile=256, overlap=64)
                    if exec_kind == "train256":
                        return _forward_at_train_res(
                            Gnet, X, style, train_side=train_side_for_style
                        )
                    raise ValueError(exec_kind)

                if via == "GA":
                    out = _apply_exec(G_A, B_img, styleGA, exec_GA)
                    print(
                        f"[RUN GA] mean={float(out.mean()):.4f}  "
                        f"std={float(out.std()):.4f}"
                    )
                else:
                    far = _apply_exec(G_A, B_img, styleGA, exec_GA)
                    if args.gb_spectral_noise:
                        far = spectral_noise_like(
                            far,
                            sigma=float(args.gb_noise_sigma),
                            gamma=float(args.gb_noise_gamma),
                        )
                    out = _apply_exec(G_B, far, styleGB, exec_GA)
                    print(
                        f"[RUN GB] mean={float(out.mean()):.4f}  "
                        f"std={float(out.std()):.4f}"
                    )

                a_name = Path(a_path).stem
                b_name = Path(b_path).stem
                out_path = out_root / f"{a_name}__TO__{b_name}.png"
                _save_any_range_as_image(out, str(out_path))
                print(f"✓ {a_name} → {b_name}  → {out_path}")

                if not first_debug_saved:
                    first_debug_saved = True
                    try:
                        import torchvision.utils as vutils
                        from torchvision.transforms.functional import to_pil_image

                        def to01(x):
                            xm, xM = float(x.min().item()), float(x.max().item())
                            return (
                                x.clamp(0, 1)
                                if (xm >= -0.01 and xM <= 1.01)
                                else (x.clamp(-1, 1) * 0.5 + 0.5)
                            )

                        grid = torch.cat(
                            [to01(A_img), to01(B_img), to01(out)], 0
                        )
                        grid = vutils.make_grid(grid, nrow=3)
                        to_pil_image(grid.cpu()).save(
                            str(out_root / "_DEBUG_A_B_OUT.png")
                        )
                        print(
                            f"• Debug grid sauvé → {out_root / '_DEBUG_A_B_OUT.png'}"
                        )
                    except Exception as e:
                        print(f"[WARN] debug grid non sauvé: {e}")

        print("✓ Terminé (style_transfer).")
        return

    # -------------- tsne / metrics / cls_tokens / style -------------------
    # Si feature_mode == "cls_tokens", on passe forcément par per_task/composite
    if args.per_task:
        if Sup is None:
            raise RuntimeError("--per_task requiert SupHeads (sup_ckpt).")
        embs_d, lbls_d, paths_d = compute_embeddings_with_paths(
            composite, loader, device, per_task=True
        )

        def _norm_key(s: str) -> str:
            return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")

        ds_for_names = (
            loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        )
        ds_task_cls = getattr(ds_for_names, "task_classes", {}) or {}
        nk = {_norm_key(k): k for k in ds_task_cls.keys()}

        safe_task_cls = {}
        for t in composite.tasks:
            ds_key = nk.get(_norm_key(t), None)
            if ds_key is None and "__DEFAULT__" in ds_task_cls:
                ds_key = "__DEFAULT__"
            names = list(ds_task_cls.get(ds_key, [])) if ds_key is not None else []
            if not names:
                names = (
                    [
                        f"class {i}"
                        for i in range(int(lbls_d[t].max() + 1))
                    ]
                    if t in lbls_d and lbls_d[t].size
                    else ["class 0"]
                )
            if (t in lbls_d) and (lbls_d[t].size > 0) and np.any(lbls_d[t] < 0):
                names = names + ["Unknown"]
                unk_id = len(names) - 1
                lbls_d[t] = np.where(lbls_d[t] < 0, unk_id, lbls_d[t]).astype(
                    np.int64
                )
            safe_task_cls[t] = names

        if args.mode == "tsne_interactive":
            if args.metrics.strip():
                flat_embs = (
                    np.concatenate(list(embs_d.values()))
                    if embs_d
                    else np.zeros((0, 1))
                )
                flat_labels = (
                    np.concatenate(list(lbls_d.values()))
                    if lbls_d
                    else np.zeros((0,), np.int64)
                )
                _, scores, _ = compute_metrics(
                    flat_embs,
                    flat_labels,
                    metrics=[
                        m.strip()
                        for m in args.metrics.split(",")
                        if m.strip()
                    ],
                    pca_dim=None,
                    l2_norm=False,
                )
            else:
                scores = None

            plot_tsne_interactive(
                embs_d,
                lbls_d,
                safe_task_cls,
                paths_d,
                save_dir=str(wdir),
                metric_scores=scores,
            )
            return

        metrics_req = [
            m.strip() for m in args.metrics.split(",") if m.strip()
        ]
        if not metrics_req:
            raise RuntimeError(
                "--metrics est requis en mode 'passe_by_metrics'."
            )

        order = list(embs_d.keys())
        embs_flat = np.concatenate([embs_d[t] for t in order], 0)
        labels_flat = np.concatenate([lbls_d[t] for t in order], 0)
        paths_flat = np.concatenate(
            [np.asarray(paths_d[t]) for t in order], 0
        )

        embs_proc, scores, groupings = compute_metrics(
            embs_flat,
            labels_flat,
            metrics=metrics_req,
            pca_dim=args.pca_dim,
            l2_norm=args.l2_norm,
        )

        embs_plot, lbls_plot, paths_plot, cls_plot = {}, {}, {}, {}
        start = 0
        for t in order:
            n = len(lbls_d[t])
            embs_plot[t] = embs_proc[start : start + n]
            lbls_plot[t] = labels_flat[start : start + n]
            paths_plot[t] = paths_flat[start : start + n]
            cls_plot[t] = safe_task_cls[t]
            start += n

        has_extra_view = False
        if "cluster_id" in groupings:
            has_extra_view = True
            cid = groupings["cluster_id"].astype(int)
            embs_plot["cluster_id"] = embs_proc
            lbls_plot["cluster_id"] = cid
            paths_plot["cluster_id"] = paths_flat
            cls_plot["cluster_id"] = [f"cluster {i}" for i in np.unique(cid)]

        if "knn_pred" in groupings:
            has_extra_view = True
            kp = groupings["knn_pred"].astype(int)
            embs_plot["knn_pred"] = embs_proc
            lbls_plot["knn_pred"] = kp
            paths_plot["knn_pred"] = paths_flat
            cls_plot["knn_pred"] = [f"pred {i}" for i in np.unique(kp)]

        if has_extra_view:
            plot_tsne_cluster_knn(
                embs_plot,
                lbls_plot,
                cls_plot,
                paths_plot,
                cluster_id=groupings.get("cluster_id"),
                knn_pred=groupings.get("knn_pred"),
                metric_scores=scores,
                save_dir=str(wdir),
            )
        else:
            plot_tsne_interactive(
                embs_plot,
                lbls_plot,
                cls_plot,
                paths_plot,
                metric_scores=scores,
                save_dir=str(wdir),
            )
        return

    # =====================================================  global (style | sem)  ===================================
    if args.feature_mode == "sem_resnet50":
        embs_d, lbls_d, class_maps, paths_d = compute_sem_embeddings(
            sem_backbone,
            loader,
            device,
            imagenet_norm=bool(int(args.sem_imagenet_norm)),
            pca_dim=args.pca_dim if args.mode == "tsne_interactive" else None,
            l2_norm=args.l2_norm if args.mode == "tsne_interactive" else False,
        )
    else:
        embs_d, lbls_d, class_maps, paths_d = compute_style_embeddings(
            G,
            loader,
            device,
            embed_type=args.embed_type,
            token_pool=args.token_pool,
            layers=args.layers,
            pca_dim=args.pca_dim if args.mode == "tsne_interactive" else None,
            l2_norm=args.l2_norm if args.mode == "tsne_interactive" else False,
            delta_weights=args.delta_weights,
        )

    if args.mode == "tsne_interactive":
        if args.metrics.strip():
            flat_embs = (
                np.concatenate(list(embs_d.values()))
                if embs_d
                else np.zeros((0, 1))
            )
            flat_labels = (
                np.concatenate(list(lbls_d.values()))
                if lbls_d
                else np.zeros((0,), np.int64)
            )
            _, scores, _ = compute_metrics(
                flat_embs,
                flat_labels,
                metrics=[
                    m.strip()
                    for m in args.metrics.split(",")
                    if m.strip()
                ],
                pca_dim=None,
                l2_norm=False,
            )
        else:
            scores = None

        plot_tsne_interactive(
            embs_d,
            lbls_d,
            class_maps,
            paths_d,
            save_dir=str(wdir),
            metric_scores=scores,
        )
        return

    metrics_req = [
        m.strip() for m in args.metrics.split(",") if m.strip()
    ]
    if not metrics_req:
        raise RuntimeError(
            "--metrics est requis en mode 'passe_by_metrics'."
        )

    embs_cat, labels_cat, paths_cat = [], [], []
    lengths = {}
    for task_name in embs_d:
        embs_cat.append(embs_d[task_name])
        labels_cat.append(lbls_d[task_name])
        paths_cat.append(np.asarray(paths_d[task_name]))
        lengths[task_name] = len(lbls_d[task_name])
    embs_flat = np.concatenate(embs_cat, 0)
    labels_flat = np.concatenate(labels_cat, 0)
    paths_flat = np.concatenate(paths_cat, 0)

    embs_proc, scores, groupings = compute_metrics(
        embs_flat,
        labels_flat,
        metrics=metrics_req,
        pca_dim=args.pca_dim,
        l2_norm=args.l2_norm,
    )

    embs_plot, lbls_plot, paths_plot, cls_plot = {}, {}, {}, {}
    start = 0
    for task_name, n in lengths.items():
        end = start + n
        embs_plot[task_name] = embs_proc[start:end]
        lbls_plot[task_name] = labels_flat[start:end]
        paths_plot[task_name] = paths_flat[start:end]
        cls_plot[task_name] = class_maps[task_name]
        start = end

    has_extra_view = False
    if "cluster_id" in groupings:
        has_extra_view = True
        cid = groupings["cluster_id"].astype(int)
        embs_plot["cluster_id"] = embs_proc
        lbls_plot["cluster_id"] = cid
        paths_plot["cluster_id"] = paths_flat
        cls_plot["cluster_id"] = [f"cluster {i}" for i in np.unique(cid)]

    if "knn_pred" in groupings:
        has_extra_view = True
        kp = groupings["knn_pred"].astype(int)
        embs_plot["knn_pred"] = embs_proc
        lbls_plot["knn_pred"] = kp
        paths_plot["knn_pred"] = paths_flat
        cls_plot["knn_pred"] = [f"pred {i}" for i in np.unique(kp)]

    if has_extra_view:
        plot_tsne_cluster_knn(
            embs_plot,
            lbls_plot,
            cls_plot,
            paths_plot,
            cluster_id=groupings.get("cluster_id"),
            knn_pred=groupings.get("knn_pred"),
            metric_scores=scores,
            save_dir=str(wdir),
        )
    else:
        plot_tsne_interactive(
            embs_plot,
            lbls_plot,
            cls_plot,
            paths_plot,
            metric_scores=scores,
            save_dir=str(wdir),
        )


if __name__ == "__main__":
    main()