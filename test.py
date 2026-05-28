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
from PIL import Image, ImageTk
import tkinter as tk


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.fusion_head import VectorGatedFusionHead



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






def _normalize_task_classes_for_display(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        names = []
        if isinstance(v, list):
            names = [str(x) for x in v]
        elif isinstance(v, dict):
            if isinstance(v.get("classes"), list):
                names = [str(x) for x in v["classes"]]
            elif isinstance(v.get("names"), list):
                names = [str(x) for x in v["names"]]
            elif isinstance(v.get("num_classes"), int):
                names = [f"class {i}" for i in range(int(v["num_classes"]))]
        elif isinstance(v, int):
            names = [f"class {i}" for i in range(int(v))]
        if names:
            out[str(k)] = names
    return out


def _infer_imagenet_classes_from_ann_dir(ann_dir: str):
    import xml.etree.ElementTree as ET
    ap = Path(str(ann_dir))
    if ap.name.lower() == 'val':
        val_dir = ap
    else:
        val_dir = ap / 'val'
        if not val_dir.exists():
            val_dir = ap
    synsets = set()
    if val_dir.exists():
        for xml_path in sorted(val_dir.glob('*.xml')):
            try:
                root = ET.parse(str(xml_path)).getroot()
                for obj in root.findall('object'):
                    syn = (obj.findtext('name') or '').strip()
                    if syn.startswith('n'):
                        synsets.add(syn)
            except Exception:
                continue
    return sorted(synsets)


def _load_task_classes_for_camera(args):
    task_classes = {}
    if getattr(args, 'classes_json', None):
        try:
            with open(args.classes_json, 'r', encoding='utf-8') as f:
                task_classes = _normalize_task_classes_for_display(json.load(f))
        except Exception as e:
            print(f"[backbone_camera] Impossible de lire classes_json: {e}")
    if task_classes:
        return task_classes

    synsets = []
    syn_to_name = {}
    syn_map = getattr(args, 'imagenet_synset_mapping', None)
    if syn_map and Path(syn_map).exists():
        try:
            for ln in Path(syn_map).read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                syn = parts[0]
                if syn.startswith('n'):
                    synsets.append(syn)
                    syn_to_name[syn] = ' '.join(parts[1:]).strip() if len(parts) > 1 else syn
        except Exception as e:
            print(f"[backbone_camera] Lecture synset_mapping échouée: {e}")
    if not synsets and getattr(args, 'imagenet_ann_dir', None):
        synsets = _infer_imagenet_classes_from_ann_dir(args.imagenet_ann_dir)
    if synsets:
        display_names = []
        for syn in synsets:
            nm = syn_to_name.get(syn, '').strip()
            display_names.append(f"{syn} — {nm}" if nm and nm != syn else syn)
        return {'__DEFAULT__': display_names}
    return {}


def _task_display_names(task_classes: dict, task: str, fallback_n: int | None = None):
    names = []
    if isinstance(task_classes, dict):
        if task in task_classes:
            names = list(task_classes[task])
        elif '__DEFAULT__' in task_classes:
            names = list(task_classes['__DEFAULT__'])
    if not names and fallback_n is not None and fallback_n > 0:
        names = [f"class {i}" for i in range(int(fallback_n))]
    return names


class _TkCameraDisplay:
    def __init__(self, title: str = "RT-STORM backbone camera"):
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._closed = False
        self._last_key = None
        self._photo = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)
        self.root.focus_force()

    def _on_close(self):
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_key(self, event):
        self._last_key = getattr(event, "keysym", None)

    @property
    def closed(self) -> bool:
        return self._closed

    def show_bgr(self, frame_bgr):
        if self._closed:
            return False
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(image=im)
        self.label.configure(image=self._photo)
        self.label.image = self._photo
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True
            return False
        if self._last_key in ("q", "Escape"):
            self._closed = True
            return False
        return True

    def close(self):
        if not self._closed:
            self._on_close()


def run_backbone_camera(
    model: nn.Module,
    device: torch.device,
    *,
    cfg: dict,
    feature_mode: str = 'style',
    task_classes: dict | None = None,
    cam_index: int = 0,
    topk: int = 3,
):
    model.eval()
    img_side = _infer_img_size_from_cfg(cfg, default=256)
    tfm = _make_transform(img_side)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index={cam_index}")

    task_classes = task_classes or {}
    last_t = time.time()
    display_backend = 'cv2'
    warned_no_gui = False
    tk_display = None
    print(
        f"[backbone_camera] Caméra {cam_index} ouverte | feature_mode={feature_mode} | "
        f"resize={img_side} | topk={topk}. Appuyez sur 'q' pour quitter."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tfm(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_out = model(x)

            if torch.is_tensor(raw_out):
                logits_d = {'__DEFAULT__': raw_out}
            elif isinstance(raw_out, dict):
                logits_d = raw_out
            else:
                raise RuntimeError(f"run_backbone_camera: sortie non supportée ({type(raw_out)})")

            lines = [f"mode={feature_mode}"]
            multi_task_classes = isinstance(task_classes, dict) and any(k != '__DEFAULT__' for k in task_classes.keys())
            for task, logits in logits_d.items():
                if logits is None or (not torch.is_tensor(logits)) or logits.numel() == 0:
                    continue
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = torch.softmax(logits[0], dim=0)
                names = _task_display_names(task_classes, task, fallback_n=int(probs.numel()))

                if multi_task_classes and len(logits_d) > 1:
                    best_idx = int(torch.argmax(probs).item())
                    best_prob = float(probs[best_idx].item())
                    label = names[best_idx] if 0 <= best_idx < len(names) else f"class {best_idx}"
                    lines.append(f"{task}: {label} ({100.0 * best_prob:.1f}%)")
                else:
                    k = max(1, min(int(topk), int(probs.numel())))
                    vals, idxs = torch.topk(probs, k=k)
                    if len(logits_d) > 1:
                        lines.append(f"[{task}]")
                    for rank, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
                        label = names[idx] if 0 <= idx < len(names) else f"class {idx}"
                        lines.append(f"{rank}. {label} ({100.0 * float(v):.1f}%)")

            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            lines.append(f"FPS: {fps:.1f}")

            y = 28
            for line in lines[:12]:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                y += 24

            if display_backend == 'cv2':
                try:
                    cv2.imshow('RT-STORM backbone camera', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                except cv2.error:
                    try:
                        tk_display = _TkCameraDisplay('RT-STORM backbone camera')
                        display_backend = 'tk'
                        print('[backbone_camera] OpenCV GUI indisponible. Bascule automatique vers une fenêtre Tkinter.')
                        if not tk_display.show_bgr(frame):
                            break
                    except Exception as e:
                        display_backend = 'console'
                        if not warned_no_gui:
                            warned_no_gui = True
                            print(
                                '[backbone_camera] Aucun backend d\'affichage disponible ' 
                                f'(OpenCV headless / Tk indisponible: {e}). ' 
                                'Affichage désactivé ; les prédictions continuent côté console. ' 
                                'Interrompez avec Ctrl+C.'
                            )
            elif display_backend == 'tk':
                if tk_display is None:
                    try:
                        tk_display = _TkCameraDisplay('RT-STORM backbone camera')
                    except Exception as e:
                        display_backend = 'console'
                        if not warned_no_gui:
                            warned_no_gui = True
                            print(
                                '[backbone_camera] Impossible d\'initialiser Tkinter ' 
                                f'({e}). Affichage désactivé ; les prédictions continuent côté console. ' 
                                'Interrompez avec Ctrl+C.'
                            )
                if display_backend == 'tk' and tk_display is not None:
                    if not tk_display.show_bgr(frame):
                        break
            else:
                print("\r" + " | ".join(lines[:4])[:220], end="", flush=True)
    finally:
        cap.release()
        if display_backend == 'cv2':
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if tk_display is not None:
            try:
                tk_display.close()
            except Exception:
                pass
        if display_backend == 'console' and warned_no_gui:
            print()


def main() -> None:
    # ------------------------ ARGUMENTS ------------------------
    ap = argparse.ArgumentParser(
        "Exploration : t-SNE / métriques / sup_predict / inference / style_transfer / detect_transformer"
    )

    # ckpts / modèles -------------------------------------------------------
    ap.add_argument("--cfg",         required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--sup_ckpt")
    ap.add_argument("--sup_in_dim", type=int)
    ap.add_argument(
        "--token_dim",
        type=int,
        default=None,
        help="Dimension des tokens style/contenu du générateur. Si fournie, surcharge cfg['model']['token_dim'] pour le test.",
    )
    # ckpts séparés pour GA/GB (style GAN)
    ap.add_argument("--ckpt_GA")
    ap.add_argument("--ckpt_GB")

    # données --------------------------------------------------------------
    ap.add_argument("--data_json")
    ap.add_argument("--classes_json")
    ap.add_argument("--search_folder")
    ap.add_argument("--find_images_by_sub_folder", action="store_true")
    ap.add_argument("--data", help="Dossier racine au format ImageFolder (sous-dossiers = classes)")

    # ImageNet ILSVRC CLS-LOC (optionnel) -------------------------------
    ap.add_argument("--imagenet_split", default=None,
                    help="ImageNet CLS-LOC split: auto|train|val|test. Si défini, --data peut pointer vers train/ ou val/.")
    ap.add_argument("--imagenet_ann_dir", default=None,
                    help="Chemin vers ILSVRC/Annotations/CLS-LOC (pour split=val/test).")
    ap.add_argument("--imagenet_imagesets_dir", default=None,
                    help="Chemin vers ILSVRC/ImageSets/CLS-LOC (pour lister val/train/test).")
    ap.add_argument("--imagenet_synset_mapping", default=None,
                    help="Chemin vers LOC_synset_mapping.txt.")
    ap.add_argument("--imagenet_val_solution_csv", default=None,
                    help="Chemin vers LOC_val_solution.csv (fallback si XML indisponibles).")
    ap.add_argument("--imagenet_label_base", type=int, default=1,
                    help="Base des labels val_solution (souvent 1).")
    ap.add_argument("--imagenet_num_classes", type=int, default=1000,
                    help="Nombre de classes ImageNet.")
    ap.add_argument("--imagenet_return_bbox", type=int, default=0,
                    help="1=renvoyer aussi bbox (pas nécessaire pour sup_predict/top1).")

    # modes principaux ------------------------------------------------------
    ap.add_argument(
        "--mode",
        choices=["tsne_interactive", "passe_by_metrics", "sup_predict", "inference", "style_transfer",
                 "detect_transformer", "backbone_camera"],
        default="tsne_interactive",
    )

    # source des features pour tsne / metrics / sup_predict
    ap.add_argument(
        "--feature_mode",
        choices=["style", "cls_tokens", "sem_resnet50", "fusion"],
        default="style",
        help="Source des embeddings pour tsne/metrics/sup_predict : "
             "'style' (GAN), 'cls_tokens' (SupHeads/per_task), 'sem_resnet50' (backbone sémantique) ou 'fusion' (style+sémantique via FusionHead).",
    )


    # Source utilisée pour charger/entraîner SupHeads (utile pour auto-load du backbone sémantique)
    ap.add_argument(
        "--sup_feat_source",
        choices=["generator", "sem_resnet50", "fusion"],
        default="generator",
        help="Source des features attendues par SupHeads. Si sem_resnet50 ou fusion, on peut auto-charger le backbone sémantique depuis weights_dir (SemBackbone_epoch*.pt).",
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
            "tokL", "tokL_mean", "tokL_w",
            "style_tok",
            "mapG", "mapL", "mapL_mean", "mapL_w",
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
            "Pour 'tok6_w' : 6 poids 'wG,w5,w4,w3,w2,w1' (compat). Pour 'tokL_w' : (L+1) poids 'wG,wL,...,w1'. "
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
    ap.add_argument(
        "--dump_param_count_json",
        action="store_true",
        help="Si activé, écrit un JSON avec le nombre de paramètres par partie (style / contenu sémantique) et les totaux.",
    )
    ap.add_argument(
        "--param_count_json_name",
        type=str,
        default="model_parameter_counts.json",
        help="Nom du fichier JSON de comptage des paramètres.",
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
    ap.add_argument(
        "--camera_topk",
        type=int,
        default=3,
        help="Nombre de prédictions à afficher par tâche en mode backbone_camera.",
    )
    ap.add_argument(
        "--inference_json_name",
        default="inference_predictions.json",
        help="Nom du fichier JSON produit en mode inference.",
    )
    ap.add_argument(
        "--inference_save_csv",
        action="store_true",
        help="En mode inference, écrire aussi un CSV plat des prédictions.",
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
    # Robust device selection (CPU-only PyTorch builds cannot move tensors to CUDA)
    device_str = str(args.device).lower().strip()
    if device_str.startswith("cuda") and (not torch.cuda.is_available()):
        print(f"[WARN] --device={args.device} requested but CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    cfg = json.load(open(args.cfg))
    if args.token_dim is not None:
        cfg.setdefault("model", {})
        cfg["model"]["token_dim"] = int(args.token_dim)
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

        _maybe_write_param_count_report(out_dir)
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

    def _find_latest_named_ckpt(weights_dir: Path, stem: str, extra_roots: list[Path] | None = None) -> Path | None:
        def _safe_resolve(x: Path) -> Path:
            try:
                return x.resolve()
            except Exception:
                return x

        def _score_ckpt_path(p: Path) -> tuple:
            name = p.name.lower()
            # priorité: best > last > epoch > autre ; .pth/.pt > safetensors ; puis mtime
            kind = 0
            if "best" in name:
                kind = 3
            elif "last" in name:
                kind = 2
            elif "epoch" in name:
                kind = 1
            ext = p.suffix.lower()
            ext_score = 2 if ext in (".pth", ".pt") else 1
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            return (kind, ext_score, mtime)

        def _collect(root: Path):
            try:
                root = Path(root)
            except Exception:
                return []
            if not str(root) or not root.exists():
                return []

            # Motifs volontairement larges et robustes:
            # - .pth, .pt, .safetensors
            # - best / last / epoch / n'importe quel suffixe
            patterns = [
                f"{stem}*.pth", f"{stem}*.pt", f"{stem}*.safetensors",
                f"{stem.lower()}*.pth", f"{stem.lower()}*.pt", f"{stem.lower()}*.safetensors",
            ]

            pats = []
            seen = set()
            # D'abord dans la racine
            for pat in patterns:
                for cand in root.glob(pat):
                    key = str(_safe_resolve(cand))
                    if key not in seen:
                        seen.add(key)
                        pats.append(cand)
            # Puis récursivement
            for pat in patterns:
                for cand in root.rglob(pat):
                    key = str(_safe_resolve(cand))
                    if key not in seen:
                        seen.add(key)
                        pats.append(cand)

            # Fallback ultime: recherche par nom contenant stem, quelle que soit l'extension
            if not pats:
                stem_low = stem.lower()
                for cand in root.rglob("*"):
                    try:
                        if cand.is_file() and stem_low in cand.name.lower() and cand.suffix.lower() in (".pth", ".pt", ".safetensors"):
                            key = str(_safe_resolve(cand))
                            if key not in seen:
                                seen.add(key)
                                pats.append(cand)
                    except Exception:
                        continue
            return pats

        roots = []
        for r in [weights_dir, *(extra_roots or [])]:
            try:
                rp = Path(r)
            except Exception:
                continue
            if not str(rp):
                continue
            rr = _safe_resolve(rp)
            if rr not in roots:
                roots.append(rr)
            # ajoute aussi le parent si le chemin pointe vers un fichier ou un sous-dossier trop spécifique
            par = rr.parent
            if str(par) and par not in roots:
                roots.append(par)

        pats = []
        for root in roots:
            pats.extend(_collect(root))
        if not pats:
            searched = ", ".join(str(r) for r in roots)
            print(f"[WARN] Aucun checkpoint '{stem}' trouvé. Dossiers inspectés: {searched}")
            return None

        uniq = {str(_safe_resolve(p)): p for p in pats}
        pats = list(uniq.values())
        pats.sort(key=_score_ckpt_path)
        chosen = pats[-1]
        print(f"[INFO] {stem} candidates found ({len(pats)}). Selected: {chosen}")
        return chosen

    # ----------- Backbone sémantique (optionnel, pour tests) -----------
    sem_backbone = None
    sem_out_ch = None
    if (args.feature_mode == "sem_resnet50") or (str(args.sup_feat_source).lower() == "fusion"):
        sem_backbone, sem_out_ch = build_sem_backbone_for_eval(
            device=device,
            arch=args.det_sem_backbone,
            return_layer=args.det_sem_return_layer,
            pretrained=bool(int(args.sem_pretrained)),
            pretrained_path=str(args.sem_pretrained_path or ""),
            strict=bool(int(args.sem_pretrained_strict)),
            verbose=bool(int(args.sem_pretrained_verbose)),
            weights_dir=(wdir if args.sup_feat_source in ("sem_resnet50", "fusion") else None),
            sem_filename="SemBackbone",
        )
        print(f"✓ sem_backbone prêt | arch={args.det_sem_backbone} | layer={args.det_sem_return_layer} | out_ch={sem_out_ch}")

    # ----------- Charger G/Sup selon le mode -----------
    G, Sup, task_cls = None, None, None

    sem_only_modes = {"tsne_interactive", "passe_by_metrics", "sup_predict", "inference", "backbone_camera"}
    if (args.feature_mode == "sem_resnet50") and (str(args.sup_feat_source).lower() != "fusion") and (args.mode in sem_only_modes):
        # Pas besoin de G_A/G_B.
        # Charger SupHeads si requis (sup_predict) ou si demandé (tsne_use_supheads / per_task)
        need_sup = (args.mode in {"sup_predict", "inference", "backbone_camera"}) or bool(args.tsne_use_supheads) or bool(args.per_task)
        if need_sup:
            if not args.sup_ckpt:
                raise RuntimeError("SupHeads requis mais --sup_ckpt manquant (utilise sup_predict/backbone_camera/tsne_use_supheads/per_task).")
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
        # En mode fusion, on charge d'abord le générateur seul puis on reconstruit SupHeads
        # explicitement à partir du checkpoint pour respecter la vraie input_dim.
        fusion_mode = (str(args.sup_feat_source).lower() == "fusion")
        G, Sup, task_cls = load_models(
            weights_dir=wdir,
            device=device,
            cfg=cfg,
            ckpt_gen=args.ckpt,
            sup_ckpt=(None if fusion_mode else args.sup_ckpt),
            classes_json=args.classes_json,
            sup_in_dim=args.sup_in_dim,
            ckpt_GA=args.ckpt_GA,
            ckpt_GB=args.ckpt_GB,
        )

    # Récupération GA/GB uniquement si un générateur est chargé
    if G is not None:
        G_A = getattr(G, "GA", getattr(G, "G_A", G))
        G_B = getattr(G, "GB", getattr(G, "G_B", G))
    fusion_head = None
    if str(args.sup_feat_source).lower() == "fusion":
        if G is None:
            raise RuntimeError("sup_feat_source=fusion requiert un générateur chargé.")
        if sem_backbone is None:
            raise RuntimeError("sup_feat_source=fusion requiert aussi un backbone sémantique chargé.")
        fusion_ckpt = _find_latest_named_ckpt(
            wdir,
            "FusionHead",
            extra_roots=[
                Path(getattr(args, "out_dir", "") or ""),
                (Path(getattr(args, "sup_ckpt", "")).parent if getattr(args, "sup_ckpt", None) else wdir),
            ],
        )
        if fusion_ckpt is None:
            raise FileNotFoundError(f"Aucun checkpoint FusionHead trouvé dans {wdir}")
        fusion_obj = _read_state_any(fusion_ckpt, device)
        fusion_sd = fusion_obj.get("state_dict", fusion_obj) if isinstance(fusion_obj, dict) else fusion_obj
        style_in_dim = None
        try:
            if hasattr(G, "sup_in_dim_for"):
                style_in_dim = int(G.sup_in_dim_for(args.embed_type))
        except Exception:
            style_in_dim = None
        if style_in_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, _infer_img_size_from_cfg(cfg, default=256), _infer_img_size_from_cfg(cfg, default=256), device=device)
                if hasattr(G, "sup_features"):
                    style_in_dim = int(G.sup_features(dummy, args.embed_type, delta_weights=args.delta_weights).shape[1])
                else:
                    z, _ = G.encode_content(dummy)
                    style_in_dim = int(F.adaptive_avg_pool2d(z, 1).flatten(1).shape[1])
        sem_in_dim = int((fusion_obj.get("sem_in_dim") if isinstance(fusion_obj, dict) else None) or sem_out_ch or 2048)
        fusion_dim = int((fusion_obj.get("fusion_dim") if isinstance(fusion_obj, dict) else None) or args.sup_in_dim or 1024)
        fusion_head = VectorGatedFusionHead(style_in_dim=style_in_dim, sem_in_dim=sem_in_dim, fusion_dim=fusion_dim, dropout=float(getattr(args, "fusion_dropout", 0.1))).to(device)
        miss, unexp = fusion_head.load_state_dict(fusion_sd, strict=False)
        if miss:
            print(f"[WARN] FusionHead missing keys ({len(miss)})")
        if unexp:
            print(f"[WARN] FusionHead unexpected keys ({len(unexp)})")
        fusion_head.eval()
        print(f"✓ FusionHead loaded – {fusion_ckpt.name} | style_in_dim={style_in_dim} | sem_in_dim={sem_in_dim} | fusion_dim={fusion_dim}")

        # Recharger SupHeads explicitement avec la vraie dimension du checkpoint.
        if not args.sup_ckpt:
            raise RuntimeError("sup_feat_source=fusion requiert --sup_ckpt pour charger SupHeads.")
        from models.sup_heads import SupHeads
        sup_path = Path(args.sup_ckpt)
        if not sup_path.is_absolute():
            cand = wdir / sup_path
            if cand.exists():
                sup_path = cand
        if not sup_path.exists():
            raise FileNotFoundError(f"SupHeads introuvable: {sup_path}")
        sup_obj = _read_state_any(sup_path, device)
        sup_sd = sup_obj.get("sup_heads", sup_obj) if isinstance(sup_obj, dict) else sup_obj
        if isinstance(sup_sd, dict) and "state_dict" in sup_sd and isinstance(sup_sd["state_dict"], dict):
            sup_sd = sup_sd["state_dict"]
        if isinstance(sup_sd, dict) and any(k.startswith("sup_heads.") for k in sup_sd.keys()):
            sup_sd = {k[len("sup_heads."):]: v for k, v in sup_sd.items()}
        tasks_auto, in_dim_auto = _infer_tasks_and_in_dim_from_sup_state(sup_sd)
        if (tasks_auto is None) and args.classes_json:
            with open(args.classes_json, "r", encoding="utf-8") as f:
                cj = json.load(f)
            if isinstance(cj, dict) and cj:
                tasks_auto = {str(t): len(v) for t, v in cj.items() if isinstance(v, (list, tuple))}
        if tasks_auto is None:
            raise RuntimeError("Impossible d'inférer les tâches depuis SupHeads en mode fusion.")
        if in_dim_auto is None:
            in_dim_auto = fusion_dim
        if int(in_dim_auto) != int(fusion_dim):
            print(f"[Fusion/SupHeads] input_dim du checkpoint={int(in_dim_auto)} différente de fusion_dim={int(fusion_dim)}. Utilisation de la dimension du checkpoint.")
        Sup = SupHeads(tasks_auto, int(in_dim_auto), token_mode="flat").to(device)
        missing, unexpected = Sup.load_state_dict(sup_sd, strict=False)
        if missing:
            print(f"[WARN] SupHeads(fusion): missing keys ({len(missing)})")
        if unexpected:
            print(f"[WARN] SupHeads(fusion): unexpected keys ({len(unexpected)})")
        Sup.eval()
        print(f"✓ SupHeads loaded (fusion) – {sup_path.name} | tasks={list(tasks_auto.keys())} | in_dim={int(in_dim_auto)}")

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
        if str(args.sup_feat_source).lower() == "fusion":
            class FusionWrap(nn.Module):
                def __init__(self, G: nn.Module, sem_backbone: nn.Module, fusion_head: nn.Module, Sup: nn.Module, imagenet_norm: bool = True):
                    super().__init__()
                    self.G = G
                    self.sem_backbone = sem_backbone
                    self.fusion_head = fusion_head
                    self.Sup = Sup
                    self.tasks = list(Sup.tasks.keys()) if hasattr(Sup, "tasks") else ["__DEFAULT__"]
                    self.feat_type = str(args.embed_type or "tok6")
                    self.delta_weights = str(args.delta_weights or "")
                    self.imagenet_norm = bool(imagenet_norm)
                    self.register_buffer("_im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
                    self.register_buffer("_im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

                def _sem_features(self, imgs: torch.Tensor):
                    x = imgs
                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)
                    if self.imagenet_norm:
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
                    return feat.mean(dim=(2, 3))

                def sup_features(self, imgs: torch.Tensor):
                    if hasattr(self.G, "sup_features"):
                        style_feat = self.G.sup_features(imgs, self.feat_type, delta_weights=self.delta_weights)
                    else:
                        z, _ = self.G.encode_content(imgs)
                        style_feat = F.adaptive_avg_pool2d(z, 1).flatten(1)
                    sem_feat = self._sem_features(imgs)
                    return self.fusion_head(style_feat, sem_feat)

                def forward(self, imgs: torch.Tensor, *, return_task_embeddings: bool = False, return_embeddings: bool = False):
                    feats = self.sup_features(imgs)
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

            composite = FusionWrap(G, sem_backbone, fusion_head, Sup, imagenet_norm=bool(int(args.sem_imagenet_norm))).to(device)
            print(f"✓ composite(fusion) prêt | tasks={composite.tasks} | feat_type={composite.feat_type} | imagenet_norm={bool(int(args.sem_imagenet_norm))}")
        elif args.feature_mode == "sem_resnet50":
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
    elif args.mode in {"sup_predict", "inference", "backbone_camera"}:
        raise RuntimeError("Les modes 'sup_predict', 'inference' et 'backbone_camera' requièrent un SupHeads (--sup_ckpt).")

    def _iter_unique_named_parameters(module: nn.Module | None):
        if module is None or not isinstance(module, nn.Module):
            return
        seen = set()
        for name, p in module.named_parameters(recurse=True):
            if p is None:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            yield name, p

    def _param_stats_from_module(module: nn.Module | None):
        stats = {
            "trainable": 0,
            "non_trainable": 0,
            "total": 0,
            "num_tensors": 0,
            "num_trainable_tensors": 0,
            "num_non_trainable_tensors": 0,
        }
        if module is None or not isinstance(module, nn.Module):
            return stats
        for _, p in _iter_unique_named_parameters(module):
            n = int(p.numel())
            stats["num_tensors"] += 1
            stats["total"] += n
            if bool(p.requires_grad):
                stats["trainable"] += n
                stats["num_trainable_tensors"] += 1
            else:
                stats["non_trainable"] += n
                stats["num_non_trainable_tensors"] += 1
        return stats

    def _param_stats_from_modules(modules):
        stats = {
            "trainable": 0,
            "non_trainable": 0,
            "total": 0,
            "num_tensors": 0,
            "num_trainable_tensors": 0,
            "num_non_trainable_tensors": 0,
        }
        seen = set()
        for module in modules:
            if module is None or not isinstance(module, nn.Module):
                continue
            for _, p in module.named_parameters(recurse=True):
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                n = int(p.numel())
                stats["num_tensors"] += 1
                stats["total"] += n
                if bool(p.requires_grad):
                    stats["trainable"] += n
                    stats["num_trainable_tensors"] += 1
                else:
                    stats["non_trainable"] += n
                    stats["num_non_trainable_tensors"] += 1
        return stats

    def _build_param_count_report():
        style_modules = []
        semantic_modules = []
        uncategorized_modules = []

        if G is not None:
            style_modules.append(("generator", G))
            if G_A is not None and G_A is not G:
                style_modules.append(("generator_A", G_A))
            if G_B is not None and G_B is not G and G_B is not G_A:
                style_modules.append(("generator_B", G_B))
        if Sup is not None:
            semantic_modules.append(("sup_heads", Sup))
        if sem_backbone is not None:
            semantic_modules.append(("semantic_backbone", sem_backbone))

        style_stats = _param_stats_from_modules([m for _, m in style_modules])
        semantic_stats = _param_stats_from_modules([m for _, m in semantic_modules])
        uncategorized_stats = _param_stats_from_modules([m for _, m in uncategorized_modules])
        total_stats = _param_stats_from_modules([m for _, m in (style_modules + semantic_modules + uncategorized_modules)])

        report = {
            "summary": {
                "style_part": style_stats,
                "semantic_content_part": semantic_stats,
                "uncategorized_part": uncategorized_stats,
                "total": total_stats,
            },
            "components": {
                "style_part": {name: _param_stats_from_module(mod) for name, mod in style_modules},
                "semantic_content_part": {name: _param_stats_from_module(mod) for name, mod in semantic_modules},
                "uncategorized_part": {name: _param_stats_from_module(mod) for name, mod in uncategorized_modules},
            },
            "metadata": {
                "mode": str(args.mode),
                "feature_mode": str(args.feature_mode),
                "sup_feat_source": str(args.sup_feat_source),
                "style_components_loaded": [name for name, _ in style_modules],
                "semantic_components_loaded": [name for name, _ in semantic_modules],
                "uncategorized_components_loaded": [name for name, _ in uncategorized_modules],
                "notes": {
                    "style_part": "Modules de génération / transfert de style chargés pendant le test.",
                    "semantic_content_part": "Backbone sémantique et/ou têtes supervisées chargés pendant le test.",
                    "non_trainable": "Paramètres ayant requires_grad=False.",
                },
            },
        }
        return report

    def _maybe_write_param_count_report(out_dir_hint: Path | None = None):
        if not bool(getattr(args, "dump_param_count_json", False)):
            return None
        report = _build_param_count_report()
        base_dir = Path(out_dir_hint) if out_dir_hint is not None else Path(args.out_dir or args.weights_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / str(args.param_count_json_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"✓ comptage des paramètres sauvegardé → {out_path}")
        return out_path

    if args.mode == "backbone_camera":
        if composite is None:
            raise RuntimeError("Le mode 'backbone_camera' requiert un modèle composite (backbone + SupHeads).")
        task_classes_cam = _load_task_classes_for_camera(args)
        if not task_classes_cam:
            print("[backbone_camera] Aucun mapping de classes trouvé via --classes_json / --imagenet_ann_dir / --imagenet_synset_mapping. Fallback sur des noms génériques 'class i'.")
        run_backbone_camera(
            composite,
            device,
            cfg=cfg,
            feature_mode=("fusion" if str(args.sup_feat_source).lower()=="fusion" else args.feature_mode),
            task_classes=task_classes_cam,
            cam_index=args.camera_index,
            topk=args.camera_topk,
        )
        return

    # ========================= MODE: inference ===========================
    if args.mode == "inference":
        if Sup is None:
            raise RuntimeError(
                "Le mode 'inference' requiert un SupHeads (fournis via --sup_ckpt)."
            )

        out_dir = Path(args.out_dir or (wdir / "inference"))
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks_list = list(Sup.tasks.keys()) if hasattr(Sup, "tasks") else ["__DEFAULT__"]

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
            if not names and getattr(args, "classes_json", None):
                try:
                    with open(args.classes_json, "r", encoding="utf-8") as f:
                        cj = _normalize_task_classes_for_display(json.load(f))
                    if t in cj:
                        names = list(cj[t])
                    elif "__DEFAULT__" in cj:
                        names = list(cj["__DEFAULT__"])
                except Exception:
                    pass
            if not names:
                cam_tc = _load_task_classes_for_camera(args)
                if t in cam_tc:
                    names = list(cam_tc[t])
                elif "__DEFAULT__" in cam_tc:
                    names = list(cam_tc["__DEFAULT__"])
            if not names:
                names = [f"class {i}" for i in range(32)]
            safe_task_cls_global[t] = names

        def _sample_path_at_index(ds_obj, ds_idx: int):
            try:
                sample = ds_obj.samples[ds_idx]
            except Exception:
                return None
            if isinstance(sample, dict):
                return sample.get("path", None)
            if isinstance(sample, (list, tuple)) and len(sample) >= 1:
                return sample[0]
            return None

        records = []
        flat_rows = []
        idx_global = 0
        for batch in loader:
            if len(batch) == 3:
                imgs, _raw, _paths = batch
            else:
                imgs, _raw = batch
                _paths = None
            imgs = imgs.to(device)

            with torch.no_grad():
                if str(args.sup_feat_source).lower() == "fusion":
                    logits = composite(imgs)
                else:
                    if args.feature_mode == "sem_resnet50":
                        feats = composite.sup_features(imgs)
                    else:
                        if hasattr(G, "sup_features"):
                            feats = G.sup_features(
                                imgs, args.embed_type, delta_weights=args.delta_weights
                            )
                        else:
                            z, _ = G.encode_content(imgs)
                            feats = F.adaptive_avg_pool2d(z, 1).flatten(1)
                    logits = Sup(feats)
                if not isinstance(logits, dict):
                    logits = {"__DEFAULT__": logits}

            B = imgs.size(0)
            for i in range(B):
                if isinstance(loader.dataset, Subset):
                    img_index = loader.dataset.indices[idx_global + i]
                    img_path = _sample_path_at_index(loader.dataset.dataset, img_index)
                else:
                    img_path = _sample_path_at_index(loader.dataset, idx_global + i)
                if img_path is None and _paths is not None and i < len(_paths):
                    img_path = _paths[i]

                base = os.path.basename(str(img_path)) if img_path else f"sample_{idx_global + i:06d}"
                image_id = os.path.splitext(base)[0]
                rec = {
                    "index": int(idx_global + i),
                    "image": base,
                    "image_id": image_id,
                    "path": str(img_path) if img_path is not None else None,
                    "predictions": {},
                }
                for t, out in logits.items():
                    cur = out[i]
                    probs = torch.softmax(cur, dim=0)
                    pred_idx = int(torch.argmax(probs).item())
                    pred_prob = float(probs[pred_idx].item())
                    names = safe_task_cls_global.get(t, [])
                    pred_label = names[pred_idx] if 0 <= pred_idx < len(names) else f"class {pred_idx}"
                    rec["predictions"][t] = {
                        "pred_idx": pred_idx,
                        "pred_label": pred_label,
                        "pred_prob": pred_prob,
                    }
                    flat_rows.append({
                        "image": base,
                        "image_id": image_id,
                        "path": str(img_path) if img_path is not None else "",
                        "task": t,
                        "pred_idx": pred_idx,
                        "pred_label": pred_label,
                        "pred_prob": pred_prob,
                    })
                records.append(rec)
            idx_global += B

        report = {
            "metadata": {
                "mode": "inference",
                "feature_mode": str(args.feature_mode),
                "sup_feat_source": str(args.sup_feat_source),
                "num_images": int(len(records)),
                "tasks": tasks_list,
            },
            "predictions": records,
        }

        json_path = out_dir / str(args.inference_json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"✓ prédictions sauvegardées → {json_path}")

        if bool(args.inference_save_csv):
            import csv
            csv_path = out_dir / "inference_predictions.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "image_id", "path", "task", "pred_idx", "pred_label", "pred_prob"])
                writer.writeheader()
                writer.writerows(flat_rows)
            print(f"✓ CSV d'inférence sauvegardé → {csv_path}")

        _maybe_write_param_count_report(out_dir)
        return

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
                if str(args.sup_feat_source).lower() == "fusion":
                    logits = composite(imgs)
                else:
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

        def _sample_path_at_index(ds_obj, ds_idx: int):
            try:
                sample = ds_obj.samples[ds_idx]
            except Exception:
                return None
            if isinstance(sample, dict):
                return sample.get("path", None)
            if isinstance(sample, (list, tuple)) and len(sample) >= 1:
                return sample[0]
            return None

        def _write_submission_csv(task_name: str, y_pred_arr: np.ndarray, class_names: list[str]):
            """
            Écrit un CSV de soumission compatible ImageNet avec les colonnes :
              - ImageId
              - PredictionString

            Pour CLS, PredictionString contient en priorité le synset / label texte si disponible,
            sinon l'indice de classe prédit sous forme de chaîne.
            ImageId est écrit sans extension.
            """
            try:
                ds_base = loader.dataset
                indices = None
                if isinstance(ds_base, Subset):
                    indices = list(getattr(ds_base, "indices", []))
                    ds_base = ds_base.dataset
                if not hasattr(ds_base, "samples"):
                    return None

                num_preds = int(len(y_pred_arr))
                if indices is None:
                    ds_indices = list(range(min(num_preds, len(ds_base.samples))))
                else:
                    ds_indices = indices[:num_preds]

                rows = []
                for ds_idx, pred in zip(ds_indices, y_pred_arr.tolist()[: len(ds_indices)]):
                    img_path = _sample_path_at_index(ds_base, ds_idx)
                    if not img_path:
                        continue

                    base = os.path.basename(str(img_path))
                    image_id = os.path.splitext(base)[0]
                    pred_i = int(pred)

                    pred_str = ""
                    if 0 <= pred_i < len(class_names):
                        pred_str = str(class_names[pred_i]).strip()
                    if not pred_str:
                        pred_str = str(pred_i)

                    rows.append((image_id, pred_str))

                if not rows:
                    return None

                sub_path = out_dir / f"submission_{task_name}.csv"
                with open(sub_path, "w", encoding="utf-8") as f:
                    f.write("ImageId,PredictionString\n")
                    for image_id, pred_str in rows:
                        f.write(f"{image_id},{pred_str}\n")
                return sub_path
            except Exception as e:
                print(f"[WARN] impossible d'écrire le CSV de soumission pour {task_name}: {e}")
                return None

        metrics = {}
        for t in tasks_list:
            names = safe_task_cls_global[t]
            y_pred = np.array(all_preds[t], dtype=np.int64)
            y_true = np.array(all_trues[t], dtype=np.int64)
            mask = (y_true >= 0) & (y_pred >= 0)
            valid_gt = int(mask.sum())
            submission_path = None

            if valid_gt == 0:
                acc = top1_acc = prec = rec = f1 = 0.0
                cm = None
                submission_path = _write_submission_csv(t, y_pred, names)
                print(f"[{t}] no GT labels → métriques supervisées ignorées")
            else:
                yt = y_true[mask]
                yp = y_pred[mask]
                acc = float((yp == yt).mean()) if yt.size > 0 else 0.0
                top1_acc = acc
                if _SK_OK and yt.size > 0:
                    prec = float(
                        precision_score(
                            yt,
                            yp,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    rec = float(
                        recall_score(
                            yt,
                            yp,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    f1 = float(
                        f1_score(
                            yt,
                            yp,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    labels_used = np.unique(yt)
                    labels_used = labels_used[(labels_used >= 0) & (labels_used < len(names))]
                    if labels_used.size == 0:
                        cm = None
                    else:
                        cm = confusion_matrix(yt, yp, labels=labels_used)
                else:
                    prec = rec = f1 = 0.0
                    cm = None

            metrics[t] = {
                "accuracy": acc,
                "top1_accuracy": top1_acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "num_predictions": int(len(y_pred)),
                "num_valid_gt": valid_gt,
                "supervised_metrics_available": bool(valid_gt > 0),
                "confusion_matrix": (cm.tolist() if cm is not None else None),
            }
            if cm is not None:
                metrics[t]["confusion_matrix_labels"] = [
                    names[int(i)] if 0 <= int(i) < len(names) else f"id{int(i)}"
                    for i in labels_used.tolist()
                ]
            if submission_path is not None:
                metrics[t]["submission_csv"] = str(submission_path)
            print(
                f"[{t}] acc={acc:.4f}  top1={top1_acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}"
            )

        vals = list(metrics.values())
        avg = {
            "accuracy": float(np.mean([v["accuracy"] for v in vals] or [0.0])),
            "top1_accuracy": float(np.mean([v["top1_accuracy"] for v in vals] or [0.0])),
            "precision": float(np.mean([v["precision"] for v in vals] or [0.0])),
            "recall": float(np.mean([v["recall"] for v in vals] or [0.0])),
            "f1": float(np.mean([v["f1"] for v in vals] or [0.0])),
            "mean_batch_time_sec": float(
                np.mean(times) if len(times) > 0 else 0.0
            ),
        }
        metrics["average"] = avg
        print(
            f"[AVERAGE] acc={avg['accuracy']:.4f}  top1={avg['top1_accuracy']:.4f}  prec={avg['precision']:.4f}  "
            f"rec={avg['recall']:.4f}  f1={avg['f1']:.4f}"
        )

        metrics_path = (
            Path(args.out_dir) if args.out_dir else (wdir / "sup_predict")
        ) / "sup_predict_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ métriques sauvegardées → {metrics_path}")
        _maybe_write_param_count_report(out_dir)
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
        _maybe_write_param_count_report(out_root)
        return

    # -------------- tsne / metrics / cls_tokens / style -------------------
    # Si feature_mode == "cls_tokens", on passe forcément par per_task/composite
    if args.per_task:
        if Sup is None:
            raise RuntimeError("--per_task requiert SupHeads (sup_ckpt).")

        def _eval_top1_percent(_composite, _loader):
            """Compute Top-1 accuracy (%) per task using SupHeads logits."""
            if _composite.Sup is None:
                return {}
            top1 = {t: [0, 0] for t in _composite.tasks}  # correct, total
            _composite.eval()
            for batch in _loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    imgs, yb = batch[0], batch[1]
                else:
                    continue
                imgs = imgs.to(device, non_blocking=True)
                feats = _composite._sup_features_from_G(imgs, feat_type=_composite.feat_type, delta_weights=_composite.delta_weights)
                logits_d = _composite.Sup(feats)
                # yb can be dict(task->tensor/list) or tensor/int
                if isinstance(yb, dict):
                    for t in _composite.tasks:
                        if t not in logits_d:
                            continue
                        yt = yb.get(t, None)
                        if yt is None:
                            continue
                        yt = torch.as_tensor(yt, device=device)
                        m = yt >= 0
                        if m.any():
                            pred = logits_d[t].argmax(1)
                            top1[t][0] += int((pred[m] == yt[m]).sum().item())
                            top1[t][1] += int(m.sum().item())
                else:
                    # single-task default
                    yt = torch.as_tensor(yb, device=device)
                    m = yt >= 0
                    if m.any() and ("__DEFAULT__" in logits_d or "default" in logits_d):
                        key = "__DEFAULT__" if "__DEFAULT__" in logits_d else "default"
                        pred = logits_d[key].argmax(1)
                        top1[key][0] += int((pred[m] == yt[m]).sum().item())
                        top1[key][1] += int(m.sum().item())
            out = {}
            for t, (c, n) in top1.items():
                if n > 0:
                    out[f"top1_{t}"] = 100.0 * (c / n)
            return out

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

            # Add Top-1 (%) if SupHeads is used
            try:
                top1_scores = _eval_top1_percent(composite, loader)
                if top1_scores:
                    scores = dict(scores or {})
                    scores.update(top1_scores)
            except Exception as _e:
                pass

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

        # Add Top-1 (%) if SupHeads is used
        try:
            top1_scores = _eval_top1_percent(composite, loader)
            if top1_scores:
                scores = dict(scores or {})
                scores.update(top1_scores)
        except Exception:
            pass

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
        _maybe_write_param_count_report(Path(wdir))
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

    _maybe_write_param_count_report(Path(wdir))


if __name__ == "__main__":
    main()