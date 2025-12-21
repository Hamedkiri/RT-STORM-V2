# file: training/train_detection_transformer.py

import json
import torch
from tqdm import tqdm
from pathlib import Path

from data import build_detection_dataloader
from models.generator import UNetGenerator
from models.det_transformer import SimpleDETRHead
from training.detr_criterion import SetCriterionDETR
from helpers import freeze, unfreeze


def _parse_save_freq(save_freq_str):
    """
    Interprète une chaîne save_freq (SAUVEGARDE PAR ÉPOQUE UNIQUEMENT) :

      - 'none'       -> ('none', None)   → pas de snapshots intermédiaires
      - 'epoch'      -> ('epoch', 1)     → chaque époque
      - 'epoch:5'    -> ('epoch', 5)     → toutes les 5 époques
      - '5'          -> ('epoch', 5)     → alias de 'epoch:5'

    Toute autre valeur est ignorée → ('none', None).
    """
    if save_freq_str is None:
        return "none", None
    sf = str(save_freq_str).strip().lower()
    if sf == "" or sf == "none":
        return "none", None

    if sf.startswith("epoch"):
        parts = sf.split(":", 1)
        if len(parts) == 2:
            try:
                n = int(parts[1])
                return "epoch", max(1, n)
            except ValueError:
                pass
        return "epoch", 1

    # Si c'est juste un entier → interprété comme epoch:N
    try:
        n = int(sf)
        return "epoch", max(1, n)
    except ValueError:
        # Valeur non reconnue → désactive
        return "none", None


def train_detection_transformer(opt, dev):
    """
    Mode détection uniquement.

    Deux têtes possibles (choisies via opt.det_head_type ou config) :

      1) 'simple_unet' (défaut) :
         - Backbone = UNetGenerator
         - Tête     = SimpleDETRHead (transformer style DETR)
         - Critère  = SetCriterionDETR (Hungarian + L1 + GIoU, etc.)

      2) 'detr_resnet50' :
         - Modèle complet = torchvision.models.detection.detr_resnet50
         - Critère = celui intégré à torchvision (loss_dict = model(images, targets))

    Options importantes attendues dans opt / config :
      - det_head_type ∈ {"simple_unet", "detr_resnet50"}
      - det_backbone_ckpt : checkpoint backbone (pour simple_unet) ou modèle complet (pour detr_resnet50)
      - det_feat_branch : "content" | "style" | "concat" (simple_unet)
      - det_freeze_backbone : True/False
      - det_lr, det_weight_decay
      - det_num_queries, det_nheads, det_dec_layers, det_token_dim, det_d_model (simple_unet)
      - save_freq : "none" | "epoch" | "epoch:N" | "N"
      - epoch_ckpt_interval : (optionnel) surcharge l'intervalle en mode 'epoch'
    """
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data détection
    train_loader, val_loader, num_classes = build_detection_dataloader(opt)
    # num_classes inclut la classe "fond" (0) si défini ainsi dans build_detection_dataloader.

    # 2) Choix de la tête de détection
    det_head_type = str(getattr(opt, "det_head_type", "simple_unet")).lower()
    print(f"🔧 Tête de détection choisie : {det_head_type}")

    # --- Paramètres communs pour l'optimisation & sauvegarde ---
    save_freq_mode, save_freq_interval = _parse_save_freq(
        getattr(opt, "save_freq", "epoch")
    )
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    lr = float(getattr(opt, "det_lr", getattr(opt, "lr", 1e-4)))
    weight_decay = float(getattr(opt, "det_weight_decay", 1e-4))

    epochs = int(getattr(opt, "epochs", 50))
    best_val_loss = float("inf")
    global_step = 0  # pour info/debug

    # Quelques variables pour uniformiser le code
    criterion = None           # utilisé seulement pour 'simple_unet'
    backbone_module = None     # pour gérer le gel / dégel
    freeze_backbone = bool(getattr(opt, "det_freeze_backbone", False))

    # 3) Construction du modèle selon la tête choisie
    if det_head_type == "simple_unet":
        # Backbone = UNetGenerator
        token_dim = int(getattr(opt, "det_token_dim", 256))
        d_model = int(getattr(opt, "det_d_model", token_dim))  # par défaut même dim
        G_det = UNetGenerator(token_dim=token_dim).to(dev)

        # Option pour charger des poids pré-entraînés (style auto-supervisé),
        # en privilégiant G_B si présent dans le checkpoint.
        det_backbone_ckpt = getattr(opt, "det_backbone_ckpt", None)
        if det_backbone_ckpt:
            ckpt = torch.load(det_backbone_ckpt, map_location="cpu")
            if isinstance(ckpt, dict):
                if "G_B" in ckpt:
                    state_dict = ckpt["G_B"]
                    src_key = "G_B"
                elif "G_A" in ckpt:
                    state_dict = ckpt["G_A"]
                    src_key = "G_A"
                elif "backbone" in ckpt:
                    state_dict = ckpt["backbone"]
                    src_key = "backbone"
                else:
                    state_dict = ckpt
                    src_key = "<full_dict>"
            else:
                state_dict = ckpt
                src_key = "<non_dict>"

            try:
                G_det.load_state_dict(state_dict, strict=False)
                print(
                    f"✓ Backbone UNetGenerator chargé depuis {det_backbone_ckpt} "
                    f"(clé {src_key})"
                )
            except Exception as e:
                print(
                    f"⚠️ Erreur lors du chargement du backbone depuis "
                    f"{det_backbone_ckpt} (clé {src_key}) : {e}"
                )

        feat_branch = str(getattr(opt, "det_feat_branch", "content"))
        num_queries = int(getattr(opt, "det_num_queries", 300))
        nheads = int(getattr(opt, "det_nheads", 8))
        num_decoder_layers = int(getattr(opt, "det_dec_layers", 6))

        print(
            f"🔍 Détection (simple_unet) "
            f"feat_branch={feat_branch}, num_queries={num_queries}, "
            f"token_dim={token_dim}, d_model={d_model}, "
            f"nheads={nheads}, layers={num_decoder_layers}"
        )

        det_model = SimpleDETRHead(
            generator=G_det,
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=d_model,
            nheads=nheads,
            num_decoder_layers=num_decoder_layers,
            feat_branch=feat_branch,
        ).to(dev)

        # Geler ou non le backbone
        backbone_module = G_det
        if freeze_backbone:
            freeze(G_det)
            print("🧊 Backbone gelé (G_det - UNetGenerator)")
        else:
            unfreeze(G_det)
            print("🔥 Backbone entraînable (G_det - UNetGenerator)")

        # Optimizer : tête seule ou tête + backbone
        params_head = [p for p in det_model.parameters() if p.requires_grad]
        params_backbone = [p for p in G_det.parameters() if p.requires_grad]

        if freeze_backbone:
            optim_params = params_head
        else:
            optim_params = list(
                {id(p): p for p in (params_head + params_backbone)}.values()
            )

        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)

        # Critère DETR custom
        criterion = SetCriterionDETR(
            num_classes=num_classes,
            matcher=None,  # GreedyMatcher par défaut dans ton implémentation
            eos_coef=float(getattr(opt, "det_eos_coef", 0.1)),
        )

    elif det_head_type == "detr_resnet50":
        # Modèle complet DETR ResNet-50 torchvision
        try:
            from torchvision.models.detection import detr_resnet50
        except ImportError as e:
            raise ImportError(
                "detr_resnet50 n'est pas disponible dans cette version de torchvision. "
                "Installe une version plus récente ou utilise det_head_type='simple_unet'."
            ) from e

        det_model = detr_resnet50(weights=None, num_classes=num_classes).to(dev)

        # Option pour charger un ckpt de modèle complet
        det_backbone_ckpt = getattr(opt, "det_backbone_ckpt", None)
        if det_backbone_ckpt:
            ckpt = torch.load(det_backbone_ckpt, map_location="cpu")

            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
                src_key = "model"
            else:
                state_dict = ckpt
                src_key = "<full_dict>"

            try:
                det_model.load_state_dict(state_dict, strict=False)
                print(
                    f"✓ Modèle DETR ResNet50 chargé depuis {det_backbone_ckpt} "
                    f"(clé {src_key})"
                )
            except Exception as e:
                print(
                    f"⚠️ Erreur lors du chargement du modèle DETR "
                    f"depuis {det_backbone_ckpt} : {e}"
                )

        # Gestion éventuelle du gel de backbone (si accessible)
        if hasattr(det_model, "backbone"):
            backbone_module = det_model.backbone
        else:
            backbone_module = None

        if freeze_backbone:
            if backbone_module is not None:
                freeze(backbone_module)
                print("🧊 Backbone gelé (DETR ResNet50.backbone)")
            else:
                freeze(det_model)
                print(
                    "🧊 det_freeze_backbone demandé, mais aucun 'backbone' explicite ; "
                    "gel de tout le modèle."
                )
        else:
            unfreeze(det_model)
            print("🔥 DETR ResNet50 entraînable (entier ou backbone)")

        optimizer = torch.optim.AdamW(
            [p for p in det_model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        print(
            "🔍 Détection avec DETR ResNet50 (torchvision). "
            "Critère = pertes intégrées au modèle."
        )

    else:
        raise ValueError(
            f"det_head_type inconnu : '{det_head_type}' "
            "(attendu : 'simple_unet' ou 'detr_resnet50')."
        )

    # 3.bis) Sauvegarde des hyperparamètres dans un JSON (pour faciliter les tests)
    hparams = {
        "det_head_type": det_head_type,
        "num_classes": int(num_classes),
        "det_lr": float(lr),
        "det_weight_decay": float(weight_decay),
        "det_freeze_backbone": bool(freeze_backbone),
        "epochs": int(epochs),
    }
    if det_head_type == "simple_unet":
        hparams.update(
            {
                "backbone_type": "UNetGenerator",
                "det_feat_branch": str(getattr(opt, "det_feat_branch", "content")),
                "det_num_queries": int(getattr(opt, "det_num_queries", 300)),
                "det_nheads": int(getattr(opt, "det_nheads", 8)),
                "det_dec_layers": int(getattr(opt, "det_dec_layers", 6)),
                "det_token_dim": int(getattr(opt, "det_token_dim", 256)),
                "det_d_model": int(getattr(opt, "det_d_model", int(getattr(opt, "det_token_dim", 256)))),
                "det_eos_coef": float(getattr(opt, "det_eos_coef", 0.1)),
            }
        )
    else:  # detr_resnet50
        hparams.update(
            {
                "backbone_type": "ResNet50 (torchvision DETR)",
            }
        )

    hparams_path = save_dir / "det_hparams.json"
    hparams_path.write_text(json.dumps(hparams, indent=2, sort_keys=True))
    print(f"📝 Hyperparamètres de détection sauvegardés → {hparams_path}")

    # 4) Boucle d'entraînement
    for epoch in range(epochs):
        det_model.train()
        if backbone_module is not None:
            backbone_module.train(not freeze_backbone)

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[DET-{det_head_type}] epoch {epoch+1}/{epochs}",
            ncols=160,
            leave=False,
        )

        for imgs, targets in pbar:
            imgs = imgs.to(dev)
            targets_list = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if det_head_type == "detr_resnet50":
                # Pour torchvision DETR, on a besoin d'une liste d'images
                imgs_list = [img for img in imgs]  # chaque img: (3,H,W)
                loss_dict = det_model(imgs_list, targets_list)  # dict de pertes
                loss = sum(loss_dict.values())
            else:  # 'simple_unet'
                pred_logits, pred_boxes = det_model(imgs)
                outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                loss_dict = criterion(outputs, targets_list)
                loss = sum(
                    loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                    for k in loss_dict.keys()
                )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{total_loss / max(1, n_batches):.4f}")

        train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{epochs} - train loss = {train_loss:.4f}")

        # --- Validation simple (même critère, sans gradient) ---
        # Pour avoir des pertes avec torchvision DETR, il faut le laisser en mode 'train'
        if det_head_type == "detr_resnet50":
            was_training = det_model.training
            det_model.train()
        else:
            det_model.eval()
            if backbone_module is not None:
                backbone_module.eval()

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(dev)
                targets_list = [{k: v.to(dev) for k, v in t.items()} for t in targets]

                if det_head_type == "detr_resnet50":
                    imgs_list = [img for img in imgs]
                    loss_dict = det_model(imgs_list, targets_list)
                    loss = sum(loss_dict.values())
                else:
                    pred_logits, pred_boxes = det_model(imgs)
                    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                    loss_dict = criterion(outputs, targets_list)
                    loss = sum(
                        loss_dict[k] * criterion.weight_dict.get(k, 1.0)
                        for k in loss_dict.keys()
                    )

                val_loss += float(loss.item())
                val_batches += 1

        val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch+1}/{epochs} - val loss = {val_loss:.4f}")

        if det_head_type == "detr_resnet50":
            det_model.train(was_training)

        # --- Sauvegarde last + best (à CHAQUE époque) ---
        if det_head_type == "simple_unet":
            ckpt_last = {
                "epoch": epoch,
                "global_step": global_step,
                "backbone": G_det.state_dict(),
                "det_head": det_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "feat_branch": str(getattr(opt, "det_feat_branch", "content")),
                "num_classes": int(num_classes),
                "head_type": det_head_type,
                "hparams": hparams,
            }
        else:  # detr_resnet50
            ckpt_last = {
                "epoch": epoch,
                "global_step": global_step,
                "model": det_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_classes": int(num_classes),
                "head_type": det_head_type,
                "hparams": hparams,
            }

        torch.save(ckpt_last, save_dir / "detector_last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_last, save_dir / "detector_best.pth")
            print(f"✓ Nouveau meilleur modèle (val loss={val_loss:.4f}) sauvegardé.")

        # --- Sauvegarde en fonction de save_freq (mode 'epoch') ---
        if save_freq_mode == "epoch" and save_freq_interval is not None:
            if (epoch + 1) % save_freq_interval == 0:
                ckpt_epoch_path = save_dir / f"detector_epoch_{epoch+1:04d}.pth"
                torch.save(ckpt_last, ckpt_epoch_path)
                print(f"💾 Checkpoint (epoch) sauvegardé → {ckpt_epoch_path}")

    print("✅ Entraînement détection terminé.")
