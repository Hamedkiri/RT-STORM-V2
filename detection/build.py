# detection/build.py
import os
import json
import torch
import torchvision.models as torchvision_models

from torchvision.models._utils import IntermediateLayerGetter

from models.detection.fastrnn_detector import FastRNNDetector


def build_resnet_backbone_from_moco(args, device):
    """
    Retourne:
      backbone_getter: IntermediateLayerGetter(resnet, {"layer4":"0"})
      out_channels: 2048 (ResNet50/101)
    """
    backbone_cnn = torchvision_models.__dict__[args.arch](zero_init_residual=True, pretrained=False)

    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        print(f"=> Chargement poids MoCo backbone depuis '{args.pretrained_weights}'")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        backbone_state = {}
        for k, v in state_dict.items():
            k_clean = k[len("module."):] if k.startswith("module.") else k
            if not k_clean.startswith("base_encoder."):
                continue
            new_k = k_clean[len("base_encoder."):]
            if new_k.startswith("fc."):
                continue
            backbone_state[new_k] = v

        original_sd = backbone_cnn.state_dict()
        original_sd.update(backbone_state)
        backbone_cnn.load_state_dict(original_sd, strict=True)
        print("Backbone ResNet chargé depuis MoCo (strict=True).")
    else:
        if args.pretrained_weights:
            print(f"=> Warning: poids '{args.pretrained_weights}' introuvable. Backbone random.")

    backbone = IntermediateLayerGetter(backbone_cnn, {"layer4": "0"})
    backbone.out_channels = 2048

    if args.freeze_backbone:
        print("=> Gel du backbone.")
        for p in backbone.parameters():
            p.requires_grad = False

    return backbone, 2048


def build_fastrnn_detector_from_moco_resnet(args, device):
    # (Optionnel) mapping des classes comme tu fais déjà
    # -> args.num_detection_classes doit inclure background (donc K+1)
    backbone, out_ch = build_resnet_backbone_from_moco(args, device)

    det = FastRNNDetector(
        backbone=backbone,
        out_channels=out_ch,
        num_classes=args.num_detection_classes,
        head_hidden=int(getattr(args, "fastrnn_hidden", 256)),
        head_bidir=bool(getattr(args, "fastrnn_bidir", True)),
        head_dropout=float(getattr(args, "fastrnn_dropout", 0.0)),
        size_divisible=32,
        focal_alpha=float(getattr(args, "fastrnn_focal_alpha", 0.25)),
        focal_gamma=float(getattr(args, "fastrnn_focal_gamma", 2.0)),
        score_thresh=float(getattr(args, "fastrnn_score_thresh", 0.05)),
        nms_thresh=float(getattr(args, "fastrnn_nms_thresh", 0.5)),
        topk=int(getattr(args, "fastrnn_topk", 1000)),
    ).to(device)

    return det
