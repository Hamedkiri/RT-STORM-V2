# data.py
# -*- coding: utf-8 -*-
"""
Datasets & dataloaders:
- Style / SSL datasets (JSON, ImageFolder, flat folders) -> build_dataloader(opt)
- Detection COCO-like dataset -> build_detection_dataloader(opt)

✅ Goal (align with main_moco_modified_classifier.py detection defaults):
- Detection images are returned as list[Tensor] + list[Target] (NOT stacked),
  so torchvision FasterRCNN can apply its own internal transform (normalize/resize).
- No Normalize([-1,1]) in detection.
- Random init backbone / SSL ckpt handled elsewhere (training script).
- COCO category ids are mapped to contiguous labels 1..K (bg=0), like main_moco mapping.
- Robust filtering: invalid bbox, iscrowd, optional drop_empty at dataset init.

If you *need* fixed-size tensors + torch.stack (for your custom code), you can enable:
  --det_fixed_size 1  --det_img_h 256 --det_img_w 256
and (optionally) --det_stack_batch 1
But by default, it behaves like main_moco (variable-size, list inputs).

Expected COCO-like annotation JSON:
{
  "images": [{"id":..., "file_name":..., "width":..., "height":...}, ...],
  "annotations": [{"image_id":..., "category_id":..., "bbox":[x,y,w,h], "iscrowd":0, "area":...}, ...],
  "categories": [{"id":..., "name":"..."}, ...]
}
"""

import os, json, random
from typing import Optional, Dict, Any, List, Tuple, Iterable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image


# -----------------------------
# Utils
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Unlabeled dataset (folder flat)
# -----------------------------
class UnlabeledImageDataset(Dataset):
    """
    Dataset pour auto-supervisé quand on a un dossier SANS classes.
    - root peut contenir des images directement OU en sous-dossiers arbitraires.
    - retourne (image_tensor, 0)
    """
    def __init__(self, root: str, transform=None, recursive: bool = True):
        self.root = str(root)
        self.transform = transform
        root_p = Path(self.root)

        if not root_p.exists():
            raise FileNotFoundError(f"[UnlabeledImageDataset] root introuvable: {self.root}")

        if recursive:
            paths = [p for p in root_p.rglob("*") if p.is_file() and is_image_path(p)]
        else:
            paths = [p for p in root_p.glob("*") if p.is_file() and is_image_path(p)]

        if len(paths) == 0:
            raise FileNotFoundError(
                f"[UnlabeledImageDataset] Aucune image trouvée dans: {self.root} "
                f"(extensions supportées: {sorted(list(IMG_EXTS))})"
            )

        self.samples = [str(p) for p in sorted(paths)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # label dummy


# -----------------------------
# JSON datasets (supervised multitask / or unsupervised paths-only)
# -----------------------------
class MultiTaskDataset(torch.utils.data.Dataset):
    """
    Dataset multitâche supervisé basé sur:
      - data_json: dict {folder: {img_name: {image_path: ..., task1:..., task2:...}}}
      - classes_json: dict {task: [class names]}
    Retourne: (img_tensor, labels_dict)
    """
    def __init__(self, data_json, classes_json, transform=None, search_folder=None, find_images_by_sub_folder=None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        with open(classes_json, 'r') as f:
            self.classes = json.load(f)

        self.transform = transform
        self.search_folder = search_folder
        self.find_images_by_sub_folder = find_images_by_sub_folder
        self.samples = []
        self.class_to_idx = {}
        self.task_classes = {}

        # Construire la correspondance des classes
        for task, class_list in self.classes.items():
            self.task_classes[task] = class_list
            self.class_to_idx[task] = {cls.lower(): idx for idx, cls in enumerate(class_list)}

        # Construire la liste des échantillons
        for folder, images in self.data.items():
            for _img_name, img_info in images.items():
                orig_path = img_info['image_path']
                if self.search_folder:
                    image_identifier = os.path.join(self.search_folder, os.path.basename(orig_path))
                elif self.find_images_by_sub_folder:
                    subfolder = os.path.basename(os.path.dirname(orig_path))
                    image_identifier = os.path.join(
                        self.find_images_by_sub_folder,
                        subfolder,
                        os.path.basename(orig_path)
                    )
                else:
                    image_identifier = orig_path

                labels = {}
                for task in self.classes:
                    label_val = img_info.get(task)
                    if label_val is not None:
                        lbl = str(label_val).lower()
                        labels[task] = self.class_to_idx[task].get(lbl)
                        if labels[task] is None:
                            print(f"Warning: label '{lbl}' for task '{task}' not found")
                    else:
                        labels[task] = None

                self.samples.append((image_identifier, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, labels


class UnsupervisedJsonDataset(Dataset):
    """
    Dataset auto-supervisé basé sur data_json, sans classes_json.
    Il ignore tout sauf image_path.
    Retourne (img_tensor, 0)
    """
    def __init__(self, data_json: str, transform=None, search_folder=None, find_images_by_sub_folder=None):
        with open(data_json, "r") as f:
            data = json.load(f)

        self.transform = transform
        self.samples = []
        self.search_folder = search_folder
        self.find_images_by_sub_folder = find_images_by_sub_folder

        for folder, images in data.items():
            for _img_name, img_info in images.items():
                orig_path = img_info["image_path"]
                if self.search_folder:
                    path = os.path.join(self.search_folder, os.path.basename(orig_path))
                elif self.find_images_by_sub_folder:
                    subfolder = os.path.basename(os.path.dirname(orig_path))
                    path = os.path.join(self.find_images_by_sub_folder, subfolder, os.path.basename(orig_path))
                else:
                    path = orig_path
                self.samples.append(path)

        if len(self.samples) == 0:
            raise RuntimeError(f"[UnsupervisedJsonDataset] Aucun sample dans {data_json}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# -----------------------------
# ImageNet-like dataset (train via ImageFolder, val via annotations)
# -----------------------------
class ImageNetCLSLDataset(Dataset):
    """ILSVRC 2012 CLS-LOC style dataset (classification, optional bbox).

    Supports:
      - train: Data/CLS-LOC/train/<synset>/*.JPEG (labels from folder)
      - val:   Data/CLS-LOC/val/*.JPEG (labels from XML or LOC_val_solution.csv)
      - test:  Data/CLS-LOC/test/*.JPEG (no labels)

    If imagesets_root is provided, it uses ImageSets/CLS-LOC/{split}.txt to select ids.
    Synset->idx mapping is stable: prefers LOC_synset_mapping.txt order if provided.
    """

    def __init__(
        self,
        images_root: str,
        split: str,
        ann_root: str | None = None,
        imagesets_root: str | None = None,
        synset_mapping_file: str | None = None,
        val_solution_csv: str | None = None,
        transform=None,
        label_base: int = 1,
        num_classes: int = 1000,
        return_bbox: bool = False,
    ):
        self.images_root = str(images_root)
        self.split = self._normalize_split(split, self.images_root)
        self.ann_root = self._normalize_ann_root(ann_root, self.split)
        self.imagesets_root = str(imagesets_root) if imagesets_root else None
        self.synset_mapping_file = str(synset_mapping_file) if synset_mapping_file else None
        self.val_solution_csv = str(val_solution_csv) if val_solution_csv else None
        self.transform = transform
        self.label_base = int(label_base)
        self.num_classes = int(num_classes)
        self.return_bbox = bool(return_bbox)

        rp = Path(self.images_root)
        if not rp.exists():
            raise FileNotFoundError(f"[ImageNetCLSLDataset] images_root not found: {self.images_root}")

        self.synset_to_idx, self._synset_names = self._build_synset_to_idx(rp)

        # Expose ImageFolder-like attrs
        syn_by_idx = [None] * (max(self.synset_to_idx.values()) + 1 if self.synset_to_idx else 0)
        for syn, idx in self.synset_to_idx.items():
            if 0 <= idx < len(syn_by_idx):
                syn_by_idx[idx] = syn
        self.classes = [s for s in syn_by_idx if s is not None]
        self.class_to_idx = dict(self.synset_to_idx)
        self.class_names = [self._synset_names.get(s, s) for s in self.classes]

        # Read ids list from ImageSets if available
        ids = None
        if self.imagesets_root:
            f = Path(self.imagesets_root) / f"{self.split}.txt"
            if f.exists():
                ids = []
                for ln in f.read_text(encoding='utf-8').splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    ids.append(ln.split()[0])

        self._val_csv_map = None
        if self.split == 'val' and self.val_solution_csv:
            self._val_csv_map = self._parse_val_solution_csv(self.val_solution_csv)

        self.samples: list[dict] = []
        if self.split == 'train':
            self._build_train_samples(rp, ids)
        elif self.split == 'val':
            self._build_val_samples(rp, ids)
        elif self.split == 'test':
            self._build_test_samples(rp, ids)
        else:
            raise ValueError(f"[ImageNetCLSLDataset] split must be train|val|test, got {self.split}")

        if len(self.samples) == 0:
            raise RuntimeError(f"[ImageNetCLSLDataset] no samples found: root={self.images_root} split={self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        y = s.get('y', None)
        if not self.return_bbox:
            # for test split, y can be None; return -1 to keep collate simple
            return img, (-1 if y is None else int(y))

        out = {"default": (-1 if y is None else int(y)), "bbox": s.get("bbox", None)}
        return img, out

    # -------------------- mapping helpers --------------------
    def _normalize_split(self, split, images_root: str) -> str:
        s = str(split).lower() if split is not None else 'auto'
        if s in {'train', 'val', 'test'}:
            return s
        rp = Path(images_root)
        bn = rp.name.lower()
        if bn in {'train', 'val', 'test'}:
            return bn
        for cand in ('train', 'val', 'test'):
            if (rp / cand).exists():
                return cand if sum((rp / c).exists() for c in ('train', 'val', 'test')) == 1 else 'val'
        return 'val'

    def _normalize_ann_root(self, ann_root, split: str) -> str | None:
        if not ann_root:
            return None
        ap = Path(str(ann_root))
        if ap.name.lower() == split:
            ap = ap.parent
        return str(ap)

    def _build_synset_to_idx(self, rp: Path) -> tuple[Dict[str, int], Dict[str, str]]:
        names: Dict[str, str] = {}
        if self.synset_mapping_file and Path(self.synset_mapping_file).exists():
            synsets = []
            for ln in Path(self.synset_mapping_file).read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                syn = parts[0]
                if not syn.startswith('n'):
                    continue
                synsets.append(syn)
                if len(parts) > 1:
                    names[syn] = " ".join(parts[1:])
            synsets = synsets[: self.num_classes] if self.num_classes else synsets
            return {s: i for i, s in enumerate(synsets)}, names

        # fallback 1: from train folders
        train_root = (rp / 'train') if (rp / 'train').exists() else rp
        subdirs = [d for d in train_root.iterdir() if d.is_dir() and d.name.startswith('n')]
        synsets = sorted([d.name for d in subdirs])
        if synsets:
            synsets = synsets[: self.num_classes] if self.num_classes else synsets
            return {s: i for i, s in enumerate(synsets)}, names

        # fallback 2: infer from val XML annotations when using a flat val folder
        inferred = self._infer_synsets_from_val_annotations(rp)
        if inferred:
            inferred = inferred[: self.num_classes] if self.num_classes else inferred
            return {s: i for i, s in enumerate(inferred)}, names

        # fallback 3: from LOC_val_solution.csv
        if self.val_solution_csv and Path(self.val_solution_csv).exists():
            mp = self._parse_val_solution_csv(self.val_solution_csv)
            synsets = sorted({v for v in mp.values() if isinstance(v, str) and v.startswith('n')})
            synsets = synsets[: self.num_classes] if self.num_classes else synsets
            if synsets:
                return {s: i for i, s in enumerate(synsets)}, names

        return {}, names

    def _infer_synsets_from_val_annotations(self, rp: Path) -> list[str]:
        ann_base = Path(self.ann_root) if self.ann_root else None
        if ann_base is None:
            return []
        val_ann = ann_base / 'val'
        if not val_ann.exists():
            val_ann = ann_base
        if not val_ann.exists():
            return []
        synsets: set[str] = set()
        try:
            import xml.etree.ElementTree as ET
            for xml_path in sorted(val_ann.glob('*.xml')):
                try:
                    root = ET.parse(str(xml_path)).getroot()
                    for obj in root.findall('object'):
                        syn = (obj.findtext('name') or '').strip()
                        if syn.startswith('n'):
                            synsets.add(syn)
                except Exception:
                    continue
        except Exception:
            return []
        return sorted(synsets)

    def _parse_val_solution_csv(self, csv_path: str) -> Dict[str, str]:
        import csv
        mp = {}
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get('ImageId') or row.get('image_id') or row.get('ImageID')
                ps = row.get('PredictionString') or row.get('predictionstring')
                if not img_id or not ps:
                    continue
                parts = ps.strip().split()
                if parts:
                    mp[img_id] = parts[0]
        return mp

    def _xml_to_label_bbox(self, xml_path: Path) -> tuple[str | None, list[int] | None]:
        import xml.etree.ElementTree as ET
        if not xml_path.exists():
            return None, None
        try:
            root = ET.parse(str(xml_path)).getroot()
            obj = root.find('object')
            if obj is None:
                return None, None
            name = obj.findtext('name')
            bbox_el = obj.find('bndbox')
            bbox = None
            if bbox_el is not None:
                def _get(tag):
                    v = bbox_el.findtext(tag)
                    return int(float(v)) if v is not None else None
                xmin, ymin, xmax, ymax = _get('xmin'), _get('ymin'), _get('xmax'), _get('ymax')
                if None not in (xmin, ymin, xmax, ymax):
                    bbox = [xmin, ymin, xmax, ymax]
            return name, bbox
        except Exception:
            return None, None

    # -------------------- sample builders --------------------
    def _build_train_samples(self, rp: Path, ids: list[str] | None):
        train_root = (rp / 'train') if (rp / 'train').exists() else rp
        ann_root = Path(self.ann_root) / 'train' if self.ann_root else None

        for syn_dir in sorted([d for d in train_root.iterdir() if d.is_dir() and d.name.startswith('n')]):
            syn = syn_dir.name
            if syn not in self.synset_to_idx:
                continue
            y = self.synset_to_idx[syn]
            paths = [p for p in syn_dir.rglob('*') if p.is_file() and is_image_path(p)]
            for p in sorted(paths):
                rel_id = f"{syn}/{p.stem}"
                if ids is not None and rel_id not in ids and p.stem not in ids:
                    continue
                bbox = None
                if self.return_bbox and ann_root is not None:
                    xml_path = ann_root / syn / f"{p.stem}.xml"
                    _, bbox = self._xml_to_label_bbox(xml_path)
                self.samples.append({"path": str(p), "y": y, "bbox": bbox})

    def _build_val_samples(self, rp: Path, ids: list[str] | None):
        val_root = (rp / 'val') if (rp / 'val').exists() else rp
        ann_root = (Path(self.ann_root) / 'val') if self.ann_root else None

        # determine candidate image files
        img_files = sorted([p for p in val_root.iterdir() if p.is_file() and is_image_path(p)])
        for p in img_files:
            img_id = p.stem
            if ids is not None and img_id not in ids and p.name not in ids:
                continue
            syn = None
            bbox = None
            if ann_root is not None:
                xml_path = ann_root / f"{img_id}.xml"
                syn, bbox = self._xml_to_label_bbox(xml_path)
            if syn is None and self._val_csv_map is not None:
                syn = self._val_csv_map.get(img_id)
            y = self.synset_to_idx.get(syn, None) if syn else None
            self.samples.append({"path": str(p), "y": y, "bbox": bbox})

    def _build_test_samples(self, rp: Path, ids: list[str] | None):
        test_root = (rp / 'test') if (rp / 'test').exists() else rp
        img_files = sorted([p for p in test_root.iterdir() if p.is_file() and is_image_path(p)])
        for p in img_files:
            img_id = p.stem
            if ids is not None and img_id not in ids and p.name not in ids:
                continue
            self.samples.append({"path": str(p), "y": None, "bbox": None})


def _print_dataset_classes_once(ds, *, title: str, max_show: int = 20):
    """Pretty-print dataset class information."""
    if not hasattr(ds, "classes") or not ds.classes:
        return
    classes = list(ds.classes)
    names = list(getattr(ds, "class_names", []))
    print("=" * 88)
    print(f"[DATA] {title}")
    print(f"[DATA] num_classes={len(classes)}")
    show = min(max_show, len(classes))
    for i in range(show):
        syn = classes[i]
        nm = names[i] if (i < len(names) and names[i]) else ""
        extra = f" — {nm}" if nm and nm != syn else ""
        print(f"  {i:4d}: {syn}{extra}")
    if len(classes) > show:
        print(f"  ... ({len(classes)-show} more)")
    print("=" * 88)

    def _parse_val_solution_csv(self, csv_path: str) -> Dict[str, str]:
        import csv
        mp = {}
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # expected columns: ImageId, PredictionString
            for row in reader:
                img_id = row.get('ImageId') or row.get('image_id') or row.get('ImageID')
                ps = row.get('PredictionString') or row.get('predictionstring')
                if not img_id or not ps:
                    continue
                # PredictionString starts with synset then bbox coords (possibly multiple boxes)
                parts = ps.strip().split()
                if len(parts) >= 1:
                    mp[img_id] = parts[0]
        return mp

    def _xml_to_label_bbox(self, xml_path: Path) -> tuple[str | None, list[int] | None]:
        import xml.etree.ElementTree as ET
        if not xml_path.exists():
            return None, None
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            obj = root.find('object')
            if obj is None:
                return None, None
            name = obj.findtext('name')
            bbox_el = obj.find('bndbox')
            bbox = None
            if bbox_el is not None:
                def _get(tag):
                    v = bbox_el.findtext(tag)
                    return int(float(v)) if v is not None else None
                xmin, ymin, xmax, ymax = _get('xmin'), _get('ymin'), _get('xmax'), _get('ymax')
                if None not in (xmin, ymin, xmax, ymax):
                    bbox = [xmin, ymin, xmax, ymax]
            return name, bbox
        except Exception:
            return None, None

    # -------------------- sample builders --------------------
    def _build_train_samples(self, rp: Path, ids: list[str] | None):
        # images are in train/<synset>/*
        train_root = rp / 'train' if (rp / 'train').exists() else rp
        ann_root = Path(self.ann_root) / 'train' if self.ann_root else None

        for syn_dir in sorted([d for d in train_root.iterdir() if d.is_dir() and d.name.startswith('n')]):
            syn = syn_dir.name
            if syn not in self.synset_to_idx:
                continue
            y = self.synset_to_idx[syn]
            # list images
            paths = [p for p in syn_dir.rglob('*') if p.is_file() and is_image_path(p)]
            for p in sorted(paths):
                rel_id = f"{syn}/{p.stem}"  # used in imagesets
                if ids is not None and rel_id not in ids and p.stem not in ids:
                    continue
                bbox = None
                if self.return_bbox and ann_root is not None:
                    xmlp = ann_root / syn / f"{p.stem}.xml"
                    _, bbox = self._xml_to_label_bbox(xmlp)
                self.samples.append({'path': str(p), 'y': int(y), 'bbox': bbox})

    def _build_val_samples(self, rp: Path, ids: list[str] | None):
        val_root = rp / 'val' if (rp / 'val').exists() else rp
        ann_root = Path(self.ann_root) / 'val' if self.ann_root else None

        # list images
        paths = [p for p in val_root.glob('*') if p.is_file() and is_image_path(p)]
        paths = sorted(paths)
        for p in paths:
            img_id = p.stem
            if ids is not None and img_id not in ids:
                continue

            syn = None
            bbox = None
            if ann_root is not None:
                xmlp = ann_root / f"{img_id}.xml"
                syn, bbox = self._xml_to_label_bbox(xmlp)
            if syn is None and self._val_csv_map is not None:
                syn = self._val_csv_map.get(img_id)

            if syn is None:
                # no label -> skip (test-like)
                continue
            if syn not in self.synset_to_idx:
                # unknown synset -> skip
                continue
            y = self.synset_to_idx[syn]
            self.samples.append({'path': str(p), 'y': int(y), 'bbox': bbox if self.return_bbox else None})

    def _build_test_samples(self, rp: Path, ids: list[str] | None):
        test_root = rp / 'test' if (rp / 'test').exists() else rp
        paths = [p for p in test_root.glob('*') if p.is_file() and is_image_path(p)]
        paths = sorted(paths)
        for p in paths:
            img_id = p.stem
            if ids is not None and img_id not in ids:
                continue
            self.samples.append({'path': str(p), 'y': 0, 'bbox': None})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if not self.return_bbox:
            return img, int(s['y'])
        # multitask-like dict
        return img, {'default': int(s['y']), 'bbox': s.get('bbox')}
# -----------------------------
# ImageFolder structure detection
# -----------------------------
def _has_class_subfolders(root: str) -> bool:
    """
    Détecte si root ressemble à un ImageFolder:
    root/
      classA/...
      classB/...
    """
    rp = Path(root)
    if (not rp.exists()) or (not rp.is_dir()):
        return False
    subdirs = [p for p in rp.iterdir() if p.is_dir()]
    if len(subdirs) == 0:
        return False
    # au moins un sous-dossier qui contient une image
    for sd in subdirs:
        for f in sd.rglob("*"):
            if f.is_file() and is_image_path(f):
                return True
    return False


# -----------------------------
# Main: build_dataloader (STYLE/SSL)
# -----------------------------
def build_dataloader(opt):
    """
    Retourne une **liste** de DataLoader, de longueur k_folds.
    Si k_folds == 1 -> une liste d’un seul DataLoader.

    ⚠️ En mode détection (mode qui commence par 'detect'), on ne construit
    aucun dataloader de style → on retourne [].
    """
    mode = str(getattr(opt, "mode", "")).lower()

    # ------------------------------------------------------------------
    # 1) Cas détection : pas de dataset 'style' → on retourne une liste vide
    # ------------------------------------------------------------------
    if mode.startswith("detect"):
        print("[data] mode detect_* → aucun dataloader de style construit (build_dataloader -> []).")
        return []

    # ------------------------------------------------------------------
    # 2) Transforms (comme avant)
    # ------------------------------------------------------------------
    crop = int(getattr(opt, "crop_size", 256))
    resize = int(round(crop * 286 / 256))
    tf = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    data_json    = getattr(opt, "data_json", None)
    classes_json = getattr(opt, "classes_json", None)
    data_root    = getattr(opt, "data", None)
    imagenet_ann = getattr(opt, "imagenet_ann", None)
    imagenet_label_base = int(getattr(opt, "imagenet_label_base", 1))
    imagenet_num_classes = int(getattr(opt, "imagenet_num_classes", 1000))

    # ------------------------------------------------------------------
    # 3) Construction dataset
    # ------------------------------------------------------------------
    if data_json:
        # JSON mode
        if classes_json:
            full_ds = MultiTaskDataset(
                data_json=data_json,
                classes_json=classes_json,
                transform=tf,
                search_folder=getattr(opt, "search_folder", None),
                find_images_by_sub_folder=getattr(opt, "find_images_by_sub_folder", None),
            )
        else:
            # Auto-supervisé: pas besoin des classes
            full_ds = UnsupervisedJsonDataset(
                data_json=data_json,
                transform=tf,
                search_folder=getattr(opt, "search_folder", None),
                find_images_by_sub_folder=getattr(opt, "find_images_by_sub_folder", None),
            )
    else:
        # Folder mode
        if data_root is None:
            raise ValueError(
                "Ni --data_json ni --data n'ont été fournis, "
                "et le mode n'est pas 'detect_*'. Impossible de construire le dataset."
            )

        # ✅ ImageNet (ILSVRC 2012 CLS-LOC) supervised finetuning support
        # If root has class subfolders -> ImageNet train layout.
        # If root is flat -> ImageNet val/test layout (needs XML annotations for labels).
        imagenet_split = str(getattr(opt, "imagenet_split", "auto")).lower()
        imagenet_ann_dir = getattr(opt, "imagenet_ann_dir", None)
        imagenet_imagesets_dir = getattr(opt, "imagenet_imagesets_dir", None)
        imagenet_synset_mapping = getattr(opt, "imagenet_synset_mapping", None)
        imagenet_val_solution_csv = getattr(opt, "imagenet_val_solution_csv", None)
        imagenet_label_base = int(getattr(opt, "imagenet_label_base", 1))
        imagenet_num_classes = int(getattr(opt, "imagenet_num_classes", 1000))
        imagenet_return_bbox = bool(getattr(opt, "imagenet_return_bbox", False))

        has_sub = _has_class_subfolders(data_root)
        if imagenet_split == "auto":
            if has_sub:
                imagenet_split = "train"
            else:
                # try guess from folder name
                bn = Path(data_root).name.lower()
                imagenet_split = "test" if "test" in bn else "val"

        use_imagenet = (imagenet_ann_dir is not None) or (imagenet_imagesets_dir is not None) or (imagenet_synset_mapping is not None) or (imagenet_val_solution_csv is not None)

        if imagenet_split == "train" and has_sub and not use_imagenet:
            # simplest: ImageFolder
            full_ds = datasets.ImageFolder(root=data_root, transform=tf)
        elif imagenet_split in ("train","val","test"):
            # robust ILSVRC loader (train folder or flat val/test)
            full_ds = ImageNetCLSLDataset(
                images_root=data_root,
                split=imagenet_split,
                ann_root=imagenet_ann_dir,
                imagesets_root=imagenet_imagesets_dir,
                synset_mapping_file=imagenet_synset_mapping,
                val_solution_csv=imagenet_val_solution_csv,
                transform=tf,
                label_base=imagenet_label_base,
                num_classes=imagenet_num_classes,
                return_bbox=imagenet_return_bbox,
            )
        else:
            full_ds = UnlabeledImageDataset(root=data_root, transform=tf, recursive=True)
            # Auto-supervisé / style: on accepte un dossier flat
            # (et même en hybrid/sup_freeze on avertit)
            if mode in ("hybrid", "sup_freeze"):
                print(
                    "[WARN][data] mode supervisé/hybride mais dataset sans classes détecté. "
                    "La phase C risque de ne pas fonctionner. Utilise --data_json/--classes_json ou un ImageFolder."
                )
            full_ds = UnlabeledImageDataset(root=data_root, transform=tf, recursive=True)

    # ------------------------------------------------------------------
    # 3b) Print class info once (useful for supervised finetuning)
    # ------------------------------------------------------------------
    try:
        # Only print when supervision is used (sup_freeze / hybrid) or when ImageNet loader is active
        if mode in ("sup_freeze", "hybrid") or isinstance(full_ds, ImageNetCLSLDataset):
            if not hasattr(opt, "_printed_classes"):
                setattr(opt, "_printed_classes", True)
                _print_dataset_classes_once(full_ds, title=f"dataset={type(full_ds).__name__} | mode={mode}")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) k-fold split
    # ------------------------------------------------------------------
    k = max(1, int(getattr(opt, "k_folds", 1)))
    if k == 1:
        subsets = [full_ds]
    else:
        indices = list(range(len(full_ds)))
        seed_val = getattr(opt, "seed", 42)
        random.Random(seed_val).shuffle(indices)

        fold_sizes = [len(indices) // k] * k
        for i in range(len(indices) % k):
            fold_sizes[i] += 1

        splits, pos = [], 0
        for sz in fold_sizes:
            splits.append(indices[pos:pos + sz])
            pos += sz

        subsets = [Subset(full_ds, idxs) for idxs in splits]

    # ------------------------------------------------------------------
    # 5) Dataloaders
    # ------------------------------------------------------------------
    loaders = [
        DataLoader(
            sub,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=getattr(opt, "num_workers", 4),
            drop_last=True,
            pin_memory=True,
        )
        for sub in subsets
    ]
    return loaders


def infer_tasks_from_dataset(loader, opt):
    # récupère le dataset réel si Subset
    ds = loader.dataset.dataset if isinstance(loader.dataset, torch.utils.data.Subset) else loader.dataset

    # MultiTaskDataset -> .task_classes: dict {task: [class names]}
    if hasattr(ds, "task_classes") and isinstance(ds.task_classes, dict) and ds.task_classes:
        return {task: len(cls_list) for task, cls_list in ds.task_classes.items()}

    # ImageFolder -> .classes: list
    if hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and len(ds.classes) > 0:
        return {"default": len(ds.classes)}

    # ImageNetCLSLDataset (ILSVRC)
    if isinstance(ds, ImageNetCLSLDataset):
        return {"default": int(getattr(opt, "imagenet_num_classes", 1000))}

    # UnlabeledImageDataset / UnsupervisedJsonDataset -> fallback
    return {"default": int(getattr(opt, "sup_num_classes", 2))}


# ---------------------------------------------------------------------------------------
# Detection: COCO-like dataset (main_moco-like behavior by default)
# ---------------------------------------------------------------------------------------

def _clamp_boxes_(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    boxes: (N,4) in xyxy
    """
    if boxes.numel() == 0:
        return boxes
    boxes[:, 0].clamp_(0, max(0, w - 1))
    boxes[:, 2].clamp_(0, max(0, w - 1))
    boxes[:, 1].clamp_(0, max(0, h - 1))
    boxes[:, 3].clamp_(0, max(0, h - 1))
    return boxes


def _remove_invalid_boxes(target: Dict[str, Any], min_size: float = 1.0) -> Dict[str, Any]:
    """
    Supprime les boxes dégénérées (x2<=x1, y2<=y1 ou trop petites).
    """
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return target

    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)

    if keep.all():
        return target

    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]
    if "area" in target and isinstance(target["area"], torch.Tensor) and target["area"].numel() == keep.numel():
        target["area"] = target["area"][keep]
    if "iscrowd" in target and isinstance(target["iscrowd"], torch.Tensor) and target["iscrowd"].numel() == keep.numel():
        target["iscrowd"] = target["iscrowd"][keep]
    return target


# ---- Detection transforms (image + target together)
class DetCompose:
    def __init__(self, ops: List[Any]):
        self.ops = ops

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        for op in self.ops:
            img, target = op(img, target)
        return img, target


class DetToTensor:
    """PIL -> Tensor float32 in [0,1]. (No Normalize here.)"""
    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        return F.to_tensor(img), target


class DetRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.p:
            return img, target
        w, h = img.size
        img = F.hflip(img)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = float(w) - x2
            boxes[:, 2] = float(w) - x1
            target["boxes"] = _clamp_boxes_(boxes, w, h)
            target = _remove_invalid_boxes(target, min_size=1.0)

            # recompute area
            ws = (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0)
            hs = (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
            target["area"] = ws * hs

        return img, target


class DetResize:
    """
    Optional fixed-size resize (H,W) with bbox scaling.
    Use only if you really need fixed size batches.
    """
    def __init__(self, size_hw: Tuple[int, int], interpolation=transforms.InterpolationMode.BILINEAR):
        self.size_hw = (int(size_hw[0]), int(size_hw[1]))  # (H,W)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        orig_w, orig_h = img.size
        new_h, new_w = self.size_hw

        img = F.resize(img, [new_h, new_w], interpolation=self.interpolation)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            sx = float(new_w) / float(orig_w)
            sy = float(new_h) / float(orig_h)
            boxes = boxes.clone()
            boxes[:, 0] *= sx
            boxes[:, 2] *= sx
            boxes[:, 1] *= sy
            boxes[:, 3] *= sy
            target["boxes"] = _clamp_boxes_(boxes, new_w, new_h)
            target = _remove_invalid_boxes(target, min_size=1.0)

            ws = (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0)
            hs = (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
            target["area"] = ws * hs

        return img, target


def _parse_allowed_ids(opt, coco_categories: List[Dict[str, Any]]) -> Optional[List[int]]:
    """
    main_moco-like:
    - if opt.det_allowed_ids_json -> load list[int]
    - elif opt.det_allowed_ids -> "1,2,3"
    - else -> None (all categories in JSON order)
    """
    p = str(getattr(opt, "det_allowed_ids_json", "") or "").strip()
    if p:
        lst = _read_json(p)
        if not isinstance(lst, list):
            raise ValueError(f"--det_allowed_ids_json must be a JSON list[int], got {type(lst)}")
        return [int(x) for x in lst]

    s = str(getattr(opt, "det_allowed_ids", "") or "").strip()
    if s:
        out = []
        for tok in s.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))
        return out

    # None means "use all categories" in the JSON order
    return None


class CocoStyleDetectionDataset(Dataset):
    """
    COCO-like detection dataset.
    Returns: image_tensor, target_dict

    ✅ category_id -> contiguous labels 1..K (bg=0)
    ✅ drop_empty can remove empty images at init (recommended for FasterRCNN training)
    ✅ ignores iscrowd==1 and invalid bboxes
    """
    def __init__(
        self,
        img_root: str,
        ann_file: str,
        det_transforms=None,
        allowed_cat_ids: Optional[List[int]] = None,
        drop_empty: bool = True,
    ):
        super().__init__()
        self.img_root = str(img_root)
        self.ann_file = str(ann_file)
        self.det_transforms = det_transforms
        self.drop_empty = bool(drop_empty)

        coco = _read_json(self.ann_file)

        # images
        self.images = {img["id"]: img for img in coco.get("images", [])}
        if not self.images:
            raise RuntimeError(f"[CocoStyleDetectionDataset] No images found in ann_file: {self.ann_file}")

        # categories
        cats = coco.get("categories", [])
        if not isinstance(cats, list) or len(cats) == 0:
            raise RuntimeError(f"[CocoStyleDetectionDataset] No categories found in ann_file: {self.ann_file}")

        cat_ids_in_json = [int(c["id"]) for c in cats]
        if allowed_cat_ids is None:
            allowed_cat_ids = cat_ids_in_json
        else:
            allowed_cat_ids = [int(x) for x in allowed_cat_ids]

        # Keep only allowed categories that exist in JSON (stable order)
        allowed_set = set(allowed_cat_ids)
        allowed_order = [cid for cid in cat_ids_in_json if cid in allowed_set]
        if len(allowed_order) == 0:
            raise RuntimeError("[CocoStyleDetectionDataset] allowed_cat_ids is empty after filtering with JSON categories")

        # mapping: category_id -> label in [1..K]
        self.cat2label = {cid: i + 1 for i, cid in enumerate(allowed_order)}

        # annotations by image
        self.ann_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.images.keys()}
        for ann in coco.get("annotations", []):
            img_id = ann.get("image_id", None)
            if img_id not in self.ann_by_img:
                continue
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            cat_id = ann.get("category_id", None)
            if cat_id not in self.cat2label:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None or (not isinstance(bbox, (list, tuple))) or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            try:
                x, y, w, h = float(x), float(y), float(w), float(h)
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue

            self.ann_by_img[img_id].append(ann)

        all_ids = sorted(self.images.keys())
        self.original_len = len(all_ids)

        if self.drop_empty:
            kept, removed = [], 0
            for img_id in all_ids:
                if len(self.ann_by_img.get(img_id, [])) > 0:
                    kept.append(img_id)
                else:
                    removed += 1
            self.ids = kept
            self.removed_empty = int(removed)
        else:
            self.ids = all_ids
            self.removed_empty = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.images[img_id]
        file_name = info["file_name"]
        path = os.path.join(self.img_root, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[CocoStyleDetectionDataset] Image not found: {path}")

        img = Image.open(path).convert("RGB")

        anns = self.ann_by_img.get(img_id, [])
        boxes_list, labels_list, area_list, iscrowd_list = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            boxes_list.append([x1, y1, x2, y2])

            cat_id = int(ann["category_id"])
            labels_list.append(int(self.cat2label.get(cat_id, 0)))

            area_list.append(float(ann.get("area", w * h)))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_list, dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            area = torch.tensor(area_list, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd_list, dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.det_transforms is not None:
            img, target = self.det_transforms(img, target)

        # final safety clamp/remove invalid
        if isinstance(img, torch.Tensor):
            _, H, W = img.shape
            target["boxes"] = _clamp_boxes_(target["boxes"], W, H)
            target = _remove_invalid_boxes(target, min_size=1.0)

        return img, target


def build_detection_dataloader(opt):
    """
    main_moco-like defaults:
    - returns (train_loader, val_loader, num_classes)
    - images are list[Tensor] (variable sizes), targets are list[dict]
    - ToTensor only (+ optional hflip)
    - COCO category_id mapped to contiguous labels [1..K], bg=0 -> num_classes = K+1
    - drop_empty enabled by default

    Optional fixed-size mode (if you absolutely want torch.stack):
      --det_fixed_size 1 --det_img_h 256 --det_img_w 256
      --det_stack_batch 1
    """
    # required paths
    train_img_root = getattr(opt, "det_train_img_root", None)
    train_ann = getattr(opt, "det_train_ann", None)
    val_img_root = getattr(opt, "det_val_img_root", None)
    val_ann = getattr(opt, "det_val_ann", None)
    if not (train_img_root and train_ann and val_img_root and val_ann):
        raise ValueError(
            "Detection requires: --det_train_img_root, --det_train_ann, --det_val_img_root, --det_val_ann"
        )

    drop_empty = bool(int(getattr(opt, "det_drop_empty", 1)))
    hflip_p = float(getattr(opt, "det_hflip_prob", 0.0))

    # main_moco-like: no fixed resize by default
    fixed_size = bool(int(getattr(opt, "det_fixed_size", 0)))
    det_h = int(getattr(opt, "det_img_h", 256))
    det_w = int(getattr(opt, "det_img_w", 256))

    # build transforms
    train_ops = []
    if fixed_size:
        train_ops.append(DetResize((det_h, det_w), interpolation=transforms.InterpolationMode.BILINEAR))
    if hflip_p > 0:
        train_ops.append(DetRandomHorizontalFlip(p=hflip_p))
    train_ops.append(DetToTensor())
    train_tf = DetCompose(train_ops)

    val_ops = []
    if fixed_size:
        val_ops.append(DetResize((det_h, det_w), interpolation=transforms.InterpolationMode.BILINEAR))
    val_ops.append(DetToTensor())
    val_tf = DetCompose(val_ops)

    # load train coco once to get categories order and allowed ids
    train_coco = _read_json(train_ann)
    cats = train_coco.get("categories", [])
    if not isinstance(cats, list) or len(cats) == 0:
        raise RuntimeError(f"[DET][DATA] Train ann has no categories: {train_ann}")

    allowed_ids = _parse_allowed_ids(opt, cats)

    train_ds = CocoStyleDetectionDataset(
        img_root=train_img_root,
        ann_file=train_ann,
        det_transforms=train_tf,
        allowed_cat_ids=allowed_ids,  # None -> all in JSON order
        drop_empty=drop_empty,
    )

    # IMPORTANT: val uses EXACT same mapping keys as train (cat2label keys)
    val_ds = CocoStyleDetectionDataset(
        img_root=val_img_root,
        ann_file=val_ann,
        det_transforms=val_tf,
        allowed_cat_ids=list(train_ds.cat2label.keys()),
        drop_empty=drop_empty,
    )

    # collate behavior
    stack_batch = bool(int(getattr(opt, "det_stack_batch", 0)))

    def collate_fn(batch):
        imgs = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        if stack_batch:
            # only safe if fixed_size=True (same H,W)
            return torch.stack(imgs, dim=0), targets
        return imgs, targets

    train_loader = DataLoader(
        train_ds,
        batch_size=int(getattr(opt, "batch_size", 4)),
        shuffle=True,
        num_workers=int(getattr(opt, "num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(opt, "batch_size", 4)),
        shuffle=False,
        num_workers=int(getattr(opt, "num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    removed_train = int(getattr(train_ds, "removed_empty", 0))
    removed_val = int(getattr(val_ds, "removed_empty", 0))

    # bg=0, classes=1..K
    num_classes = len(train_ds.cat2label) + 1

    print(
        "[DET][DATA] drop_empty={} | fixed_size={} | stack_batch={} | "
        "train: {} -> {} (removed={}) | val: {} -> {} (removed={}) | num_classes={} (incl bg) | hflip_p={}".format(
            drop_empty, fixed_size, stack_batch,
            train_ds.original_len, len(train_ds), removed_train,
            val_ds.original_len, len(val_ds), removed_val,
            num_classes, hflip_p
        )
    )
    if fixed_size:
        print(f"[DET][DATA] fixed det_size = {det_h}x{det_w}")

    return train_loader, val_loader, num_classes


# ======================================================================
# ImageNetCLSLDataset fallback method definitions (robust packaging)
# Some environments may load an older data.py variant where these methods
# are not bound to the class due to packaging/indentation issues.
# This guard ensures the class always has the required builders.
# ======================================================================
try:
    ImageNetCLSLDataset
except NameError:
    ImageNetCLSLDataset = None

if ImageNetCLSLDataset is not None:
    from pathlib import Path as _PPath
    import os as _os
    import xml.etree.ElementTree as _ET

    def _in_val_ids(_ids, img_id):
        return True if _ids is None else (img_id in _ids or f"val/{img_id}" in _ids)

    def _parse_xml_label_bbox(self, xml_path: _PPath):
        try:
            root = _ET.parse(str(xml_path)).getroot()
        except Exception:
            return None, None
        # label via <object><name>
        syn = None
        bbox = None
        obj = root.find("object")
        if obj is not None:
            name = obj.findtext("name")
            if name:
                syn = name.strip()
            b = obj.find("bndbox")
            if b is not None:
                try:
                    xmin = int(float(b.findtext("xmin")))
                    ymin = int(float(b.findtext("ymin")))
                    xmax = int(float(b.findtext("xmax")))
                    ymax = int(float(b.findtext("ymax")))
                    bbox = (xmin, ymin, xmax, ymax)
                except Exception:
                    bbox = None
        return syn, bbox

    def _fallback_build_train_samples(self, rp: _PPath, ids):
        train_root = rp / "train" if (rp / "train").exists() else rp
        ann_root = _PPath(self.ann_root) / "train" if self.ann_root else None
        samples = []
        for syn in sorted([d.name for d in train_root.iterdir() if d.is_dir()]):
            if syn not in self.synset_to_idx:
                continue
            y = int(self.synset_to_idx[syn])
            for imgp in sorted((train_root / syn).glob("*.JPEG")):
                img_id = imgp.stem
                if ids is not None and not (img_id in ids or f"{syn}/{img_id}" in ids):
                    continue
                bbox = None
                if ann_root is not None:
                    xmlp = ann_root / syn / f"{img_id}.xml"
                    if xmlp.exists():
                        _, bbox = _parse_xml_label_bbox(self, xmlp)
                samples.append({"path": str(imgp), "y": y, "bbox": bbox})
        self.samples = samples

    def _fallback_build_val_samples(self, rp: _PPath, ids):
        val_root = rp / "val" if (rp / "val").exists() else rp
        ann_root = _PPath(self.ann_root) / "val" if self.ann_root else None
        samples=[]
        for imgp in sorted(val_root.glob("*.JPEG")):
            img_id = imgp.stem
            if ids is not None and not _in_val_ids(ids, img_id):
                continue
            syn=None; bbox=None
            if ann_root is not None:
                xmlp = ann_root / f"{img_id}.xml"
                if xmlp.exists():
                    syn, bbox = _parse_xml_label_bbox(self, xmlp)
            if syn is None and getattr(self, "_val_csv_map", None) is not None:
                syn = self._val_csv_map.get(img_id)
            if syn is None:
                continue
            if syn not in self.synset_to_idx:
                continue
            y=int(self.synset_to_idx[syn])
            samples.append({"path": str(imgp), "y": y, "bbox": bbox})
        self.samples=samples

    def _fallback_build_test_samples(self, rp: _PPath, ids):
        test_root = rp / "test" if (rp / "test").exists() else rp
        samples=[]
        for imgp in sorted(test_root.glob("*.JPEG")):
            img_id=imgp.stem
            if ids is not None and not (img_id in ids or f"test/{img_id}" in ids):
                continue
            samples.append({"path": str(imgp), "y": -1, "bbox": None})
        self.samples=samples

    if not hasattr(ImageNetCLSLDataset, "_build_train_samples"):
        ImageNetCLSLDataset._build_train_samples = _fallback_build_train_samples
    if not hasattr(ImageNetCLSLDataset, "_build_val_samples"):
        ImageNetCLSLDataset._build_val_samples = _fallback_build_val_samples
    if not hasattr(ImageNetCLSLDataset, "_build_test_samples"):
        ImageNetCLSLDataset._build_test_samples = _fallback_build_test_samples
