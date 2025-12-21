# data.py
import os, json, random
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
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
            for img_name, img_info in images.items():
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
            for img_name, img_info in images.items():
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
# Main: build_dataloader
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
        print("[data] mode détecté detect_* → aucun dataloader de style construit (build_dataloader → []).")
        return []

    # ------------------------------------------------------------------
    # 2) Transforms (comme avant)
    # ------------------------------------------------------------------
    tf = transforms.Compose([
        transforms.Resize(286, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    data_json    = getattr(opt, "data_json", None)
    classes_json = getattr(opt, "classes_json", None)
    data_root    = getattr(opt, "data", None)

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

        # ✅ Fix principal : si pas de folders classes → dataset unlabeled
        if _has_class_subfolders(data_root):
            full_ds = datasets.ImageFolder(root=data_root, transform=tf)
        else:
            # Auto-supervisé / style: on accepte un dossier flat
            # (et même en hybrid/sup_freeze on avertit)
            if mode in ("hybrid", "sup_freeze"):
                print(
                    "[WARN][data] mode supervisé/hybride mais dataset sans classes détecté. "
                    "La phase C risque de ne pas fonctionner. Utilise --data_json/--classes_json ou un ImageFolder."
                )
            full_ds = UnlabeledImageDataset(root=data_root, transform=tf, recursive=True)

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

    # UnlabeledImageDataset / UnsupervisedJsonDataset -> fallback
    return {"default": int(getattr(opt, "sup_num_classes", 2))}


# ---------------------------------------------------------------------------------------
# (Le reste : CocoStyleDetectionDataset + build_detection_dataloader inchangé)
# ---------------------------------------------------------------------------------------

class CocoStyleDetectionDataset(Dataset):
    """
    Dataset de détection au format COCO-like :
      - ann_file : JSON avec "images", "annotations", "categories"
      - img_root : dossier contenant les images
    Chaque item retourne : image_tensor, target_dict
    """
    def __init__(self, img_root, ann_file, transforms=None, class_ids=None):
        super().__init__()
        self.img_root = img_root
        self.ann_file = ann_file
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}

        cats = coco.get("categories", [])
        if class_ids is None:
            self.cat2label = {c["id"]: i + 1 for i, c in enumerate(cats)}
        else:
            self.cat2label = {cid: i + 1 for i, cid in enumerate(class_ids)}

        self.ann_by_img = {img_id: [] for img_id in self.images.keys()}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.ann_by_img:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue
            self.ann_by_img[img_id].append(ann)

        self.ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        file_name = info["file_name"]
        path = os.path.join(self.img_root, file_name)

        img = Image.open(path).convert("RGB")

        anns = self.ann_by_img.get(img_id, [])
        boxes, labels, area, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            cat_id = ann["category_id"]
            labels.append(self.cat2label.get(cat_id, 0))
            area.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def det_collate_fn(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return torch.stack(imgs, dim=0), targets


def build_detection_dataloader(opt):
    from torchvision import transforms as T

    train_tf = T.Compose([
        T.Resize((getattr(opt, "det_img_h", 256), getattr(opt, "det_img_w", 256))),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_tf = T.Compose([
        T.Resize((getattr(opt, "det_img_h", 256), getattr(opt, "det_img_w", 256))),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = CocoStyleDetectionDataset(
        img_root=opt.det_train_img_root,
        ann_file=opt.det_train_ann,
        transforms=train_tf,
    )
    val_ds = CocoStyleDetectionDataset(
        img_root=opt.det_val_img_root,
        ann_file=opt.det_val_ann,
        transforms=val_tf,
        class_ids=list(train_ds.cat2label.keys()),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=getattr(opt, "num_workers", 4),
        pin_memory=True,
        collate_fn=det_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=getattr(opt, "num_workers", 4),
        pin_memory=True,
        collate_fn=det_collate_fn,
    )

    return train_loader, val_loader, len(train_ds.cat2label) + 1  # +1 pour classe fond
