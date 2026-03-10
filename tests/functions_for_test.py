# =========================================================
#  testsFile/functionsTest.py
# =========================================================



# =========================================================
#  Détection transformer : chargement + metrics + caméra
# =========================================================

import torch


from pathlib import Path


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, colorchooser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os, json, numpy as np
import matplotlib
from typing import Optional, Any

from pathlib import Path as PPath           # ← alias sûr

from typing import Optional, Any, Sequence, Tuple, Dict, List
from torchvision import transforms, datasets
from data import MultiTaskDataset
from data import ImageNetCLSLDataset

from torch.utils.data import DataLoader, Subset

from typing import Tuple, Dict, List
from training.checkpoint import _remap_keys


# =========================================================
#  Semantic backbone (ResNet) helpers for evaluation / t-SNE
# =========================================================


def build_sem_backbone_for_eval(
    *,
    device: torch.device,
    arch: str = "resnet50",
    return_layer: str = "layer4",
    pretrained: bool = True,
    pretrained_path: str = "",
    strict: bool = False,
    verbose: bool = True,
    # --- NEW: auto-load semantic backbone weights from a run folder ---
    weights_dir: Optional[Path] = None,
    sem_filename: str = "SemBackbone",
    epoch: Optional[int] = None,
):
    """Build the same semantic ResNet backbone as the detection branch.

    If weights_dir is provided and contains saved semantic weights
    (e.g. SemBackbone_epochXX.pt / .safetensors), they are loaded automatically.

    Returns: (backbone, out_channels)
    """
    from training.train_detection_transformer import _build_sem_resnet_backbone

    backbone, out_ch = _build_sem_resnet_backbone(
        pretrained=bool(pretrained),
        arch=str(arch),
        return_layer=str(return_layer),
        pretrained_path=str(pretrained_path or ""),
        strict=bool(strict),
        verbose=bool(verbose),
    )
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # --- Auto-load saved weights if available ---
    if weights_dir is not None:
        try:
            wdir = Path(weights_dir)
            if epoch is not None:
                # explicit epoch
                cand = []
                cand += list(wdir.glob(f"{sem_filename}_epoch{int(epoch)}.safetensors"))
                cand += list(wdir.glob(f"{sem_filename}_epoch{int(epoch)}.pt"))
                wfile = cand[0] if cand else None
            else:
                # pick latest by mtime
                cand = sorted(
                    list(wdir.glob(f"{sem_filename}_epoch*.safetensors")) + list(wdir.glob(f"{sem_filename}_epoch*.pt")),
                    key=lambda p: p.stat().st_mtime,
                )
                wfile = cand[-1] if cand else None

            if wfile is not None and wfile.exists():
                if wfile.suffix == ".pt":
                    obj = torch.load(str(wfile), map_location="cpu")
                    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
                    backbone.load_state_dict(sd, strict=False)
                else:
                    # safetensors
                    try:
                        from safetensors.torch import load_file as st_load
                        sd = st_load(str(wfile))
                        backbone.load_state_dict(sd, strict=False)
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Impossible de charger {wfile} (safetensors): {e}")
                if verbose:
                    print(f"✓ sem_backbone chargé depuis {wfile}")
            else:
                if verbose:
                    print(f"[INFO] Aucun poids sémantique trouvé dans {wdir} (pattern: {sem_filename}_epoch*). Utilisation backbone initialisé.")
        except Exception as e:
            if verbose:
                print(f"[WARN] Auto-load sem_backbone échoué: {e}")

    return backbone, out_ch
def build_test_dataloader(opt, cfg) -> Tuple[DataLoader, torch.utils.data.Dataset, str]:
    """
    Construire DataLoader selon --data / --data_json ou config.
    Si --data est un dossier, les sous-dossiers sont interprétés comme classes (ImageFolder).
    On harmonise l'API en ajoutant .task_classes pour la voie 'folder', afin d'être
    compatible avec les chemins de code qui s'attendent à MultiTaskDataset.

    Retourne: (loader, dataset, dataset_type) où dataset_type ∈ {"json","folder"}.
    """
    tf = transforms.Compose([
        transforms.Resize(286, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # ---------- priorité aux options CLI ----------
    if getattr(opt, "data_json", None):
        # Multi-tâches au format JSON
        ds = MultiTaskDataset(opt.data_json, getattr(opt, "classes_json", None),
                              transform=tf,
                              search_folder=getattr(opt, "search_folder", None),
                              find_images_by_sub_folder=getattr(opt, "find_images_by_sub_folder", False))
        dataset_type = "json"

    elif getattr(opt, "data", None):
        # Dossier: soit ImageNet CLS-LOC (train/val/test), soit sous-dossiers=classes → ImageFolder

        use_imagenet = False
        for k in ("imagenet_split", "imagenet_ann_dir", "imagenet_imagesets_dir",
                  "imagenet_synset_mapping", "imagenet_val_solution_csv"):
            if getattr(opt, k, None):
                use_imagenet = True
                break

        if use_imagenet:
            split = str(getattr(opt, "imagenet_split", "auto"))
            ann_dir = getattr(opt, "imagenet_ann_dir", None)
            imagesets_dir = getattr(opt, "imagenet_imagesets_dir", None)
            syn_map = getattr(opt, "imagenet_synset_mapping", None)
            val_csv = getattr(opt, "imagenet_val_solution_csv", None)
            label_base = int(getattr(opt, "imagenet_label_base", 1))
            num_classes = int(getattr(opt, "imagenet_num_classes", 1000))
            return_bbox = bool(int(getattr(opt, "imagenet_return_bbox", 0)))

            ds = ImageNetCLSLDataset(
                images_root=opt.data,
                split=split,
                ann_root=ann_dir,
                imagesets_root=imagesets_dir,
                synset_mapping_file=syn_map,
                val_solution_csv=val_csv,
                label_base=label_base,
                num_classes=num_classes,
                return_bbox=return_bbox,
                transform=tf,
            )
            dataset_type = "imagenet"

            # Harmoniser API multitâche : une seule tâche "__DEFAULT__"
            if not hasattr(ds, "task_classes"):
                classes = list(getattr(ds, "classes", []))
                setattr(ds, "task_classes", {"__DEFAULT__": classes})

        else:
            ds = datasets.ImageFolder(root=opt.data, transform=tf)
            dataset_type = "folder"

            # Harmonisation d'API : exposer .task_classes comme MultiTaskDataset
            if not hasattr(ds, "task_classes"):
                try:
                    classes = list(getattr(ds, "classes", []))
                except Exception:
                    classes = []
                setattr(ds, "task_classes", {"__DEFAULT__": classes})

    else:
        # ---------- fallback : via le fichier de config ----------
        dscfg = cfg.get("dataset", {})
        if str(dscfg.get("type", "json")).lower() == "json":
            ds = MultiTaskDataset(dscfg["data_json"], dscfg.get("classes_json", None),
                                  transform=tf,
                                  search_folder=dscfg.get("search_folder", None),
                                  find_images_by_sub_folder=dscfg.get("find_images_by_sub_folder", False))
            dataset_type = "json"
        else:
            # 'data_folder' attendu dans cfg
            data_folder = dscfg.get("data_folder", None)
            if not data_folder:
                raise ValueError("Aucun dataset spécifié (ni --data, ni --data_json, ni dataset.data_folder).")
            ds = datasets.ImageFolder(data_folder, transform=tf)
            dataset_type = "folder"
            if not hasattr(ds, "task_classes"):
                classes = list(getattr(ds, "classes", []))
                setattr(ds, "task_classes", {"__DEFAULT__": classes})

    # batch size: compat 'bs' ou 'batch_size'
    bs = getattr(opt, "bs", None)
    if bs is None:
        bs = getattr(opt, "batch_size", 1)

    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=getattr(opt, "num_workers", 4),
                        drop_last=False)
    return loader, ds, dataset_type



def plot_tsne_interactive(attentive_embeddings_data, labels_data, tasks, img_paths_data, colors=None, num_clusters=None,
                          save_dir='results', metric_scores: dict | None = None):
    """
    Ouvre une interface interactive Tkinter pour explorer un t-SNE calculé sur les attentive embeddings.

    L'interface permet :
      - de choisir une tâche (si plusieurs sont présentes),
      - de recalculer le t-SNE pour la tâche sélectionnée,
      - de zoomer/dézoomer,
      - de tracer un polygone sur le plot pour sélectionner des points,
      - d'afficher la ou les images associées à chaque point sélectionné.

    Args:
        attentive_embeddings_data (dict): Dictionnaire associant chaque tâche à ses attentive embeddings (numpy array de forme (N, C, H, W)).
        labels_data (dict): Dictionnaire associant chaque tâche à ses labels (numpy array).
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        img_paths_data (dict): Dictionnaire associant chaque tâche à la liste des chemins d'images.
        colors (list, optional): Liste de couleurs à utiliser pour le plot.
        num_clusters (int, optional): (Non utilisé ici, mais peut être étendu pour le clustering interactif).
        save_dir (str): Répertoire où sauvegarder d'éventuelles sorties (ex. fichiers JSON des points sélectionnés).
    """
    matplotlib.use('TkAgg')

    # Déterminer si l'on travaille avec un dictionnaire (plusieurs tâches) ou un seul tableau (tâche unique)
    if isinstance(attentive_embeddings_data, dict):
        single_task_mode = (len(attentive_embeddings_data) == 1)
        if single_task_mode:
            current_task_name = list(attentive_embeddings_data.keys())[0]
        else:
            current_task_name = None
    else:
        single_task_mode = True
        current_task_name = None



    tsne_results = None
    labels = None
    class_names = None
    unique_labels = None
    scatter = None
    color_map = None
    img_paths = None
    filename_to_path = None
    polygon = []
    polygon_selector = None
    polygon_cleared = True

    # Création des frames pour l'interface
    root = tk.Tk()
    root.title("Interactive t-SNE with Images")
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky='nsew')
    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky='nsew')

    # Intégration de la figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Zone d'affichage d'image et informations
    img_label = tk.Label(right_frame)
    img_label.pack(pady=10)
    label_text = tk.StringVar()
    label_label = tk.Label(right_frame, textvariable=label_text, justify='left')
    label_label.pack()
    inside_points_label = tk.StringVar()
    inside_points_count_label = tk.Label(right_frame, textvariable=inside_points_label)
    inside_points_count_label.pack()

    dropdown_points = []
    dropdown = ttk.Combobox(right_frame, state="readonly")
    dropdown.pack(fill='x', pady=5)
    dropdown.bind("<<ComboboxSelected>>", lambda event: on_dropdown_select())

    def change_class_color():
        selected = class_selector.get()
        if selected:
            label_str = selected.split(':')[0]
            label_val = int(label_str)
            color_code = colorchooser.askcolor(title="Choisir une couleur")[1]
            if color_code:
                color_map[label_val] = color_code
                scatter.set_color([color_map[int(lbl)] for lbl in labels])
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(lbl)],
                                              markerfacecolor=color_map[int(lbl)], markersize=10) for lbl in
                                   unique_labels])
                canvas.draw()

    class_selector_label = tk.Label(right_frame, text="Sélectionnez une classe :")
    class_selector_label.pack(pady=5)
    class_selector = ttk.Combobox(right_frame, state="readonly")
    class_selector.pack(pady=5)
    change_color_button = tk.Button(right_frame, text="Changer la couleur de la classe", command=change_class_color)
    change_color_button.pack(pady=5)

    button_frame = tk.Frame(right_frame)
    button_frame.pack(pady=10)
    close_button = tk.Button(button_frame, text="Fermer le polygone", command=lambda: analyze_polygon())
    close_button.pack(side='left', padx=5)
    clear_button = tk.Button(button_frame, text="Effacer le polygone", command=lambda: clear_polygon())
    clear_button.pack(side='left', padx=5)

    def clear_polygon():
        nonlocal polygon_selector, polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()
            polygon_selector.set_visible(False)
            del polygon_selector
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()
        fig.canvas.draw()
        inside_points_label.set("")
        label_text.set("")
        img_label.config(image='')
        dropdown.set('')
        dropdown['values'] = []
        polygon_cleared = True

    def update_plot(task_name):
        nonlocal tsne_results, labels, class_names, unique_labels, scatter, color_map, img_paths, filename_to_path, current_task_name
        current_task_name = task_name
        ax.clear()
        # Utiliser les attentive embeddings pour la tâche sélectionnée
        if isinstance(attentive_embeddings_data, dict):
            embeddings = attentive_embeddings_data[task_name]
            labels_local = labels_data[task_name]
            img_paths = img_paths_data[task_name]
            class_names = tasks[task_name]
        else:
            embeddings = attentive_embeddings_data
            labels_local = labels_data
            img_paths = img_paths_data
            class_names = tasks[list(tasks.keys())[0]]
        filename_to_path = {os.path.basename(path): path for path in img_paths}
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
        tsne_results = tsne.fit_transform(embeddings_flat)
        labels = labels_local
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        if colors and len(colors) >= num_classes:
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        else:
            color_palette = plt.cm.get_cmap("tab20", num_classes)
            color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels],
                             picker=True)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(label)],
                                      markerfacecolor=color_map[int(label)], markersize=10) for label in unique_labels]
        ax.legend(handles=legend_elements)
        ax.set_title(f"t-SNE pour la tâche : {task_name}" if task_name else "t-SNE")
        canvas.draw()
        class_selector['values'] = [f"{label}: {class_names[label]}" for label in unique_labels]
        if unique_labels.size > 0:
            class_selector.current(0)
        clear_polygon()

    def on_task_select(event):
        selected_task = task_selector.get()
        update_plot(selected_task)

    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        display_image(img_path, class_names[int(labels[ind])])

    fig.canvas.mpl_connect('pick_event', onpick)

    def enable_polygon_selector(event):
        nonlocal polygon_selector, polygon_cleared
        if event.button == 3:  # clic droit
            if polygon_selector is None or polygon_cleared:
                polygon_selector = PolygonSelector(ax, onselect=onselect, useblit=True)
                polygon_cleared = False
                print("Sélecteur de polygone activé.")

    def onselect(verts):
        polygon.clear()
        polygon.extend(verts)
        print("Sommets du polygone:", verts)

    def analyze_polygon():
        if len(polygon) < 3:
            print("Polygone non fermé. Sélectionnez au moins 3 points.")
            return
        inside_points = []
        outside_points = []
        polygon_path = Path(polygon)
        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
        for point in inside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        for point in outside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        filename_suffix = current_task_name.replace(' ', '_') if current_task_name else 'task'
        with open(os.path.join(save_dir, f"inside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(inside_points, f)
        with open(os.path.join(save_dir, f"outside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(outside_points, f)
        inside_points_label.set(f"Points à l'intérieur du polygone: {len(inside_points)}")
        update_dropdown(inside_points)

    def update_dropdown(inside_points):
        dropdown_values = [f"{point['filename']} ({point['class']})" for point in inside_points]
        dropdown['values'] = dropdown_values
        dropdown_points.clear()
        dropdown_points.extend(inside_points)
        if dropdown_values:
            dropdown.current(0)
            on_dropdown_select()

    def on_dropdown_select():
        selection = dropdown.current()
        if selection >= 0:
            point = dropdown_points[selection]
            img_path = filename_to_path[point['filename']]
            display_image(img_path, point['class'])

    def display_image(img_path, label):
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {label}\nFichier: {os.path.basename(img_path)}")

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    # ------------------------------------------------------------------
    # 0) Variable globale supplémentaire
    # ------------------------------------------------------------------
    last_click = {'pos': None, 'marker': None}  # on stocke aussi le handle

    # ------------------------------------------------------------------
    # 1) Gestion du clic gauche
    # ------------------------------------------------------------------
    def on_mouse_click(event):
        if event.button == 1 and event.inaxes is not None and event.xdata is not None:
            # Effacer l’ancien marqueur s’il existe
            if last_click['marker'] is not None:
                last_click['marker'].remove()

            # Mémoriser la nouvelle position
            last_click['pos'] = (event.xdata, event.ydata)

            # Dessiner la nouvelle croix et conserver son handle
            last_click['marker'] = ax.scatter(*last_click['pos'],
                                              marker='x', c='k', s=30, zorder=3)
            canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # ------------------------------------------------------------
    # 1) Remplacer complètement la fonction zoom
    # ------------------------------------------------------------
    def zoom(scale: float):
        """scale >1 : zoom avant ; scale <1 : zoom arrière, centré sur last_click"""
        if scale <= 0:
            return

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Point de référence : dernier clic ou centre courant
        if last_click['pos'] and None not in last_click['pos']:
            cx, cy = last_click['pos']
        else:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        new_w = (x1 - x0) / scale
        new_h = (y1 - y0) / scale

        ax.set_xlim(cx - new_w / 2, cx + new_w / 2)
        ax.set_ylim(cy - new_h / 2, cy + new_h / 2)
        canvas.draw_idle()

    # ------------------------------------------------------------
    # 2) Adapter la molette
    # ------------------------------------------------------------
    # Molette : on ignore désormais event.xdata / ydata
    def on_scroll(event):
        direction = getattr(event, "step", 1 if event.button == "up" else -1)
        base = 1.2
        scale = base if direction > 0 else 1 / base
        zoom(scale)

    # Clavier
    def on_key_press(event):
        base = 1.2
        if event.key in ['+', '=']:
            zoom(base)
        elif event.key == '-':
            zoom(1 / base)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key_press)


    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    if not single_task_mode:
        task_selector_label = tk.Label(right_frame, text="Sélectionnez une tâche :")
        task_selector_label.pack(pady=5)
        task_selector = ttk.Combobox(right_frame, state="readonly", values=list(tasks.keys()))
        # Attribuer la liste **après** la création
        task_names = list(tasks.keys())
        task_selector['values'] = task_names

        # Fixe l’élément affiché : préférez `.current(0)` à `.set(...)`
        if task_names:
            task_selector.current(0)  # sélectionne la première tâche visuellement

        task_selector.pack(pady=5)
        task_selector.bind("<<ComboboxSelected>>", on_task_select)

    if single_task_mode:
        update_plot(list(tasks.keys())[0])
    else:
        initial_task = list(tasks.keys())[0]
        task_selector.set(initial_task)
        update_plot(initial_task)

    # ------------------------------------------------------------------
    #  Affichage optionnel des scores ET du sélecteur de tâche virtuel
    # ------------------------------------------------------------------


    # bloc déjà présent pour les scores :
    if metric_scores:
        score_frame = tk.Frame(right_frame)
        score_frame.pack(pady=10, fill='x')
        tk.Label(score_frame, text="Scores métriques :", font=("TkDefaultFont", 10, "bold")
                 ).pack(anchor='w')
        for k, v in metric_scores.items():
            tk.Label(score_frame, text=f"{k:15s}: {v:.4f}").pack(anchor='w')

    root.mainloop()


# ======================== compute_metrics.py ==============================

from typing import Dict, List, Tuple
import numpy as np

def compute_metrics(embs      : np.ndarray,
                    labels    : np.ndarray,
                    metrics   : List[str],
                    *,
                    pca_dim   : int | None = None,
                    l2_norm   : bool       = False,
                    knn_k     : int        = 5,
                    metric    : str        = "auto"  # "auto" → cosine si l2_norm sinon euclidean
                    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Renvoie :
        embs_proc  : embeddings après PCA / L2 éventuels
        scores     : dict métriques -> valeur(s)
        groupings  : dict name -> np.ndarray[N] (ex: knn_pred, cluster_id, margin, cluster_mknn, cluster_spectral, cluster_dbscan)

    k-NN (corrigé) :
      - Leave-One-Out robuste (retrait de soi même en cas de duplicats)
      - Bris d’égalité aléatoire
      - Diagnostics same_frac / entropy / marges (d_diff - d_same)

    Clustering :
      - KMeans purity corrigée
      - Mutual-kNN + Chinese Whispers : "cluster_mknn"
      - Spectral auto (affinité self-tuning + eigengap) : "cluster_spectral"
      - DBSCAN auto (ε coude k-distance) : "cluster_dbscan"
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import (
        silhouette_score, pairwise_distances,
        davies_bouldin_score, calinski_harabasz_score
    )
    from sklearn.cluster import KMeans, DBSCAN
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(42)

    X = embs.copy()
    y_orig = labels.copy()
    classes, y = np.unique(y_orig, return_inverse=True)  # 0..C-1

    # 1) PCA
    if pca_dim is not None:
        print(f"[Info] PCA → {pca_dim} dims")
        X = PCA(n_components=pca_dim, random_state=42).fit_transform(X)

    # 2) L2
    if l2_norm:
        print("[Info] L2-normalisation")
        X = normalize(X, norm="l2", axis=1)

    # 3) métrique
    if metric == "auto":
        kn_metric = "cosine" if l2_norm else "euclidean"
    else:
        kn_metric = metric

    scores:    Dict[str, float]      = {}
    groupings: Dict[str, np.ndarray] = {}

    N = X.shape[0]
    C = len(classes)

    # -------- helpers voisins ----------
    def _neighbors_matrix(X, k, metric):
        k_cand = int(min(N, max(k + 50, k + 1)))  # marge pr duplicats
        nn = NearestNeighbors(n_neighbors=k_cand, metric=metric).fit(X)
        d, I = nn.kneighbors(X, return_distance=True)
        return d, I

    def _drop_self_per_row(idx):
        out = []
        for i in range(N):
            row = idx[i]
            out.append(row[row != i])
        max_len = max((len(r) for r in out), default=0)
        if max_len == 0:
            return np.empty((N,0), dtype=int)
        pad = -np.ones((N, max_len), dtype=int)
        for i, r in enumerate(out):
            pad[i, :len(r)] = r
        return pad

    # ================= k-NN (LOO + diag) =================
    if "knn" in metrics and N >= 2:
        k = int(max(1, min(knn_k, N - 1)))
        _, I = _neighbors_matrix(X, k, kn_metric)
        idx = _drop_self_per_row(I)

        # vote LOO
        knn_preds = np.empty(N, dtype=int)
        for i in range(N):
            nei = idx[i]
            nei = nei[nei >= 0][:k]
            if len(nei) == 0:
                counts = np.bincount(y, minlength=C)
            else:
                counts = np.bincount(y[nei], minlength=C)
            ties = np.flatnonzero(counts == counts.max())
            knn_preds[i] = rng.choice(ties)
        acc = float((knn_preds == y).mean())

        cnts = np.bincount(y, minlength=C)
        baseline = cnts.max() / float(N)
        norm_acc = (acc - baseline) / (1 - baseline + 1e-12)

        scores[f"knn_acc@{k}"]      = acc
        scores[f"knn_acc@{k}_norm"] = float(norm_acc)
        groupings["knn_pred"]       = classes[knn_preds]  # mapping inverse

        # diagnostics
        nnb = int(min(N - 1, max(k + 10, 20)))
        _, Iw = _neighbors_matrix(X, nnb, kn_metric)
        idx_w = _drop_self_per_row(Iw)

        same_list, ent_list = [], []
        d_same = np.full(N, np.nan, dtype=np.float32)
        d_diff = np.full(N, np.nan, dtype=np.float32)
        for i in range(N):
            nei = idx_w[i]; nei = nei[nei >= 0]
            if len(nei) == 0:
                same_list.append(0.0); ent_list.append(0.0)
                continue
            nei_k = nei[:k]
            labs  = y[nei_k]
            same_list.append(float((labs == y[i]).mean()))
            cts = np.bincount(labs, minlength=C)
            p = cts[cts > 0].astype(float); p /= p.sum()
            ent_list.append(float(-(p * np.log(p + 1e-12)).sum()))

            # marge via distances locales
            labs_w = y[nei]
            dist_loc = pairwise_distances(X[i:i+1], X[nei], metric=kn_metric).ravel()
            sm = (labs_w == y[i]); df = ~sm
            if np.any(sm):
                d_same[i] = float(dist_loc[np.where(sm)[0][0]])
            if np.any(df):
                d_diff[i] = float(dist_loc[np.where(df)[0][0]])

        scores[f"knn_same_frac@{k}"] = float(np.mean(same_list))
        scores[f"knn_entropy@{k}"]   = float(np.mean(ent_list))
        margin = d_diff - d_same
        groupings["d_same"] = d_same
        groupings["d_diff"] = d_diff
        groupings["margin"] = margin
        val = np.isfinite(margin)
        if val.any():
            m = margin[val]
            scores["knn_overlap_rate"] = float((m < 0).mean())
            scores["knn_margin_mean"]  = float(np.mean(m))
            q25, q50, q75 = np.percentile(m, [25, 50, 75])
            scores["knn_margin_p25"]   = float(q25)
            scores["knn_margin_p50"]   = float(q50)
            scores["knn_margin_p75"]   = float(q75)

    # =============== Mutual-kNN + Chinese Whispers =======================
    if "cluster_mknn" in metrics and N >= 2:
        k = int(max(1, min(knn_k, N - 1)))
        _, I = _neighbors_matrix(X, k, kn_metric)
        idx = _drop_self_per_row(I)
        sets = [set(r[r>=0][:k].tolist()) for r in idx]
        # arêtes mutuelles
        rows, cols = [], []
        for i in range(N):
            for j in sets[i]:
                if i in sets[j]:
                    rows.append(i); cols.append(j)
        data = np.ones(len(rows), dtype=np.uint8)
        A = coo_matrix((data, (rows, cols)), shape=(N, N))
        A = ((A + A.T) > 0).astype(np.uint8)

        # Chinese Whispers
        adj = [list(A.getrow(i).indices) for i in range(N)]
        labels_cw = np.arange(N, dtype=int)
        for _ in range(25):
            for u in rng.permutation(N):
                if not adj[u]:
                    continue
                # vote par majorité des labels voisins
                labs = labels_cw[adj[u]]
                cts = np.bincount(labs)
                ties = np.flatnonzero(cts == cts.max())
                labels_cw[u] = rng.choice(ties)
        uniq, remap = np.unique(labels_cw, return_inverse=True)
        cw = remap
        groupings["cluster_mknn"] = cw

        # purity & silhouette
        total = 0
        for cid in np.unique(cw):
            yy = y[cw == cid]
            if len(yy) > 0:
                total += np.bincount(yy, minlength=C).max()
        scores["cluster_mknn_purity"] = total / N
        try:
            scores["cluster_mknn_sil"] = float(silhouette_score(X, cw, metric=kn_metric))
        except Exception:
            pass

    # ================== Spectral auto (self-tuning + eigengap) ===========
    if "cluster_spectral" in metrics and N >= 2:
        k_nbrs = int(min(max(10, knn_k + 5), max(2, N-1)))
        nn = NearestNeighbors(n_neighbors=min(N, k_nbrs+1), metric=kn_metric).fit(X)
        D, I = nn.kneighbors(X, return_distance=True)
        D, I = D[:,1:], I[:,1:]
        sigma = D[:, -1] + 1e-12
        rows, cols, vals = [], [], []
        for i in range(N):
            for r, j in enumerate(I[i]):
                w = np.exp(-(D[i, r]**2) / (sigma[i]*sigma[j]))
                rows.append(i); cols.append(j); vals.append(w)
        W = np.zeros((N,N), dtype=np.float32)
        W[rows, cols] = vals
        W = np.maximum(W, W.T)
        d = np.clip(W.sum(1), 1e-12, None)
        Dm12 = np.diag(1.0 / np.sqrt(d))
        L = np.eye(N) - Dm12 @ W @ Dm12

        eigvals, eigvecs = np.linalg.eigh(L)
        Kmax = min(12, max(2, N-1))
        gaps = np.diff(eigvals[:Kmax+1])
        k_opt = int(np.argmax(gaps[:Kmax-1]) + 1)
        k_opt = max(2, min(k_opt, Kmax))

        U = eigvecs[:, :k_opt]
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=k_opt, n_init=20, random_state=42).fit(U)
        sp = km.labels_
        groupings["cluster_spectral"] = sp
        scores["spectral_K"] = float(k_opt)
        scores["spectral_eigengap"] = float(np.max(gaps[:Kmax-1]))
        # purity + silhouette
        total = 0
        for cid in np.unique(sp):
            yy = y[sp == cid]
            if len(yy) > 0:
                total += np.bincount(yy, minlength=C).max()
        scores["cluster_spectral_purity"] = total / N
        try:
            scores["cluster_spectral_sil"] = float(silhouette_score(X, sp, metric=kn_metric))
        except Exception:
            pass

    # ================== DBSCAN auto (ε coude) ============================
    if "cluster_dbscan" in metrics and N >= 2:
        min_samples = max(5, knn_k)
        nn = NearestNeighbors(n_neighbors=min(N, min_samples), metric=kn_metric).fit(X)
        D, _ = nn.kneighbors(X, return_distance=True)
        kth = np.sort(D[:, -1])
        d1 = np.gradient(kth); d2 = np.gradient(d1)
        i = int(np.argmax(d2))
        eps = float(kth[i] + 1e-12)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=kn_metric).fit_predict(X)
        groupings["cluster_dbscan"] = db
        scores["dbscan_eps"] = eps
        # purity/silhouette (ignore bruit -1)
        mask = db != -1
        if mask.any():
            total = 0
            for cid in np.unique(db[mask]):
                yy = y[(db == cid)]
                total += np.bincount(yy, minlength=C).max()
            scores["cluster_dbscan_purity"] = total / mask.sum()
            try:
                if len(np.unique(db[mask])) >= 2:
                    scores["cluster_dbscan_sil"] = float(
                        silhouette_score(X[mask], db[mask], metric=kn_metric)
                    )
            except Exception:
                pass

    # ================== Protos / Séparabilité ============================
    if "proto" in metrics and N >= 2 and C >= 2:
        protos = np.stack([X[y == c].mean(0) for c in range(C)])
        dmat   = pairwise_distances(protos, metric=kn_metric)
        inter_vals = dmat[np.triu_indices(C, 1)]
        scores["proto/inter_min"]    = float(np.min(inter_vals))
        scores["proto/inter_mean"]   = float(np.mean(inter_vals))
        scores["proto/inter_median"] = float(np.median(inter_vals))

        intra = []
        for c in range(C):
            Xi = X[y == c]
            if len(Xi) > 0:
                intra.append(pairwise_distances(Xi, protos[c:c+1], metric=kn_metric).mean())
        scores["proto/intra_mean"] = float(np.mean(intra)) if intra else 0.0
        if intra:
            scores["fisher_like"] = float(scores["proto/inter_mean"] /
                                          (scores["proto/intra_mean"] + 1e-12))

    if "sep" in metrics and N >= 2 and C >= 2:
        try:
            scores["silhouette"] = float(silhouette_score(X, y, metric=kn_metric))
        except Exception:
            pass

    # ================== KMeans + purity corrigée ==========================
    if "cluster" in metrics and C >= 2:
        k = C
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        preds = kmeans.labels_
        total = 0
        for cid in range(k):
            yy = y[preds == cid]
            total += np.bincount(yy, minlength=C).max() if len(yy) > 0 else 0
        scores["cluster_purity"] = total / N
        groupings["cluster_id"]  = preds

    # =============== retrieval@1 =========================================
    if "retrieval" in metrics and N >= 2:
        dmat = pairwise_distances(X, metric=kn_metric)
        np.fill_diagonal(dmat, np.inf)
        top1 = dmat.argmin(axis=1)
        scores["retrieval@1"] = float((y[top1] == y).mean())

    # =============== indices globaux =====================================
    if "indices" in metrics and N >= 2 and C >= 2:
        try:
            scores["db_index"] = float(davies_bouldin_score(X, y))
        except Exception:
            pass
        try:
            scores["ch_index"] = float(calinski_harabasz_score(X, y))
        except Exception:
            pass

    return X, scores, groupings


# ------------------------------------------------------------
#  t-SNE interactif  –  label / cluster / k-NN  + regroupement + zoom
#          + spinbox pour n_neighbors  /  n_clusters
# ------------------------------------------------------------
# ======================== plot_tsne_cluster_knn.py =======================

def plot_tsne_cluster_knn(
        embs_d: dict[str, np.ndarray],
        labels_d: dict[str, np.ndarray],
        class_names_d: dict[str, list[str]],
        paths_d: dict[str, np.ndarray],
        *,
        cluster_id: np.ndarray | None = None,
        knn_pred  : np.ndarray | None = None,
        save_dir  : str = "results",
        metric_scores: dict | None = None
):
    """
    UI t-SNE avec vues : label / cluster / knn_pred + méthodes de clustering
    plus robustes et bouton "Distances inter/intra" sur la vue courante.

    Remarques :
      - k-NN LOO par tâche (pas de mélange inter-domaine).
      - Clustering par tâche, méthodes : KMeans / Mutual-kNN / Spectral / DBSCAN.
      - Le bouton "Distances inter/intra" calcule :
          * Distances inter-centroïdes (min/mean/median)
          * Distance intra moyenne (moyenne point->centroïde)
    """
    import os, tkinter as tk, numpy as np
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
    import matplotlib, matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances, silhouette_score
    matplotlib.use("TkAgg")

    # ---------- config métrique (cosine conseillé si L2 déjà fait) -------
    KN_METRIC = "cosine"  # "euclidean" possible selon vos embeddings

    # ---------- concat global --------------------------------------------
    tasks     = list(embs_d.keys())
    embs_cat  = np.concatenate([embs_d[t]  for t in tasks])
    paths_cat = np.concatenate([paths_d[t] for t in tasks])
    labels_cat= np.concatenate([labels_d[t] for t in tasks])

    # spans par tâche
    spans, s = {}, 0
    for t in tasks:
        e = len(embs_d[t]); spans[t] = (s, s+e); s += e

    # ========== états courants (par tâche) ================================
    # on stocke des vecteurs globaux mais on réécrit seulement les tranches
    cur_cluster_id = cluster_id.copy() if cluster_id is not None else np.full(len(embs_cat), -1, dtype=int)
    cur_knn_pred   = knn_pred.copy()   if knn_pred   is not None else np.full(len(embs_cat), -1, dtype=int)

    # ---------- UI base ---------------------------------------------------
    root = tk.Tk(); root.title("t-SNE : label / cluster / k-NN (robuste)")
    L  = tk.Frame(root); L.grid(row=0, column=0, sticky="nsew")
    R  = tk.Frame(root); R.grid(row=0, column=1, sticky="nsew")
    root.grid_columnconfigure(0, weight=3); root.grid_rowconfigure(0, weight=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    canv = FigureCanvasTkAgg(fig, master=L); canv.draw()
    canv.get_tk_widget().pack(fill="both", expand=True)

    # ---------- contrôles -------------------------------------------------
    tk.Label(R, text="Tâche :").pack(pady=2)
    cb_task = ttk.Combobox(R, state="readonly", values=tasks); cb_task.current(0); cb_task.pack(fill="x")

    def available_views():
        v = ["label"]
        if np.any(cur_cluster_id >= 0): v.append("cluster_id")
        if np.any(cur_knn_pred   >= 0): v.append("knn_pred")
        return v

    tk.Label(R, text="Vue :").pack(pady=2)
    cb_view = ttk.Combobox(R, state="readonly", values=available_views()); cb_view.current(0)
    cb_view.pack(fill="x")

    # options regroupement
    regroup_var = tk.BooleanVar(master=root, value=False)
    tk.Checkbutton(R, text="Regrouper (moyenne par groupe)", variable=regroup_var, command=lambda: draw()).pack(pady=4)

    tk.Label(R, text="Selon :").pack(pady=(4,2))
    cb_grp = ttk.Combobox(R, state="readonly", values=["none","cluster_id","knn_pred"]); cb_grp.current(0); cb_grp.pack(fill="x")

    # spinboxes & méthode de clustering
    spin_frames = tk.Frame(R); spin_frames.pack(pady=(8,4), fill="x")

    tk.Label(spin_frames, text="k (k-NN):").grid(row=0, column=0, sticky="w")
    kn_var = tk.IntVar(value=5)
    tk.Spinbox(spin_frames, from_=1, to=50, textvariable=kn_var, width=5, command=lambda: recompute_knn()).grid(row=0, column=1, sticky="w")

    tk.Label(spin_frames, text="K (clusters):").grid(row=1, column=0, sticky="w")
    km_var = tk.IntVar(value=8)
    tk.Spinbox(spin_frames, from_=2, to=100, textvariable=km_var, width=5, command=lambda: recompute_cluster()).grid(row=1, column=1, sticky="w")

    tk.Label(R, text="Méthode clustering :").pack(pady=(6,2))
    cb_method = ttk.Combobox(R, state="readonly", values=["KMeans", "Mutual-kNN", "Spectral", "DBSCAN"]); cb_method.current(0)
    cb_method.pack(fill="x")

    # bouton distances inter/intra
    tk.Button(R, text="Distances inter/intra (vue courante)", command=lambda: show_inter_intra()).pack(pady=(8,4), fill="x")

    # widgets image & diag
    img_lbl = tk.Label(R); img_lbl.pack(pady=10)
    info    = tk.StringVar(); tk.Label(R, textvariable=info, justify="left").pack()

    diag_frame = tk.Frame(R); diag_frame.pack(pady=8, fill="x")
    tk.Label(diag_frame, text="Diagnostics k-NN (labels réels)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    diag_text = tk.StringVar()
    tk.Label(diag_frame, textvariable=diag_text, justify="left").pack(anchor="w")

    # --------------- Helpers de calcul -----------------------------------

    def _loo_knn_pred(Xt, yt, k):
        """Prédictions LOO pour un sous-ensemble Xt."""
        n = len(Xt)
        if n < 2:
            return np.full(n, -1, dtype=int)
        k = int(max(1, min(k, n-1)))
        nn = NearestNeighbors(n_neighbors=min(n, k+50), metric=KN_METRIC).fit(Xt)
        _, I = nn.kneighbors(Xt, return_distance=True)
        # drop self
        idx = []
        for i in range(n):
            row = I[i]
            idx.append(row[row != i])
        maxlen = max((len(r) for r in idx), default=0)
        pad = -np.ones((n, maxlen), dtype=int)
        for i, r in enumerate(idx):
            pad[i, :len(r)] = r
        rng = np.random.default_rng(42)
        preds = np.empty(n, dtype=int)
        for i in range(n):
            nei = pad[i]; nei = nei[nei>=0][:k]
            if len(nei)==0:
                counts = np.bincount(yt)
            else:
                counts = np.bincount(yt[nei], minlength=yt.max()+1)
            ties = np.flatnonzero(counts == counts.max())
            preds[i] = int(rng.choice(ties))
        return preds

    def _cluster_task(task, method):
        """Clusterise la tranche de la tâche courante selon method, met à jour cur_cluster_id."""
        nonlocal cur_cluster_id
        lo, hi = spans[task]
        Xt = embs_cat[lo:hi]
        K  = int(km_var.get())
        k  = int(kn_var.get())
        if len(Xt) < 2:
            return
        if method == "KMeans":
            lab = KMeans(n_clusters=min(max(2, K), len(Xt)-1), n_init=50, random_state=42).fit_predict(Xt)
        elif method == "Mutual-kNN":
            # mutual kNN + CW (léger)
            nn = NearestNeighbors(n_neighbors=min(len(Xt), k+1), metric=KN_METRIC).fit(Xt)
            _, I = nn.kneighbors(Xt, return_distance=True)
            I = I[:,1:]  # drop self immédiat
            sets = [set(row[:min(k, len(row))].tolist()) for row in I]
            rows, cols = [], []
            for i in range(len(Xt)):
                for j in sets[i]:
                    if i in sets[j]:
                        rows.append(i); cols.append(j)
            data = np.ones(len(rows), dtype=np.uint8)
            from scipy.sparse import coo_matrix
            A = coo_matrix((data, (rows, cols)), shape=(len(Xt), len(Xt)))
            A = ((A + A.T) > 0).astype(np.uint8)
            adj = [list(A.getrow(i).indices) for i in range(len(Xt))]
            rng = np.random.default_rng(42)
            labels_cw = np.arange(len(Xt), dtype=int)
            for _ in range(25):
                for u in rng.permutation(len(Xt)):
                    if not adj[u]:
                        continue
                    labs = labels_cw[adj[u]]
                    cts = np.bincount(labs)
                    ties = np.flatnonzero(cts == cts.max())
                    labels_cw[u] = int(rng.choice(ties))
            _, lab = np.unique(labels_cw, return_inverse=True)
        elif method == "Spectral":
            # Self-tuning + eigengap
            k_nbrs = min(max(10, k+5), max(2, len(Xt)-1))
            nn = NearestNeighbors(n_neighbors=min(len(Xt), k_nbrs+1), metric=KN_METRIC).fit(Xt)
            D, I = nn.kneighbors(Xt, return_distance=True)
            D, I = D[:,1:], I[:,1:]
            sigma = D[:, -1] + 1e-12
            rows, cols, vals = [], [], []
            for i in range(len(Xt)):
                for r, j in enumerate(I[i]):
                    w = np.exp(-(D[i, r]**2) / (sigma[i]*sigma[j]))
                    rows.append(i); cols.append(j); vals.append(w)
            W = np.zeros((len(Xt), len(Xt)), dtype=np.float32)
            W[rows, cols] = vals
            W = np.maximum(W, W.T)
            d = np.clip(W.sum(1), 1e-12, None)
            Dm12 = np.diag(1.0 / np.sqrt(d))
            L = np.eye(len(Xt)) - Dm12 @ W @ Dm12
            eigvals, eigvecs = np.linalg.eigh(L)
            Kmax = min(12, max(2, len(Xt)-1))
            gaps = np.diff(eigvals[:Kmax+1])
            k_opt = int(np.argmax(gaps[:Kmax-1]) + 1)
            k_opt = max(2, min(k_opt, Kmax))
            U = eigvecs[:, :k_opt]
            U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            lab = KMeans(n_clusters=k_opt, n_init=20, random_state=42).fit_predict(U)
        else:  # DBSCAN
            min_samples = max(5, k)
            nn = NearestNeighbors(n_neighbors=min(len(Xt), min_samples), metric=KN_METRIC).fit(Xt)
            D, _ = nn.kneighbors(Xt, return_distance=True)
            kth = np.sort(D[:, -1])
            d1 = np.gradient(kth); d2 = np.gradient(d1)
            i = int(np.argmax(d2))
            eps = float(kth[i] + 1e-12)
            lab = DBSCAN(eps=eps, min_samples=min_samples, metric=KN_METRIC).fit_predict(Xt)
        cur_cluster_id[lo:hi] = lab

    def recompute_knn():
        """k-NN LOO par tâche (pas de resubstitution)."""
        nonlocal cur_knn_pred
        k = int(kn_var.get())
        task = cb_task.get()
        lo, hi = spans[task]
        Xt = embs_cat[lo:hi]
        yt = labels_d[task].astype(int)
        preds = _loo_knn_pred(Xt, yt, k)
        if preds.size:
            cur_knn_pred[lo:hi] = preds
        update_view_boxes(); draw()

    def recompute_cluster():
        task = cb_task.get()
        _cluster_task(task, cb_method.get())
        update_view_boxes(); draw()

    # ---------- update combobox values -----------------------------------
    def update_view_boxes():
        cb_view['values'] = available_views()
        if cb_view.get() not in cb_view['values']:
            cb_view.current(0)
        # grp selon
        gv = ["none"]
        if np.any(cur_cluster_id >= 0): gv.append("cluster_id")
        if np.any(cur_knn_pred   >= 0): gv.append("knn_pred")
        cb_grp['values'] = gv
        if cb_grp.get() not in gv:
            cb_grp.current(0)

    # ---------- état courant & dessin ------------------------------------
    cur_embs = cur_lbls = cur_paths = cur_names = None
    scatter = tsne_xy = None

    def _compute_knn_diagnostics(X, y, k):
        """Diagnostics en LOO sur labels réels (pas de resubstitution)."""
        out = {}
        n = len(X)
        if n < 2: return out
        k = int(max(1, min(k, n-1)))
        preds = _loo_knn_pred(X, y, k)
        acc = (preds == y).mean()
        vals, cnts = np.unique(y, return_counts=True)
        baseline = cnts.max() / float(n)
        norm = (acc - baseline) / (1 - baseline + 1e-12)
        out["acc"] = float(acc); out["norm"] = float(norm)

        # voisinage élargi pour marges
        nnb = min(n-1, max(k+10, 20))
        nn = NearestNeighbors(n_neighbors=nnb+1, metric=KN_METRIC).fit(X)
        dists, idx = nn.kneighbors(X, return_distance=True)
        dists, idx = dists[:,1:], idx[:,1:]
        neigh_labels = y[idx]
        same_frac = (neigh_labels[:, :k] == y[:, None]).mean(axis=1).mean()
        out["same_frac"] = float(same_frac)
        ent = []
        for i in range(n):
            labs, cts = np.unique(neigh_labels[i, :k], return_counts=True)
            p = cts / cts.sum()
            ent.append(float(-(p * np.log(p + 1e-12)).sum()))
        out["entropy"] = float(np.mean(ent))

        # marges
        d_same = np.full((n,), np.nan, dtype=np.float32)
        d_diff = np.full((n,), np.nan, dtype=np.float32)
        for i in range(n):
            li = y[i]
            same_mask = (neigh_labels[i] == li)
            diff_mask = ~same_mask
            if np.any(same_mask): d_same[i] = dists[i][np.where(same_mask)[0][0]]
            if np.any(diff_mask): d_diff[i] = dists[i][np.where(diff_mask)[0][0]]
        margin = d_diff - d_same
        m = margin[np.isfinite(margin)]
        if m.size:
            out["overlap"] = float((m < 0).mean())
            out["m_mean"]  = float(np.mean(m))
            q25, q50, q75  = np.percentile(m, [25,50,75])
            out["m_p25"], out["m_p50"], out["m_p75"] = float(q25), float(q50), float(q75)
        else:
            out["overlap"] = np.nan; out["m_mean"] = np.nan
            out["m_p25"] = out["m_p50"] = out["m_p75"] = np.nan
        return out

    def _compute_inter_intra(X, labels_view):
        """Renvoie dict avec stats inter (centroïdes) et intra (moy dist -> centroïde)."""
        ids = np.unique(labels_view)
        ids = ids[ids != -1]  # ignore bruit si DBSCAN
        if ids.size < 1:
            return None
        centroids = []
        counts = []
        for cid in ids:
            Xi = X[labels_view == cid]
            centroids.append(Xi.mean(0))
            counts.append(len(Xi))
        C = np.vstack(centroids)
        d_inter = pairwise_distances(C, metric=KN_METRIC)
        if len(ids) >= 2:
            inter_vals = d_inter[np.triu_indices(len(ids), 1)]
            inter_stats = dict(min=float(np.min(inter_vals)),
                               mean=float(np.mean(inter_vals)),
                               median=float(np.median(inter_vals)))
        else:
            inter_stats = dict(min=np.nan, mean=np.nan, median=np.nan)
        # intra
        intra_list = []
        for cid, mu in zip(ids, C):
            Xi = X[labels_view == cid]
            intra_list.append(pairwise_distances(Xi, mu[None, :], metric=KN_METRIC).mean())
        intra_mean = float(np.mean(intra_list)) if len(intra_list) else np.nan
        return {
            "n_clusters": int(len(ids)),
            "cluster_ids": ids.tolist(),
            "sizes": [int(c) for c in counts],
            "inter": inter_stats,
            "intra_mean": intra_mean
        }

    def _update_diag_panel(X, y, k):
        stats = _compute_knn_diagnostics(X, y, k)
        if not stats:
            diag_text.set("Pas assez de points."); return
        txt = []
        txt.append(f"Acc@{k}: {stats.get('acc', np.nan):.4f}   (norm.: {stats.get('norm', np.nan):.4f})")
        txt.append(f"Same-class@{k}: {stats.get('same_frac', np.nan):.4f}   Entropie: {stats.get('entropy', np.nan):.4f}")
        txt.append(f"Overlap rate (marge<0): {100*stats.get('overlap', np.nan):.1f}%")
        txt.append(f"Marge  mean/p25/p50/p75: {stats.get('m_mean', np.nan):.4f} | "
                   f"{stats.get('m_p25', np.nan):.4f} / {stats.get('m_p50', np.nan):.4f} / {stats.get('m_p75', np.nan):.4f}")
        diag_text.set("\n".join(txt))

    def show_inter_intra():
        """Bouton : calcule inter/intra pour la vue courante et affiche."""
        task = cb_task.get(); lo, hi = spans[task]
        view = cb_view.get()
        Xv = embs_cat[lo:hi]
        if view == "label":
            lv = labels_d[task].astype(int)
            title = f"[{task}] Distances par labels réels"
        elif view == "cluster_id":
            lv = cur_cluster_id[lo:hi]
            title = f"[{task}] Distances par cluster ({cb_method.get()})"
        else:
            lv = cur_knn_pred[lo:hi]
            title = f"[{task}] Distances par knn_pred"
        res = _compute_inter_intra(Xv, lv)
        if res is None:
            messagebox.showinfo("Distances", "Aucun cluster valide (ou tous -1)."); return
        # fenêtre popup
        win = tk.Toplevel(root); win.title("Distances inter/intra")
        txt = tk.Text(win, width=60, height=18)
        txt.pack(fill="both", expand=True)
        txt.insert("end", title + "\n\n")
        txt.insert("end", f"#clusters: {res['n_clusters']}\n")
        txt.insert("end", f"IDs: {res['cluster_ids']}\n")
        txt.insert("end", f"Tailles: {res['sizes']}\n\n")
        inter = res['inter']
        txt.insert("end", f"Inter-centroïdes  min/mean/median: "
                          f"{inter['min']:.6f} / {inter['mean']:.6f} / {inter['median']:.6f}\n")
        txt.insert("end", f"Intra (moy. point→centroïde): {res['intra_mean']:.6f}\n")
        txt.config(state="disabled")

    def draw(*_):
        nonlocal cur_embs, cur_lbls, cur_paths, cur_names, scatter, tsne_xy
        task = cb_task.get(); view = cb_view.get()
        lo, hi = spans[task]
        cur_embs  = embs_cat [lo:hi]
        cur_paths = paths_cat[lo:hi]

        if view == "label":
            cur_lbls  = labels_d[task].astype(int)
            cur_names = class_names_d[task]
        elif view == "cluster_id":
            if np.all(cur_cluster_id[lo:hi] < 0):
                _cluster_task(task, cb_method.get())
            cur_lbls  = cur_cluster_id[lo:hi]
            uniq_ids  = np.unique(cur_lbls[cur_lbls>=0])
            cur_names = [f"cluster {i}" for i in uniq_ids] if uniq_ids.size else ["(vide)"]
        else:
            if np.all(cur_knn_pred[lo:hi] < 0):
                recompute_knn()
            cur_lbls  = cur_knn_pred[lo:hi]
            uniq_ids  = np.unique(cur_lbls[cur_lbls>=0])
            cur_names = [f"pred {i}" for i in uniq_ids] if uniq_ids.size else ["(vide)"]

        # regroupement optionnel (moyenne par groupe)
        embs_ = cur_embs.copy()
        if regroup_var.get() and cb_grp.get() != "none":
            ids_global = {"cluster_id": cur_cluster_id,
                          "knn_pred"  : cur_knn_pred}[cb_grp.get()][lo:hi]
            valids = ids_global != -1
            if np.any(valids):
                for uid in np.unique(ids_global[valids]):
                    m = (ids_global == uid)
                    embs_[m] = cur_embs[m].mean(0) + 1e-4*np.random.randn(m.sum(), cur_embs.shape[1])

        tsne_xy = TSNE(n_components=2, init="random", random_state=42).fit_transform(embs_)
        ax.clear()
        uniq = np.unique(cur_lbls)
        cmap = plt.cm.get_cmap("tab20", len(uniq) if len(uniq)>0 else 1)
        colors = [cmap(int(u) % max(len(uniq),1) / max(len(uniq),1)) for u in cur_lbls]
        scatter = ax.scatter(tsne_xy[:,0], tsne_xy[:,1], c=colors, s=22, picker=True)
        if len(uniq) <= 20 and len(uniq)>0:
            legend_elems = [plt.Line2D([0],[0], marker="o", lw=0,
                                       markerfacecolor=cmap(i/max(1,len(uniq))), markersize=6)
                            for i in range(len(uniq))]
            if cb_view.get()=="label":
                legend_names = [class_names_d[task][int(u)] for u in uniq]
            else:
                legend_names = [f"{int(u)}" for u in uniq]
            ax.legend(legend_elems, legend_names, loc="upper right", fontsize=8)
        ax.set_title(f"{task} – {view} | regroupé {regroup_var.get()}")
        canv.draw_idle()

        # diagnostics k-NN (toujours sur labels réels de la tâche)
        k = int(kn_var.get())
        _update_diag_panel(cur_embs, labels_d[task].astype(int), k)

    cb_task.bind("<<ComboboxSelected>>", draw)
    cb_view.bind("<<ComboboxSelected>>", draw)

    # ---------- interactions image ---------------------------------------
    def on_pick(ev):
        i = ev.ind[0]
        task = cb_task.get(); lo, hi = spans[task]
        img = Image.open(cur_paths[i]).resize((400,400))
        img_tk = ImageTk.PhotoImage(img)
        img_lbl.configure(image=img_tk); img_lbl.image = img_tk
        label_txt = str(int(cur_lbls[i])) if cb_view.get()!="label" else class_names_d[task][int(cur_lbls[i])]
        info.set(os.path.basename(cur_paths[i]) + f"\nclass : {label_txt}")
    fig.canvas.mpl_connect("pick_event", on_pick)

    # ---------- zoom ------------------------------------------------------
    last_click = {'pos': None, 'marker': None}
    def on_mouse_click(event):
        if event.button == 1 and event.inaxes and event.xdata is not None and event.ydata is not None:
            if last_click['marker'] is not None:
                last_click['marker'].remove()
            last_click['pos'] = (event.xdata, event.ydata)
            last_click['marker'] = ax.scatter(*last_click['pos'], marker='x', c='k', s=30, zorder=3)
            canv.draw_idle()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    def zoom(scale: float):
        if scale <= 0: return
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        cx, cy = last_click['pos'] if last_click['pos'] else ((x0 + x1)/2, (y0 + y1)/2)
        new_w = (x1 - x0) / scale; new_h = (y1 - y0) / scale
        ax.set_xlim(cx - new_w/2, cx + new_w/2); ax.set_ylim(cy - new_h/2, cy + new_h/2)
        canv.draw_idle()

    def on_scroll(event):
        direction = getattr(event, "step", 1 if getattr(event, "button", "")=="up" else -1)
        zoom(1.2 if direction > 0 else 1/1.2)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key(event):
        if event.key in ['+', '=']: zoom(1.2)
        elif event.key == '-': zoom(1/1.2)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # ---------- scores globaux (si fournis) -------------------------------
    if metric_scores:
        frm = tk.Frame(R); frm.pack(pady=6, fill="x")
        tk.Label(frm, text="Scores (globaux):", font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        for k,v in metric_scores.items():
            tk.Label(frm, text=f"{k:18s}: {v:.4f}").pack(anchor="w")

    # premier affichage
    update_view_boxes()
    draw()
    root.mainloop()


# ------------------------------------------------------------------
#  1)  CHARGEMENT  G  +  SupHeads
# ------------------------------------------------------------------
def find_latest_ckpt(folder: Path, pattern: str = "G_*.pt") -> Optional[Path]:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None





from typing import Tuple, Optional, Dict
import json
import torch
from pathlib import Path as PPath


def _split_gen_and_sup(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
Dict[str, torch.Tensor],
Optional[Dict[str, int]],
Optional[int]]:
    """
    Sépare les clés générateur / sup_heads et déduit (tasks, in_dim) si possible.

    - Compatible avec SupHeads MLP (classifiers.<task>.<idx>.weight/bias),
      ou Linear simple (classifiers.<task>.weight/bias).
    - Ignore les tenseurs non-2D pour l'inférence de (n_classes, in_dim).
    """
    gen_sd: Dict[str, torch.Tensor] = {}
    sup_sd: Dict[str, torch.Tensor] = {}

    # Pour l'inférence automatique
    tasks_auto: Dict[str, int] = {}
    in_dim_auto: Optional[int] = None

    # Collecte intermédiaire pour matcher weight<->bias par tâche
    #   weights_2d[task] = list of (full_key, tensor)
    #   biases_1d[task]  = set of possible bias prefixes "classifiers.task.<idx>.bias" / "classifiers.task.bias"
    weights_2d: Dict[str, list] = {}
    biases_1d: Dict[str, Dict[str, int]] = {}

    def _parse_task_from_classifier_key(kk: str) -> Optional[Tuple[str, str]]:
        """
        kk: clé sans le préfixe 'sup_heads.' (si présent). Ex:
            'classifiers.Weather Type.4.weight'  → ('Weather Type', '4.weight')
            'classifiers.Road Spray.weight'      → ('Road Spray', 'weight')
        Retourne (task_name, tail) ou None si pas dans 'classifiers.'.
        """
        if not kk.startswith("classifiers."):
            return None
        rest = kk[len("classifiers."):]  # e.g. 'Weather Type.4.weight'
        task, sep, tail = rest.partition(".")
        if not task:
            return None
        return task, (tail if sep else "")

    # 1) Split des clés et collecte pour analyse des classifieurs
    for k, v in sd.items():
        if k.startswith("sup_heads."):
            kk = k[len("sup_heads."):]  # strip 'sup_heads.' pour un chargement direct
            sup_sd[kk] = v

            parsed = _parse_task_from_classifier_key(kk)
            if parsed is not None:
                task, tail = parsed
                # Poids 2D potentiels (Linear)
                if tail.endswith(".weight") or tail == "weight":
                    if v.dim() == 2:
                        weights_2d.setdefault(task, []).append((kk, v))
                # Bias 1D potentiels
                if tail.endswith(".bias") or tail == "bias":
                    if v.dim() == 1:
                        # On stocke la longueur (n_out) pour le match
                        biases_1d.setdefault(task, {})[kk.replace(".bias", ".weight")] = int(v.shape[0])
        else:
            gen_sd[k] = v

    # 2) Pour chaque tâche, choisir le "dernier Linear" (idéalement via match weight<->bias)
    for task, cand_list in weights_2d.items():
        chosen = None
        # a) essayer de trouver un poids avec bias correspondant (même préfixe, même out_features)
        bmap = biases_1d.get(task, {})
        for kk, w in cand_list:
            if kk in bmap and w.dim() == 2 and int(w.shape[0]) == int(bmap[kk]):
                # match parfait: weight<->bias
                chosen = (kk, w)
                break
        # b) sinon, fallback: prendre le poids 2D avec le plus petit out_features (souvent n_classes)
        if chosen is None:
            chosen = min(cand_list, key=lambda item: int(item[1].shape[0]))

        # Inférer (n_classes, in_dim) depuis le poids choisi
        _, w = chosen
        if w.dim() == 2:
            n_cls, in_d = int(w.shape[0]), int(w.shape[1])
            tasks_auto[task] = n_cls
            if in_dim_auto is None:
                in_dim_auto = in_d

    return gen_sd, sup_sd, (tasks_auto if tasks_auto else None), in_dim_auto

def _load_any_state(path: PPath, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj

from typing import Optional, Tuple, Dict
from pathlib import Path as PPath

def load_models(
        weights_dir : PPath,
        device      : torch.device,
        cfg         : dict,
        ckpt_gen    : Optional[str] = None,
        sup_ckpt    : Optional[str] = None,
        classes_json: Optional[str] = None,
        sup_in_dim  : Optional[int] = None,
        *,
        strict_gen  : bool = True,
        strict_sup  : bool = True,
        # --- NEW: ckpts optionnels pour G_A et G_B ---
        ckpt_GA     : Optional[str] = None,
        ckpt_GB     : Optional[str] = None,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], Optional[dict]]:
    """
    Charge:
      • G  (générateur 'par défaut', strict par défaut)
      • SupHeads (optionnel, strict par défaut)
      • et, si fournis, G_A et G_B (ckpts séparés) attachés comme attributs de G:
            G.GA et G.GB

    Retour inchangé pour compat: (G, sup_heads, task_classes)
    → Récupération de GA/GB plus tard:
         G_A = getattr(G, "GA", G)
         G_B = getattr(G, "GB", G)

    Ordre de recherche SupHeads :
      1) --sup_ckpt (bundle riche recommandé)
      2) fallback: clés sup_heads.* trouvées dans le ckpt générateur (si présentes)
    """
    from models.generator import UNetGenerator
    from models.sup_heads import SupHeads
    import torch, json

# --------- NEW: infer generator architecture from saved run config ----------
def _load_train_cfg_hparams(wdir: PPath) -> Dict[str, Any]:
    """Load train_cfg.json (or hyperparameters.json/hparams.json) if present."""
    for name in ("train_cfg.json", "hyperparameters.json", "hparams.json"):
        p = PPath(wdir) / name
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                if isinstance(obj, dict) and "static_hparams" in obj and isinstance(obj["static_hparams"], dict):
                    return obj["static_hparams"]
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}

def _infer_img_size(hp: Dict[str, Any]) -> int:
    for k in ("img_size", "crop_size", "load_size", "resize", "image_size"):
        v = hp.get(k, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return 256

def _infer_gen_kwargs(wdir: PPath, cfg: dict, token_dim: int) -> Dict[str, Any]:
    hp = _load_train_cfg_hparams(wdir)
    # default to current behavior if not found
    arch_depth_delta = int(hp.get("arch_depth_delta", cfg.get("model", {}).get("arch_depth_delta", 0) or 0))
    style_token_levels = int(hp.get("style_token_levels", cfg.get("model", {}).get("style_token_levels", -1) or -1))
    unet_min_spatial = int(hp.get("unet_min_spatial", cfg.get("model", {}).get("unet_min_spatial", 2) or 2))
    img_size = int(hp.get("img_size", _infer_img_size(hp)))
    norm_variant = str(hp.get("norm_variant", "legacy") or "legacy")
    extra_bot_resblocks = int(hp.get("extra_bot_resblocks", 0) or 0)
    # allow overriding token_dim if saved
    td = hp.get("token_dim", hp.get("hid_dim", None))
    if isinstance(td, int) and td > 0:
        token_dim = int(td)

    return dict(
        token_dim=int(token_dim),
        arch_depth_delta=int(arch_depth_delta),
        style_token_levels=int(style_token_levels),
        img_size=int(img_size),
        unet_min_spatial=int(unet_min_spatial),
        norm_variant=norm_variant,
        extra_bot_resblocks=int(extra_bot_resblocks),
    )

    # ---------- helpers local ----------
    def _read_state_any(path: PPath, dev: torch.device):
        path = PPath(path)
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device=str(dev))
        obj = torch.load(path, map_location=dev)
        return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    def _align_dp_prefix(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Aligne la présence/absence de 'module.' entre ckpt et modèle."""
        if not sd:
            return sd
        m_keys = list(model.state_dict().keys())
        model_has_mod  = any(k.startswith("module.") for k in m_keys)
        ckpt_has_mod   = any(k.startswith("module.") for k in sd.keys())
        if ckpt_has_mod and (not model_has_mod):
            return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        if (not ckpt_has_mod) and model_has_mod:
            return { (f"module.{k}" if not k.startswith("module.") else k): v for k, v in sd.items() }
        return sd

    # utilise le remap défini ailleurs (attn.→attentions., clf.→classifiers., filtre anciens style-enc)
    def _maybe_remap(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            return _remap_keys(sd)  # ta fonction déjà présente dans le module
        except NameError:
            return sd

    # petit chargeur générique pour un générateur complet depuis un ckpt séparé
    def _load_generator_from_ckpt(ckpt: PPath, device: torch.device,
                                  gen_kwargs: Dict[str, Any], *, strict: bool = True) -> torch.nn.Module:
        Gx = UNetGenerator(**gen_kwargs).to(device)
        sd = _read_state_any(ckpt, device)
        sd = _maybe_remap(sd)
        sd = _align_dp_prefix(sd, Gx)
        try:
            Gx.load_state_dict(sd, strict=True if strict else False)
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"[STRICT] Chargement générateur impossible pour {ckpt.name} :\n{e}")
            Gx.load_state_dict(sd, strict=False)
        Gx.eval()
        return Gx

    # ---------- 1) Générateur 'G' ----------
    if ckpt_gen is None:
        ckpt_gen = (cfg.get("model", {}).get("gen_ckpt_best")
                    or cfg.get("model", {}).get("gen_ckpt_last"))
        ckpt_gen = (weights_dir / ckpt_gen) if ckpt_gen else find_latest_ckpt(weights_dir)

    ckpt_gen = PPath(ckpt_gen)
    if not ckpt_gen or not ckpt_gen.exists():
        raise FileNotFoundError(f"G introuvable : {ckpt_gen}")

    token_dim_cfg = int(cfg.get("model", {}).get("token_dim", 256))
    gen_kwargs = _infer_gen_kwargs(weights_dir, cfg, token_dim_cfg)
    G = UNetGenerator(**gen_kwargs).to(device)

    full_sd = _read_state_any(ckpt_gen, device)
    full_sd = _maybe_remap(full_sd)
    gen_sd, sup_from_gen_sd, tasks_auto, in_dim_auto = _split_gen_and_sup(full_sd)

    gen_sd = _align_dp_prefix(gen_sd, G)
    try:
        G.load_state_dict(gen_sd, strict=True if strict_gen else False)
    except RuntimeError as e:
        if strict_gen:
            raise RuntimeError(
                f"[STRICT] Chargement du générateur impossible pour {ckpt_gen.name} :\n{e}"
            )
        G.load_state_dict(gen_sd, strict=False)

    G.eval()
    print(f"✓ G loaded (strict={strict_gen}) – {ckpt_gen.name}")

    # ---------- 1-bis) Charger G_A et G_B si demandés ----------
    # Par défaut: si non fournis, on réutilise G pour GA/GB.
    GA = G
    GB = G

    if ckpt_GA:
        ckpt_GA_p = PPath(ckpt_GA)
        if not ckpt_GA_p.exists():
            raise FileNotFoundError(f"G_A introuvable : {ckpt_GA_p}")
        GA = _load_generator_from_ckpt(ckpt_GA_p, device, gen_kwargs, strict=strict_gen)
        print(f"✓ G_A loaded – {ckpt_GA_p.name}")

    if ckpt_GB:
        ckpt_GB_p = PPath(ckpt_GB)
        if not ckpt_GB_p.exists():
            raise FileNotFoundError(f"G_B introuvable : {ckpt_GB_p}")
        GB = _load_generator_from_ckpt(ckpt_GB_p, device, gen_kwargs, strict=strict_gen)
        print(f"✓ G_B loaded – {ckpt_GB_p.name}")

    # On attache GA/GB sur G pour compat descendante (retour inchangé).
    # Ainsi, main peut faire: G_A = getattr(G, "GA", G) ; G_B = getattr(G, "GB", G)
    setattr(G, "GA", GA)
    setattr(G, "GB", GB)

    # ---------- 2) SupHeads (optionnel) ----------
    sup_heads: Optional[torch.nn.Module] = None
    task_classes: Optional[dict]        = None

    if sup_ckpt is None:
        sup_ckpt = find_latest_ckpt(weights_dir, "SupHeads_*.pth") or \
                   find_latest_ckpt(weights_dir, "SupHeads_*.pt")
    sup_path = PPath(sup_ckpt) if sup_ckpt else None

    if sup_path and sup_path.exists():
        bundle = torch.load(sup_path, map_location=device)
        if isinstance(bundle, dict) and "meta" in bundle and "state_dict" in bundle:
            meta, state_dict = bundle["meta"], bundle["state_dict"]
            task_classes = meta.get("tasks", None)
            in_dim = meta.get("in_dim", None)
            if in_dim is None:
                for k, v in state_dict.items():
                    if k.startswith("classifiers.") and k.endswith(".weight") and v.dim() == 2:
                        in_dim = int(v.shape[1]); break
            if task_classes is None or in_dim is None:
                raise RuntimeError(f"Bundle SupHeads incomplet (meta.tasks/in_dim manquants) : {sup_path.name}")

            sup_heads = SupHeads(task_classes, in_dim, heads=4).to(device)
            state_dict = _maybe_remap(state_dict)
            state_dict = _align_dp_prefix(state_dict, sup_heads)
            try:
                sup_heads.load_state_dict(state_dict, strict=True if strict_sup else False)
            except RuntimeError as e:
                raise RuntimeError(
                    f"[STRICT] Chargement SupHeads impossible pour {sup_path.name} :\n{e}\n"
                    f"→ Vérifie que 'tasks' et 'in_dim' du bundle correspondent à l'archi de SupHeads."
                )
            sup_heads.eval()
            print(f"✓ SupHeads loaded (rich, strict={strict_sup}) – {sup_path.name}")

        else:
            # ----- legacy -----
            if classes_json is None and (tasks_auto is None or sup_in_dim is None):
                raise RuntimeError(
                    "Ancien SupHeads (state_dict brut) détecté : fournis --classes_json et --sup_in_dim "
                    "ou bien exporte un bundle riche (save_supheads_rich) pour un chargement strict."
                )
            if classes_json:
                with open(classes_json) as f:
                    task_classes = {t: len(cls) for t, cls in json.load(f).items()}
                in_dim = sup_in_dim or _detect_in_dim_from_state_dict(bundle)
            else:
                task_classes = tasks_auto
                in_dim = sup_in_dim

            sup_heads = SupHeads(task_classes, in_dim).to(device)
            state_dict = _maybe_remap(bundle)
            state_dict = _align_dp_prefix(state_dict, sup_heads)
            try:
                sup_heads.load_state_dict(state_dict, strict=True if strict_sup else False)
            except RuntimeError as e:
                raise RuntimeError(
                    f"[STRICT] Chargement SupHeads (legacy) impossible pour {sup_path.name} :\n{e}\n"
                    f"→ Assure-toi que classes_json/sup_in_dim correspondent EXACTEMENT au ckpt."
                )
            sup_heads.eval()
            print(f"✓ SupHeads loaded (legacy, strict={strict_sup}) – {sup_path.name}")

    elif sup_from_gen_sd:
        sup_from_gen_sd = _maybe_remap(sup_from_gen_sd)
        if classes_json:
            with open(classes_json) as f:
                task_classes = {t: len(cls) for t, cls in json.load(f).items()}
            in_dim = sup_in_dim or in_dim_auto
        else:
            task_classes = tasks_auto
            in_dim = in_dim_auto

        if task_classes is None or in_dim is None:
            raise RuntimeError(
                "Clés sup_heads.* trouvées dans le ckpt G mais impossible de déduire (tasks/in_dim).\n"
                "→ Fourni --classes_json et éventuellement --sup_in_dim, ou utilise un SupHeads_*.pth riche."
            )

        sup_heads = SupHeads(task_classes, in_dim).to(device)
        sup_from_gen_sd = _align_dp_prefix(sup_from_gen_sd, sup_heads)
        try:
            sup_heads.load_state_dict(sup_from_gen_sd, strict=True if strict_sup else False)
        except RuntimeError as e:
            raise RuntimeError(
                f"[STRICT] Chargement SupHeads depuis le ckpt générateur impossible :\n{e}\n"
                f"→ Mismatch d'architecture (n_classes/in_dim). Utilise un bundle riche SupHeads."
            )
        sup_heads.eval()
        print("✓ SupHeads extracted from generator checkpoint (strict).")

    # else: pas de SupHeads, OK

    return G, sup_heads, task_classes




# ---- utilitaire si tu charges un ancien .pt brut des SupHeads
def _detect_in_dim_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for k, v in state_dict.items():
        if k.startswith("classifiers.") and k.endswith(".weight"):
            # shape: (n_classes, in_dim)
            return int(v.shape[1])
    raise RuntimeError("Impossible de déduire in_dim depuis le state_dict SupHeads")



# ------------------------------- pooling utilitaire
def _pool_map(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pool spatial d’un tenseur (B,C,H,W) → (B,C) pour 'mean'/'max',
    ou flatten (B, C*H*W) si mode='none'.
    """
    if mode == "mean":
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if mode == "max":
        return F.adaptive_max_pool2d(x, 1).flatten(1)
    # "none": on garde toutes les positions
    return x.flatten(1)

def _parse_weights(w: Optional[Sequence[float] | str], n: int) -> List[float]:
    """
    Convertit None / str / séquence en liste de longueur n.
    None → [1.0]*n (poids égaux par défaut).
    """
    if w is None:
        return [1.0] * n
    if isinstance(w, str):
        vals = [float(t) for t in w.split(",") if t.strip()]
    else:
        vals = [float(t) for t in w]
    if len(vals) == n:
        return vals
    if len(vals) == 1:
        return vals * n
    # pad / tronque
    return (vals + [vals[-1]] * (n - len(vals)))[:n]

# ------------------------------- extraction 1 image → 1 vecteur
# ------------------------------- extraction 1 image → 1 vecteur (multi-échelles)
@torch.no_grad()
def extract_style_signature(
    G,
    x,                         # (1,C,H,W)
    *,
    embed_type    : str = "tok6",       # NOUVEAU défaut multi-échelles
    token_pool    : str = "mean",       # "mean" | "max" | "none" (utilisé pour mgap / tok+delta)
    layers        = None,               # compat (ignoré ici)
    delta_weights = None                # str ou séquence
) -> torch.Tensor:
    """
    Embeddings centrés tokens (nouveaux) :
      - tokG        : token global de style                              → (D)
      - tok6        : concat [tokG, t5, t4, t3, t2, t1]                  → (6D)
      - tok6_mean   : moyenne L2-normée des 6 tokens (tokG + 5 locaux)   → (D)
      - tok6_w      : moyenne pondérée des 6 tokens (wG,w5..w1)          → (D)

    Modes historiques (compat) :
      - style_tok   ≡ tokG
      - bot         : GAP du bottleneck contenu
      - bot+tok     : concat(GAP(bottleneck), tokG)
      - tok+delta   : tokG + concat_i [ w_i * pool(|m_i|) ]   (m5..m1)
      - mgap        : concat_i [ w_i * pool(m_i) ]            (m5..m1)
      - mgap+tok    : mgap + tokG

    Notes:
      - `delta_weights` :
          * pour tok6_w → 6 poids "wG,w5,w4,w3,w2,w1"
          * pour tok+delta / mgap / mgap+tok → 5 poids "w5,w4,w3,w2,w1"
    """
    import torch
    import torch.nn.functional as F

    # ---------- helpers ----------
    def _parse_weights_any(w, n, fill=1.0):
        if w is None: return [float(fill)] * n
        if isinstance(w, str):
            vals = [float(t) for t in w.split(",") if t.strip()]
        else:
            vals = [float(t) for t in w]
        if len(vals) == n: return vals
        if len(vals) == 1: return vals * n
        return (vals + [vals[-1]] * (n - len(vals)))[:n]

    def _l2(t):  # (B,D) → L2-norm per row
        return t / (t.norm(dim=1, keepdim=True) + 1e-8)

    gap = lambda t: F.adaptive_avg_pool2d(t, 1).flatten(1)

    # ---------- encode contenu (toujours dispo) ----------
    z, _skips = G.encode_content(x)  # (1, Cb, h, w)

    # ---------- style_enc : détecte (maps,toks,tokG) | (toks,tokG) | (maps,token) | (token,) ----------
    maps, toks, tokG = None, None, None
    try:
        se = G.style_enc(x)
        if isinstance(se, (list, tuple)):
            if len(se) == 3:
                # attendu : (maps, toks, tokG)
                a, b, c = se
                # robustesse : re-détecte par forme
                if isinstance(a, (list, tuple)) and len(a) and getattr(a[0], "dim", lambda: 0)() == 4:
                    maps = a
                if isinstance(b, (list, tuple)) and len(b) and getattr(b[0], "dim", lambda: 0)() == 2:
                    toks = b
                if hasattr(c, "dim") and c.dim() == 2:
                    tokG = c
            elif len(se) == 2:
                a, b = se
                # cas (maps, token) OU (toks, tokG)
                if isinstance(a, (list, tuple)) and len(a) and getattr(a[0], "dim", lambda: 0)() == 4 and hasattr(b, "dim") and b.dim() == 2:
                    maps, tokG = a, b
                elif isinstance(a, (list, tuple)) and len(a) and getattr(a[0], "dim", lambda: 0)() == 2 and hasattr(b, "dim") and b.dim() == 2:
                    toks, tokG = a, b
            elif len(se) == 1 and hasattr(se[0], "dim") and se[0].dim() == 2:
                tokG = se[0]
        elif hasattr(se, "dim") and se.dim() == 2:
            tokG = se
    except Exception:
        pass

    # ---------- NOUVEAUX MODES centrés tokens ----------
    # tok6* sont conservés pour compat; si L != 5, tok6 se comporte comme tokL.
    if embed_type in (
        "tokG", "style_tok",
        "tok6", "tok6_mean", "tok6_w",
        "tokL", "tokL_mean", "tokL_w",
    ):
        # fallback : si aucun token dispo, utilise le bottleneck
        if tokG is None and toks is None:
            return gap(z).squeeze(0)

        # si pas tokG, approx par zéro (ou moyenne des locaux)
        if tokG is None and toks is not None and len(toks) > 0:
            tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])

        seq = [tokG] + (list(toks) if toks is not None else [])
        seq = [_l2(t) for t in seq]  # (B,D) chacun
        n = len(seq)                 # = (L+1)

        if embed_type in ("tokG", "style_tok"):
            return seq[0].squeeze(0)  # (D,)

        # concat tous les tokens : [tokG] + locaux (L variable)
        if embed_type in ("tok6", "tokL"):
            return torch.cat(seq, dim=1).squeeze(0)  # ((L+1)D,)

        # moyenne uniforme
        if embed_type in ("tok6_mean", "tokL_mean"):
            return torch.stack(seq, dim=1).mean(1).squeeze(0)  # (D,)

        # moyenne pondérée : accepte n poids (si moins → pad, si plus → trim)
        if embed_type in ("tok6_w", "tokL_w"):
            w = torch.as_tensor(_parse_weights_any(delta_weights, n), device=seq[0].device, dtype=seq[0].dtype)
            W = (w / (w.sum() + 1e-8)).view(1, -1, 1)  # (1,n,1)
            S = torch.stack(seq, dim=1)                # (B,n,D)
            return (S * W).sum(1).squeeze(0)           # (D,)

# ---------- MODES HISTORIQUES (compat) ----------
    # maps requis pour mgap / tok+delta / mgap+tok ; sinon on retombe sur bot / bot+tok
    if token_pool == "max":
        pool = lambda m: F.adaptive_max_pool2d(m, 1).flatten(1)
    elif token_pool == "none":
        pool = lambda m: m.flatten(1)
    else:
        pool = lambda m: F.adaptive_avg_pool2d(m, 1).flatten(1)

    if embed_type == "bot":
        return gap(z).squeeze(0)

    if embed_type == "bot+tok":
        t = tokG if tokG is not None else gap(z)
        return torch.cat([gap(z), t], dim=1).squeeze(0)

    if maps is not None and embed_type in ("tok+delta", "mgap", "mgap+tok"):
        w5 = _parse_weights_any(delta_weights, 5)

        if embed_type == "mgap":
            mg = [wi * pool(mi) for wi, mi in zip(w5, maps)]
            return torch.cat(mg, dim=1).squeeze(0)

        if embed_type == "mgap+tok":
            mg = [wi * pool(mi) for wi, mi in zip(w5, maps)]
            t  = tokG if tokG is not None else gap(z)
            return torch.cat([torch.cat(mg, dim=1), t], dim=1).squeeze(0)

        if embed_type == "tok+delta":
            dvec = torch.cat([wi * pool(mi.abs()) for wi, mi in zip(w5, maps)], dim=1)
            t    = tokG if tokG is not None else gap(z)
            return torch.cat([t, dvec], dim=1).squeeze(0)

    # fallback ultime
    return gap(z).squeeze(0)


@torch.no_grad()
def compute_style_embeddings(
    G,
    loader,
    device,
    *,
    embed_type   : str = "tok6",
    token_pool   : str = "mean",
    layers       : str = "",
    pca_dim      : Optional[int] = None,
    l2_norm      : bool = False,
    delta_weights: Optional[Sequence[float] | str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, List[str]]]:

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset

    # --------- SAFE: choisir un générateur "feuille" et eval() non récursif ----------
    def _pick_leaf_generator(G_):
        if hasattr(G_, "G_leaf") and isinstance(getattr(G_, "G_leaf"), nn.Module):
            return getattr(G_, "G_leaf")
        for name in ("GA", "G_A", "GB", "G_B"):  # priorité implicite à GA/G_A si présent en premier dans ta liste
            if hasattr(G_, name) and isinstance(getattr(G_, name), nn.Module):
                return getattr(G_, name)
        return G_

    def _set_mode_no_recurse(mod: nn.Module, *, train: bool):
        """Place/clear le flag .training sans appeler nn.Module.train() (évite récursions/cycles)."""
        if not isinstance(mod, nn.Module):
            return
        seen = set()
        stack = [mod]
        while stack:
            m = stack.pop()
            mid = id(m)
            if mid in seen or not isinstance(m, nn.Module):
                continue
            seen.add(mid)
            object.__setattr__(m, "training", bool(train))
            # on descend via _modules en contrôlant 'seen' → pas de récursion infinie même s'il y avait un cycle
            for child in getattr(m, "_modules", {}).values():
                if isinstance(child, nn.Module):
                    stack.append(child)

    Gnet = _pick_leaf_generator(G)
    _set_mode_no_recurse(Gnet, train=False)  # équivalent à eval() mais sans récursion

    # -------------------------------------------------------------------------------
    # le reste de ta fonction reste identique, mais **utilise Gnet** au lieu de G
    # (remplace toutes les occurrences de G par Gnet ci-dessous)
    # -------------------------------------------------------------------------------

    if layers:
        _ = [l.strip() for l in str(layers).split(",") if l.strip()]

    # --- Dataset / Subset pour chemins robustes ---
    ds_wrapped = loader.dataset
    if isinstance(ds_wrapped, Subset):
        base_ds, subset_idx = ds_wrapped.dataset, ds_wrapped.indices
    else:
        base_ds, subset_idx = ds_wrapped, None

    # --- tâches & mapping classes ---
    if hasattr(base_ds, "task_classes") and isinstance(getattr(base_ds, "task_classes"), dict):
        task_names = list(base_ds.task_classes.keys())
        class_maps = dict(base_ds.task_classes)
    else:
        classes = list(getattr(base_ds, "classes", []))
        task_names = ["__DEFAULT__"]
        class_maps = {"__DEFAULT__": classes}

    embs  : Dict[str, List[np.ndarray]] = {t: [] for t in task_names}
    labels: Dict[str, List[np.ndarray]] = {t: [] for t in task_names}
    paths : Dict[str, List[str]]        = {t: [] for t in task_names}

    global_idx = 0
    for batch in loader:
        if len(batch) == 3:
            imgs, lbl_batch, path_list = batch
        else:
            imgs, lbl_batch = batch
            B = imgs.size(0)
            ids = ([subset_idx[global_idx + i] for i in range(B)]
                   if subset_idx is not None else
                   list(range(global_idx, global_idx + B)))
            samples = getattr(base_ds, "samples", None) or getattr(base_ds, "imgs", None)
            if samples is None:
                raise AttributeError("Le dataset ne possède pas d'attribut 'samples'/'imgs' pour récupérer les chemins.")
            path_list = [samples[j][0] for j in ids]

        imgs = imgs.to(device, non_blocking=True)

        sigs = []
        for i in range(imgs.size(0)):
            v = extract_style_signature(
                Gnet, imgs[i:i+1],
                embed_type=embed_type,
                token_pool=token_pool,
                layers=None,
                delta_weights=delta_weights
            ).cpu().numpy()
            sigs.append(v)
        sig_batch = np.stack(sigs, 0)

        if isinstance(lbl_batch, dict):  # Multi-tâches
            for t in task_names:
                raw = lbl_batch.get(t, None)
                if raw is None:
                    continue
                lbl_t = torch.as_tensor(raw)
                mask  = (lbl_t >= 0)
                if mask.any():
                    m = mask.cpu().numpy()
                    embs[t].append(sig_batch[m])
                    labels[t].append(lbl_t[m].cpu().numpy())
                    paths[t].extend([p for p, keep in zip(path_list, m) if keep])
        else:
            t = "__DEFAULT__"
            embs[t].append(sig_batch)
            labels[t].append(lbl_batch.cpu().numpy())
            paths[t].extend(path_list)

        global_idx += imgs.size(0)

    for t in task_names:
        embs[t]   = np.concatenate(embs[t],   0) if embs[t]   else np.zeros((0, 1), dtype=np.float32)
        labels[t] = np.concatenate(labels[t], 0) if labels[t] else np.array([], dtype=np.int64)
        paths[t]  = list(paths[t])

    if (pca_dim is not None) or l2_norm:
        order   = list(task_names)
        lengths = {t: embs[t].shape[0] for t in order}
        X = np.concatenate([embs[t] for t in order], 0) if order else np.zeros((0, 1), dtype=np.float32)

        if X.size and (pca_dim is not None) and 0 < pca_dim < X.shape[1]:
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            X = Xc @ Vt[:pca_dim].T

        if X.size and l2_norm:
            n = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(n, 1e-12, None)

        s = 0
        for t in order:
            n = lengths[t]
            embs[t] = X[s:s+n]
            s += n

    if "__DEFAULT__" in embs and "default" not in embs:
        embs["default"]   = embs["__DEFAULT__"]
        labels["default"] = labels["__DEFAULT__"]
        paths["default"]  = paths["__DEFAULT__"]
        class_maps["default"] = class_maps["__DEFAULT__"]

    return embs, labels, class_maps, paths


def compute_sem_embeddings(
    sem_backbone,
    loader,
    device,
    *,
    imagenet_norm: bool = True,
    pca_dim: Optional[int] = None,
    l2_norm: bool = False,
):
    """Compute embeddings from a semantic ResNet backbone (GAP(layerX)).

    This mirrors compute_style_embeddings output contract:
      (embs_by_task, labels_by_task, class_maps_by_task, paths_by_task)
    """
    import numpy as np
    import torch
    from torch.utils.data import Subset

    sem_backbone.eval()

    # Dataset / Subset for paths
    ds_wrapped = loader.dataset
    if isinstance(ds_wrapped, Subset):
        base_ds, subset_idx = ds_wrapped.dataset, ds_wrapped.indices
    else:
        base_ds, subset_idx = ds_wrapped, None

    # tasks & class maps
    if hasattr(base_ds, "task_classes") and isinstance(getattr(base_ds, "task_classes"), dict):
        task_names = list(base_ds.task_classes.keys())
        class_maps = dict(base_ds.task_classes)
    else:
        classes = list(getattr(base_ds, "classes", []))
        task_names = ["__DEFAULT__"]
        class_maps = {"__DEFAULT__": classes}

    embs: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    labels: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    paths: dict[str, list[str]] = {t: [] for t in task_names}

    _im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    global_idx = 0
    for batch in loader:
        if len(batch) == 3:
            imgs, lbl_batch, path_list = batch
        else:
            imgs, lbl_batch = batch
            B = imgs.size(0)
            ids = ([subset_idx[global_idx + i] for i in range(B)]
                   if subset_idx is not None else
                   list(range(global_idx, global_idx + B)))
            samples = getattr(base_ds, "samples", None) or getattr(base_ds, "imgs", None)
            if samples is None:
                raise AttributeError("Le dataset ne possède pas d'attribut 'samples'/'imgs' pour récupérer les chemins.")
            path_list = [samples[j][0] for j in ids]

        imgs = imgs.to(device, non_blocking=True)
        x = imgs
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if imagenet_norm:
            # convert [-1,1] -> [0,1] then ImageNet normalize
            x = (x + 1.0) * 0.5
            x = x.clamp(0.0, 1.0)
            x = (x - _im_mean) / _im_std

        with torch.no_grad():
            out = sem_backbone(x)
            if isinstance(out, dict):
                feat = out.get("0", None)
                if feat is None:
                    feat = next(iter(out.values()))
            else:
                feat = out
            feat = feat.mean(dim=(2, 3))  # GAP
            feat_np = feat.detach().cpu().numpy().astype(np.float32)

        if isinstance(lbl_batch, dict):
            for t in task_names:
                raw = lbl_batch.get(t, None)
                if raw is None:
                    continue
                lbl_t = torch.as_tensor(raw)
                mask = (lbl_t >= 0)
                if mask.any():
                    m = mask.cpu().numpy()
                    embs[t].append(feat_np[m])
                    labels[t].append(lbl_t[mask].cpu().numpy())
                    paths[t].extend([p for p, keep in zip(path_list, m) if keep])
        else:
            t = "__DEFAULT__"
            embs[t].append(feat_np)
            labels[t].append(lbl_batch.cpu().numpy())
            paths[t].extend(path_list)

        global_idx += imgs.size(0)

    for t in task_names:
        embs[t] = np.concatenate(embs[t], 0) if embs[t] else np.zeros((0, 1), dtype=np.float32)
        labels[t] = np.concatenate(labels[t], 0) if labels[t] else np.array([], dtype=np.int64)
        paths[t] = list(paths[t])

    # optional PCA + L2 like style branch
    if (pca_dim is not None) or l2_norm:
        order = list(task_names)
        lengths = {t: embs[t].shape[0] for t in order}
        X = np.concatenate([embs[t] for t in order], 0) if order else np.zeros((0, 1), dtype=np.float32)

        if X.size and (pca_dim is not None) and 0 < pca_dim < X.shape[1]:
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            X = Xc @ Vt[:pca_dim].T

        if X.size and l2_norm:
            n = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(n, 1e-12, None)

        s = 0
        for t in order:
            n = lengths[t]
            embs[t] = X[s:s+n]
            s += n

    if "__DEFAULT__" in embs and "default" not in embs:
        embs["default"] = embs["__DEFAULT__"]
        labels["default"] = labels["__DEFAULT__"]
        paths["default"] = paths["__DEFAULT__"]
        class_maps["default"] = class_maps["__DEFAULT__"]

    return embs, labels, class_maps, paths








# ------------------------------------------------------------------
#  4)  Parcours DataLoader → embeddings + chemins (auto feat_type)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
#  4)  Parcours DataLoader → embeddings + chemins (auto feat_type)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
#  4)  Parcours DataLoader → embeddings + chemins (auto feat_type)
# ------------------------------------------------------------------
@torch.no_grad()
def compute_embeddings_with_paths(
    model,
    loader,
    device: torch.device,
    *,
    per_task: bool = False,
    per_task_tsne: bool | None = None   # alias rétro-compat
):
    if per_task_tsne is not None:
        per_task = bool(per_task_tsne)

    import numpy as np
    import torch
    from torch.utils.data import Subset

    model.eval()

    # ---------- helpers ----------
    def _has_sup_stack(m):
        return hasattr(m, "G") and hasattr(m, "Sup") and (m.Sup is not None)

    def _get_tasks(m):
        if hasattr(m, "tasks") and m.tasks:
            return list(m.tasks)
        if _has_sup_stack(m) and hasattr(m.Sup, "tasks") and m.Sup.tasks:
            return list(m.Sup.tasks.keys())
        return []

    def _normalize_delta_w(dw):
        if isinstance(dw, (list, tuple)):
            return ",".join(str(float(x)) for x in dw)
        return str(dw)

    def _norm_key(s: str) -> str:
        return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")

    def _to_long_labels(vals, B: int) -> torch.Tensor:
        """list/tuple/ndarray/tensor/int/None -> LongTensor(B,) ; None -> -1"""
        if isinstance(vals, torch.Tensor):
            if vals.dim() == 0:
                return torch.full((B,), int(vals.item()), dtype=torch.long)
            return vals.long()
        if isinstance(vals, (list, tuple, np.ndarray)):
            arr = [(-1 if (v is None) else int(v)) for v in list(vals)]
            return torch.tensor(arr, dtype=torch.long)
        if vals is None:
            return torch.full((B,), -1, dtype=torch.long)
        # scalaire
        return torch.full((B,), int(vals), dtype=torch.long)

    def _pick_feat_type_from_sup(G, Sup, delta_weights="1,1,1,1,1"):
        target = getattr(Sup, "in_dim", None)
        if target is None and hasattr(Sup, "classifiers"):
            for m in getattr(Sup, "classifiers").values():
                target = m.weight.shape[1]; break
        if target is None:
            return None
        candidates = ["tok+delta", "bot+tok", "style_tok", "tokL", "tokL_mean", "tok6", "tok6_mean"]
        try: dev_local = next(G.parameters()).device
        except Exception: dev_local = device
        x = torch.zeros(1, 3, 256, 256, device=dev_local)
        for ft in candidates:
            try:
                f = G.sup_features(x, ft, delta_weights=delta_weights)
                if f.shape[1] == target:
                    return ft
            except Exception:
                pass
        return None

    def _manual_per_task(model, imgs, feat_type, delta_w):
        feats = model.G.sup_features(imgs, feat_type, delta_weights=delta_w)
        # tolère les deux signatures (avec / sans return_task_embeddings)
        try:
            _, emb_dict = model.Sup(feats, return_task_embeddings=True)
        except TypeError:
            # anciens SupHeads : on construit depuis les logits par tâche si possible
            logits = model.Sup(feats)
            emb_dict = {t: v for t, v in logits.items()} if isinstance(logits, dict) else {"__DEFAULT__": logits}
        return emb_dict

    def _manual_global(model, imgs, feat_type, delta_w):
        feats = model.G.sup_features(imgs, feat_type, delta_weights=delta_w)
        return feats

    # ---------- config auto / mode extraction ----------
    use_manual = _has_sup_stack(model)
    feat_type = getattr(model, "feat_type", "tok+delta")
    delta_weights = _normalize_delta_w(getattr(model, "delta_weights", "1,1,1,1,1"))

    # IMPORTANT: when using a semantic backbone adapter (feat_type like 'sem_resnet50'),
    # we must NOT auto-guess another feat_type (tok6/tok+delta/...). Those names are
    # meaningful only for the generator branch. For semantic backbones, G.sup_features
    # ignores feat_type and always returns the semantic GAP vector; auto-guessing could
    # confuse logging/debugging and (in some wrappers) fall back to a generator path.
    force_feat_type = str(feat_type).lower().startswith("sem_resnet")

    if use_manual and (not force_feat_type):
        target_in = getattr(model.Sup, "in_dim", None)
        if target_in is None and hasattr(model.Sup, "classifiers"):
            for m in model.Sup.classifiers.values():
                target_in = m.weight.shape[1]; break
        if target_in is not None:
            dim_feat = None
            try: dim_feat = model.G.sup_in_dim_for(feat_type)
            except Exception: pass
            if dim_feat != target_in:
                ft_guess = _pick_feat_type_from_sup(model.G, model.Sup, delta_weights)
                if ft_guess is not None:
                    feat_type = ft_guess
                else:
                    use_manual = False  # fallback: laisser model(...) gérer

    # ---------- buffers ----------
    if per_task:
        task_names = _get_tasks(model) or ["__DEFAULT__"]
        task_embs  = {t: [] for t in task_names}
        task_lbls  = {t: [] for t in task_names}
        task_paths = {t: [] for t in task_names}
        task_map   = None  # dataset→SupHeads
    else:
        all_embs, all_lbls, all_paths = [], [], []

    # ---------- dataset / paths ----------
    ds_wrapped = loader.dataset
    if isinstance(ds_wrapped, Subset):
        base_ds, subset_idx = ds_wrapped.dataset, ds_wrapped.indices
    else:
        base_ds, subset_idx = ds_wrapped, None

    samples_attr = getattr(base_ds, "samples", None) or getattr(base_ds, "imgs", None)

    global_idx = 0
    for batch in loader:
        # (imgs, labels_dict, paths) | (imgs, labels_dict) | (imgs, labels_tensor)
        if len(batch) == 3:
            imgs, lbl_dict, p_list = batch
        elif len(batch) == 2:
            imgs, second = batch
            if isinstance(second, dict):
                lbl_dict = second
            else:
                # ImageFolder → une seule tâche "__DEFAULT__"
                lbl_dict = {"__DEFAULT__": second}
            Btmp = imgs.size(0)
            ids = (subset_idx[global_idx:global_idx+Btmp] if subset_idx is not None
                   else list(range(global_idx, global_idx+Btmp)))
            if samples_attr is None:
                raise AttributeError("Le dataset ne possède pas 'samples'/'imgs' pour récupérer les chemins.")
            p_list = [samples_attr[i][0] for i in ids]
        else:
            raise ValueError("Batch inattendu: attendu 2 ou 3 éléments.")

        imgs = imgs.to(device, non_blocking=True)
        B = imgs.size(0)

        # construit task_map (dataset↔SupHeads) une seule fois si dict
        if per_task and task_map is None and isinstance(lbl_dict, dict):
            nk = {_norm_key(k): k for k in lbl_dict.keys()}  # ex: "__default__" -> "__DEFAULT__"
            task_names_here = _get_tasks(model) or list(lbl_dict.keys())
            task_map = {t: nk.get(_norm_key(t), None) for t in task_names_here}

            # 🔧 Cas ImageFolder: si une seule clé "__DEFAULT__", on la réplique pour toutes les tâches
            if len(nk) == 1 and "__default__" in nk:
                default_key = nk["__default__"]
                task_map = {t: default_key for t in task_names_here}

        if per_task:
            # --- embeddings par tâche ---
            if use_manual:
                emb_dict = _manual_per_task(model, imgs, feat_type, delta_weights)
            else:
                # modèle composite possiblement : tolérer les deux signatures
                try:
                    _, emb_dict = model(imgs, return_task_embeddings=True)
                except TypeError:
                    feats = model(imgs, return_embeddings=True)
                    # Si pas d'API par tâche, on duplique en "__DEFAULT__"
                    emb_dict = {"__DEFAULT__": feats}

            for t, emb_t in emb_dict.items():
                # récupère la clé dataset correspondante, avec fallback robuste
                ds_key = task_map.get(t) if task_map else t
                if isinstance(lbl_dict, dict):
                    if ds_key is None or ds_key not in lbl_dict:
                        ds_key = "__DEFAULT__" if "__DEFAULT__" in lbl_dict else next(iter(lbl_dict))
                    vals = lbl_dict.get(ds_key, None)
                else:
                    vals = lbl_dict

                ds_vals = _to_long_labels(vals, B)

                for i in range(B):
                    li = int(ds_vals[i].item())
                    task_embs[t].append(emb_t[i].detach().cpu().numpy())
                    task_lbls[t].append(li)   # plus de -1 si mapping OK
                    task_paths[t].append(p_list[i])
        else:
            if use_manual:
                feats = _manual_global(model, imgs, feat_type, delta_weights)
                embs_np = feats.detach().cpu().numpy()
            else:
                embs_np = model(imgs, return_embeddings=True).detach().cpu().numpy()

            all_embs.append(embs_np)
            # si dict, on prend une clé stable : "__DEFAULT__" si dispo
            if isinstance(lbl_dict, dict):
                key = "__DEFAULT__" if "__DEFAULT__" in lbl_dict else next(iter(lbl_dict))
                lab = _to_long_labels(lbl_dict[key], B).cpu().numpy().tolist()
            else:
                lab = _to_long_labels(lbl_dict, B).cpu().numpy().tolist()
            all_lbls.extend(lab)
            all_paths.extend(p_list)

        global_idx += B

    # ---------- pack numpy ----------
    if per_task:
        for t in list(task_embs.keys()):
            task_embs[t] = (np.stack(task_embs[t], 0)
                            if len(task_embs[t]) > 0 else np.zeros((0, 1), dtype=np.float32))
            task_lbls[t] = np.asarray(task_lbls[t], dtype=np.int64)
        return task_embs, task_lbls, task_paths
    else:
        embs = (np.concatenate(all_embs, 0)
                if len(all_embs) > 0 else np.zeros((0, 1), dtype=np.float32))
        lbls = np.asarray(all_lbls, dtype=np.int64) if len(all_embs) > 0 else np.array([], dtype=np.int64)
        return embs, lbls, all_paths


# ───────────────────────── Grad-CAM utils (à placer une seule fois dans le fichier) ─────────────────────────
import os, time, cv2, numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Subset
from PIL import Image

COLORMAP_DICT = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
}

@torch.no_grad()
def _safe_l2(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)



def compute_gradcam_supheads(
    G, Sup, x1: torch.Tensor, *,
    feat_type: str,
    delta_weights: str,
    task_name: str,
    use_source: str = "style",   # "style" | "content" | "auto"
    level: str = "t1",
    target_class: int | None = None,
    verbose: bool = False,
):
    import cv2
    import torch
    import torch.nn.functional as F
    x1 = x1.requires_grad_(True)

    # ---------- 1) forward unique : content + style ----------
    z, skips = G.encode_content(x1)       # z, (s1..s5)
    maps, toks, tokG = G.style_enc(x1)    # maps: [m5..m1], toks: [t5..t1], tokG

    bot = F.adaptive_avg_pool2d(z, 1).flatten(1)

    # robustesse tokens
    if toks is None: toks = []
    if tokG is None:
        tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1]) if toks else bot.new_zeros(bot.shape[0], bot.shape[1])

    def _safe_l2(x): return x / (x.norm(dim=1, keepdim=True) + 1e-8)

    tlist = [_safe_l2(tokG)] + [_safe_l2(t) for t in toks]  # [tG,t5,t4,t3,t2,t1]

    ft = feat_type.strip().lower()
    if ft == "styletok": ft = "style_tok"
    if ft == "tokg":     ft = "tokg"

    def _parse_weights(dw, n, device):
        try:
            vals = [float(t) for t in str(dw).split(",") if t.strip()]
        except Exception:
            vals = []
        if len(vals) != n: vals = [1.0]*n
        w = torch.tensor(vals, dtype=torch.float32, device=device)
        return w

    # ---------- 2) features EXACTEMENT comme en sup ----------
    if ft in ("tok6", "tok6_mean", "tok6_w", "tokL", "tokL_mean", "tokL_w", "style_tok", "tokg"):
        if ft in ("style_tok", "tokg"):
            feats = tlist[0]
        elif ft == "tok6":
            # compat: tokG + 5 locaux (si dispo)
            seq = tlist[:6] if len(tlist) >= 6 else (tlist + [tlist[-1]] * (6 - len(tlist)))
            feats = torch.cat(seq, dim=1)
        elif ft == "tok6_mean":
            seq = tlist[:6] if len(tlist) >= 6 else (tlist + [tlist[-1]] * (6 - len(tlist)))
            feats = torch.stack(seq, dim=1).mean(1)
        elif ft == "tok6_w":
            seq = tlist[:6] if len(tlist) >= 6 else (tlist + [tlist[-1]] * (6 - len(tlist)))
            w = _parse_weights(delta_weights, 6, x1.device).view(1, -1, 1)
            w = w / (w.sum() + 1e-8)
            S = torch.stack(seq, dim=1)     # Bx6xD
            feats = (S * w).sum(1)          # BxD
        elif ft == "tokL":
            feats = torch.cat(tlist, dim=1)
        elif ft == "tokL_mean":
            feats = torch.stack(tlist, dim=1).mean(1)
        else:  # tokL_w
            n = len(tlist)
            w = _parse_weights(delta_weights, n, x1.device).view(1, -1, 1)
            w = w / (w.sum() + 1e-8)
            S = torch.stack(tlist, dim=1)   # BxNxD
            feats = (S * w).sum(1)          # BxD
    elif ft == "bot":
        feats = bot
    elif ft == "bot+tok":
        feats = torch.cat([bot, tlist[0]], dim=1)
    elif ft == "mgap":
        pool = lambda m: F.adaptive_avg_pool2d(m.abs(), 1).flatten(1)
        w5 = _parse_weights(delta_weights, 5, x1.device)
        if not (isinstance(maps, (list, tuple)) and len(maps) >= 5):
            feats = bot
        else:
            feats = torch.cat([wi*pool(mi) for wi, mi in zip(w5, maps[:5])], dim=1)
    elif ft == "mgap+tok":
        pool = lambda m: F.adaptive_avg_pool2d(m.abs(), 1).flatten(1)
        w5 = _parse_weights(delta_weights, 5, x1.device)
        if not (isinstance(maps, (list, tuple)) and len(maps) >= 5):
            feats = torch.cat([bot, tlist[0]], dim=1)
        else:
            mg = [wi*pool(mi) for wi, mi in zip(w5, maps[:5])]
            feats = torch.cat(mg + [tlist[0]], dim=1)
    elif ft == "tok+delta":
        pool = lambda m: F.adaptive_avg_pool2d(m.abs(), 1).flatten(1)
        w5 = _parse_weights(delta_weights, 5, x1.device)
        if not (isinstance(maps, (list, tuple)) and len(maps) >= 5):
            feats = tlist[0]
        else:
            dvec = torch.cat([wi*pool(mi) for wi, mi in zip(w5, maps[:5])], dim=1)
            feats = torch.cat([tlist[0], dvec], dim=1)
    else:
        feats = bot

    logits = Sup(feats)
    if not isinstance(logits, dict):
        logits = {"__DEFAULT__": logits}
    if task_name not in logits:
        task_name = next(iter(logits.keys()))
    out = logits[task_name]  # BxC

    if target_class is None:
        target_class = int(out.softmax(1).argmax(1).item())
    score = out[:, target_class].sum()

    # ---------- 3) choisir automatiquement une fmap connectée ----------
    cand = []

    def _append_if_tensor(t, name):
        if isinstance(t, torch.Tensor) and t.requires_grad and t.dim() >= 3:
            cand.append((t, name))

    # choix initial (respecte use_source)
    use_src = use_source.lower()
    if use_src == "auto":
        # heuristique: si features utilisent maps (mgap / mgap+tok / tok+delta) → style d'abord
        if ft in ("mgap", "mgap+tok", "tok+delta"):
            use_src = "style"
        else:
            use_src = "style"  # on tente style d'abord puis on retombe sur content

    if use_src == "style":
        if isinstance(maps, (list, tuple)) and len(maps) >= 5:
            order = ["t1","t2","t3","t4","t5"]
            idx_map = {"t5":0,"t4":1,"t3":2,"t2":3,"t1":4}
            sel = level.lower()
            # d'abord le niveau demandé
            i0 = idx_map.get(sel, 4)
            _append_if_tensor(maps[i0], f"style:{sel}")
            # puis tous les autres (t1..t5)
            for k in order:
                i = idx_map[k]
                if i != i0: _append_if_tensor(maps[i], f"style:{k}")
    else:
        _append_if_tensor(z, "content:bot")
        if isinstance(skips, (list, tuple)) and len(skips) >= 4:
            # ordre habituel: skip16, skip32, skip64
            _append_if_tensor(skips[3], "content:skip16")
            _append_if_tensor(skips[2], "content:skip32")
            _append_if_tensor(skips[1], "content:skip64")

    # fallback: si style → tenter content aussi (et inversement)
    if use_src == "style":
        _append_if_tensor(z, "content:bot")
        if isinstance(skips, (list, tuple)) and len(skips) >= 4:
            _append_if_tensor(skips[3], "content:skip16")
            _append_if_tensor(skips[2], "content:skip32")
            _append_if_tensor(skips[1], "content:skip64")
    else:
        if isinstance(maps, (list, tuple)) and len(maps) >= 5:
            for nm, i in zip(["t1","t2","t3","t4","t5"], [4,3,2,1,0]):
                _append_if_tensor(maps[i], f"style:{nm}")

    cam = None
    chosen = None
    for fmap, tag in cand:
        grads = torch.autograd.grad(score, fmap, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grads is None:
            continue
        if not torch.isfinite(grads).any() or grads.abs().sum() == 0:
            continue
        # Grad-CAM
        weights = grads.mean(dim=(2,3), keepdim=True)  # BxCx1x1
        cam_tensor = (weights * fmap).sum(1, keepdim=True).clamp(min=0)[0,0]
        # normalisation
        cam_tensor = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-8)
        H0, W0 = x1.shape[-2:]
        cam = cv2.resize(cam_tensor.detach().cpu().float().numpy(), (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        chosen = tag
        if verbose:
            print(f"[Grad-CAM] fmap choisie: {tag}")
        break

    # ---------- 4) dernier recours : saliency sur l'entrée ----------
    if cam is None:
        x1_grad = torch.autograd.grad(score, x1, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if (x1_grad is not None) and torch.isfinite(x1_grad).any():
            sal = x1_grad.detach().abs().mean(dim=1, keepdim=True)[0,0]  # HxW
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            cam = sal.cpu().numpy().astype(np.float32)
            chosen = "input-saliency"
            if verbose:
                print("[Grad-CAM] fallback → input saliency")
        else:
            # rien à afficher : renvoyer une carte nulle
            H0, W0 = x1.shape[-2:]
            cam = np.zeros((H0, W0), dtype=np.float32)
            if verbose:
                print("[Grad-CAM] aucun gradient disponible")

    return cam, target_class

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────
import os, math, glob
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# fichiers images autorisés
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _list_images_sorted(path_like: str) -> list[str]:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() in _IMG_EXTS:
        return [str(p)]
    if p.is_dir():
        files = [str(q) for q in sorted(p.rglob("*")) if q.suffix.lower() in _IMG_EXTS]
        return files
    raise FileNotFoundError(f"Chemin introuvable: {path_like}")

def _infer_img_size_from_cfg(cfg: dict, default: int = 256) -> int:
    for k in ["img_size", "image_size", "size"]:
        if k in cfg and isinstance(cfg[k], int):
            return int(cfg[k])
    # chemins type cfg["data"]["img_size"]
    try:
        for k in ["img_size", "image_size", "size"]:
            if k in cfg.get("data", {}) and isinstance(cfg["data"][k], int):
                return int(cfg["data"][k])
    except Exception:
        pass
    return default

def _make_transform(side: int | None):
    tfm = []
    if side and side > 0:
        tfm.append(T.Resize((side, side), interpolation=T.InterpolationMode.BICUBIC, antialias=True))
    tfm += [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return T.Compose(tfm)

def _load_img_tensor(path: str, tfm, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)   # 1xCxHxW in [-1,1]
    return x

@torch.no_grad()
def _save_tensor_as_image(x: torch.Tensor, path: str):
    x = x.detach().clamp(-1, 1)
    x = (x * 0.5 + 0.5)  # [0,1]
    x = x[0].cpu()       # CxHxW
    T.ToPILImage()(x).save(path)



def build_style_cond_from_img(G, imgs: torch.Tensor, *, token_gain: float = 1.0):
    """
    Recalque le format utilisé à l'entraînement: {"tokens": ((t5,g),...,(t1,g)), "token": (tG,g)}
    """
    maps, toks, tokG = G.style_enc(imgs)
    if tokG is None and toks and len(toks) > 0:
        tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])
    toks = tuple((t, token_gain) for t in (toks or []))
    tokG = (tokG, token_gain)
    return {"tokens": toks, "token": tokG}

def spectral_noise_like(x: torch.Tensor, sigma: float = 0.02, gamma: float = 1.0) -> torch.Tensor:
    """
    Petit bruit 'spectral' simple : on perturbe l'amplitude FFT puis iFFT.
    x : BxCxHxW, [-1,1]
    """
    X = torch.fft.rfft2(x, dim=(-2, -1))
    mag = torch.abs(X)
    phase = torch.angle(X)
    n = torch.randn_like(mag) * float(sigma)
    if gamma != 1.0:
        n = n.sign() * n.abs().pow(float(gamma))
    mag2 = (mag * (1.0 + n)).clamp_min(0.0)
    X2 = mag2 * torch.exp(1j * phase)
    y = torch.fft.irfft2(X2, s=x.shape[-2:])
    return y.clamp(-1, 1)




from pathlib import Path as PPath
from typing import Tuple, Optional, Dict

import torch


def load_detection_model(
    weights_dir: PPath,
    device: torch.device,
    cfg: dict,
    ckpt: Optional[str] = None,
) -> Tuple[torch.nn.Module, Optional[Dict[int, str]]]:
    """
    Chargement générique d'un modèle de détection.

    Hypothèses :
      - ckpt est un checkpoint produit par train_detection_transformer
        (avec une clé "hparams") OU un ckpt plus "brut".
      - On supporte deux types de têtes :
          * 'simple_unet' / 'unet_detr'
          * 'detr_resnet50' / 'detr'

    Priorité pour config :
      1) hparams du checkpoint (si présents)
      2) cfg["detector"][...] ou cfg["det_*"]
      3) valeurs par défaut
    """
    import torchvision
    from pathlib import Path as _Path

    # -------------------------------------------------------------------------
    # 1) Charger éventuellement le checkpoint pour récupérer les hparams
    # -------------------------------------------------------------------------
    ckpt_path = None
    ckpt_state = None
    ckpt_hparams = None

    if ckpt is not None:
        ckpt_path = _Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (weights_dir / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint détection introuvable : {ckpt_path}")

        ckpt_state = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt_state, dict) and "hparams" in ckpt_state:
            ckpt_hparams = ckpt_state["hparams"]

    # -------------------------------------------------------------------------
    # 2) Lecture de la config de détection (cfg + hparams)
    # -------------------------------------------------------------------------
    det_cfg = cfg.get("detector", {})

    # Nom de la tête de détection
    def _get_head_type():
        # priorité hparams si dispo
        if ckpt_hparams is not None and "det_head_type" in ckpt_hparams:
            return str(ckpt_hparams["det_head_type"]).lower()
        # sinon cfg
        name = (
            det_cfg.get("head_type")
            or det_cfg.get("name")
            or cfg.get("det_head_type")
            or cfg.get("detector_name")
        )
        if name is None:
            return "detr_resnet50"
        return str(name).lower()

    head_type = _get_head_type()

    # Nombre de classes
    def _get_num_classes():
        if "num_classes" in det_cfg:
            return int(det_cfg["num_classes"])
        if ckpt_hparams is not None and "num_classes" in ckpt_hparams:
            return int(ckpt_hparams["num_classes"])
        return int(cfg.get("det_num_classes", 91))

    num_classes = _get_num_classes()

    # Label map optionnel (pour affichage des catégories)
    label_map = det_cfg.get("label_map", cfg.get("det_label_map", None))

    print(
        f"[load_detection_model] Head '{head_type}' "
        f"(num_classes={num_classes})"
    )

    # -------------------------------------------------------------------------
    # 3) CAS 1 : DETR ResNet-50 torchvision
    # -------------------------------------------------------------------------
    if head_type in {"detr_resnet50", "detr"}:
        try:
            from torchvision.models.detection import detr_resnet50
        except ImportError as e:
            raise ImportError(
                "detr_resnet50 n'est pas disponible dans cette version de torchvision. "
                "Installe une version plus récente ou utilise det_head_type='simple_unet'."
            ) from e

        model = detr_resnet50(weights=None, num_classes=num_classes)

        # Chargement du state_dict si ckpt fourni
        if ckpt_state is not None:
            state = ckpt_state
            # ckpt d'entraînement detection : {"model": ..., "optimizer": ..., "hparams": ...}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]

            try:
                model.load_state_dict(state, strict=True)
                print("  ↳ Modèle DETR ResNet50 chargé (strict=True)")
            except Exception as e:
                print(f"  ! Erreur strict=True : {e}")
                model.load_state_dict(state, strict=False)
                print("  ↳ Modèle DETR ResNet50 chargé (strict=False)")

            if ckpt_path is not None:
                print(f"✓ Détecteur DETR ResNet50 chargé depuis {ckpt_path.name}")
        else:
            print(
                "✓ Détecteur DETR ResNet50 initialisé sans checkpoint "
                "(poids aléatoires ou par défaut)"
            )

        model.to(device).eval()
        return model, label_map

    # -------------------------------------------------------------------------
    # 4) CAS 2 : SimpleDETRHead + UNetGenerator (tête 'simple_unet')
    # -------------------------------------------------------------------------
    elif head_type in {"simple_unet", "unet_detr"}:
        from models.generator import UNetGenerator
        from models.det_transformer import SimpleDETRHead

        # Hyperparamètres de la tête : priorité hparams -> cfg -> défauts
        def _hp(name, default):
            if ckpt_hparams is not None and name in ckpt_hparams:
                return ckpt_hparams[name]
            if name in det_cfg:
                return det_cfg[name]
            return cfg.get(name, default)

        feat_branch = str(
            _hp("det_feat_branch", det_cfg.get("feat_branch", "content"))
        )
        num_queries = int(_hp("det_num_queries", det_cfg.get("num_queries", 300)))
        d_model = int(_hp("d_model", 256))
        nheads = int(_hp("det_nheads", det_cfg.get("nheads", 8)))
        num_decoder_layers = int(
            _hp("det_dec_layers", det_cfg.get("num_decoder_layers", 6))
        )
        token_dim = int(_hp("det_token_dim", det_cfg.get("token_dim", 256)))

        print(
            f"[load_detection_model] Head 'simple_unet' "
            f"(feat_branch={feat_branch}, num_queries={num_queries}, "
            f"d_model={d_model}, nheads={nheads}, layers={num_decoder_layers}, "
            f"token_dim={token_dim}, num_classes={num_classes})"
        )

        # Backbone UNet + tête DETR-like
        G_det = UNetGenerator(token_dim=token_dim)
        model = SimpleDETRHead(
            generator=G_det,
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=d_model,
            nheads=nheads,
            num_decoder_layers=num_decoder_layers,
            feat_branch=feat_branch,
        )

        # --------- helper : chargement avec filtrage de formes ----------------
        def _load_head_state_with_shape_filter(m: torch.nn.Module, raw_state: dict, desc: str):
            """
            Charge raw_state dans m en filtrant les clés dont la forme ne
            correspond pas aux paramètres actuels (pour encoder.proj.s4/s5, etc.).
            """
            cur_state = m.state_dict()
            filtered = {}
            skipped = []

            for k, v in raw_state.items():
                if k not in cur_state:
                    skipped.append((k, "missing_in_model"))
                    continue
                if cur_state[k].shape != v.shape:
                    skipped.append((k, f"shape_ckpt={tuple(v.shape)}, shape_model={tuple(cur_state[k].shape)}"))
                    continue
                filtered[k] = v

            m.load_state_dict(filtered, strict=False)

            if skipped:
                print(f"  ↳ {desc} : certaines clés ont été ignorées pour cause de shape mismatch ou absence :")
                for k, why in skipped:
                    print(f"     - {k}: {why}")
            else:
                print(f"  ↳ {desc} chargé sans filtrage (toutes les clés compatibles).")

        # ---------------------------------------------------------------------
        # Chargement depuis ckpt si dispo
        # ---------------------------------------------------------------------
        if ckpt_state is not None:
            state = ckpt_state
            if isinstance(state, dict):
                # 1) Backbone UNet : priorité aux clés "backbone", puis "G_B", puis "G_A"
                backbone_loaded = False
                if "backbone" in state:
                    try:
                        G_det.load_state_dict(state["backbone"], strict=False)
                        print("  ↳ Backbone UNet chargé depuis clé 'backbone'")
                        backbone_loaded = True
                    except Exception as e:
                        print(f"  ! Erreur chargement backbone : {e}")
                if not backbone_loaded and "G_B" in state:
                    try:
                        G_det.load_state_dict(state["G_B"], strict=False)
                        print("  ↳ Backbone UNet chargé depuis clé 'G_B'")
                        backbone_loaded = True
                    except Exception as e:
                        print(f"  ! Erreur chargement G_B : {e}")
                if not backbone_loaded and "G_A" in state:
                    try:
                        G_det.load_state_dict(state["G_A"], strict=False)
                        print("  ↳ Backbone UNet chargé depuis clé 'G_A'")
                        backbone_loaded = True
                    except Exception as e:
                        print(f"  ! Erreur chargement G_A : {e}")
                if not backbone_loaded:
                    print("  (i) Aucun backbone explicite trouvé, G_det reste initialisé.")

                # 2) Tête de détection
                if "det_head" in state:
                    raw_head = state["det_head"]
                    if not isinstance(raw_head, dict):
                        print("  ! det_head n'est pas un dict, tentative de chargement direct.")
                        raw_head = state["det_head"].state_dict() if hasattr(state["det_head"], "state_dict") else {}
                    _load_head_state_with_shape_filter(model, raw_head, "Tête SimpleDETRHead (det_head)")
                elif "model" in state:
                    raw_head = state["model"]
                    if not isinstance(raw_head, dict):
                        raw_head = raw_head.state_dict() if hasattr(raw_head, "state_dict") else {}
                    _load_head_state_with_shape_filter(model, raw_head, "Tête SimpleDETRHead (model)")
                else:
                    # Dernier fallback : on suppose que tout le dict est un state_dict de model
                    raw_head = state
                    if not isinstance(raw_head, dict):
                        raw_head = raw_head.state_dict() if hasattr(raw_head, "state_dict") else {}
                    _load_head_state_with_shape_filter(model, raw_head, "Tête SimpleDETRHead (state direct)")
            else:
                # state non dict → on tente direct (avec filtrage minimal)
                raw_head = state
                if not isinstance(raw_head, dict):
                    raw_head = raw_head.state_dict() if hasattr(raw_head, "state_dict") else {}
                _load_head_state_with_shape_filter(model, raw_head, "Tête SimpleDETRHead (state non-dict)")

            if ckpt_path is not None:
                print(f"✓ Détecteur SimpleDETRHead+UNet chargé depuis {ckpt_path.name}")
        else:
            print(
                "✓ Détecteur SimpleDETRHead+UNet initialisé sans checkpoint "
                "(poids aléatoires ou par défaut)"
            )

        model.to(device).eval()
        return model, label_map

    # -------------------------------------------------------------------------
    # CAS inconnu
    # -------------------------------------------------------------------------
    else:
        raise ValueError(
            f"[load_detection_model] Modèle de détection inconnu: '{head_type}'. "
            "Utilise 'detr_resnet50' ou 'simple_unet' (unet_detr), "
            "ou adapte cette fonction à ton architecture."
        )

def build_detection_dataloader(opt, cfg) -> torch.utils.data.DataLoader:
    """
    Construction d'un DataLoader pour la détection.

    Cette implémentation est volontairement GENERIQUE :
    elle s'attend à ce que cfg['detector']['dataset'] te donne de quoi
    construire un Dataset torch qui renvoie (img, target) avec :
        target = {"boxes": Tensor[N,4] (xyxy), "labels": Tensor[N]}
    ou que tu adaptes directement cette fonction à ton dataset.

    Exemple d'adaptation :
      - utiliser torchvision.datasets.CocoDetection
      - utiliser ton propre Dataset COCO-style
    """
    from torch.utils.data import DataLoader

    det_cfg = cfg.get("detector", {})
    ds = det_cfg.get("dataset", None)
    if ds is None:
        raise RuntimeError(
            "[build_detection_dataloader] Aucun dataset de détection configuré.\n"
            "Adapte cette fonction pour ton cas (COCO, BDD, etc.)."
        )

    # Ici, on suppose que `dataset` est déjà un objet Dataset prêt à l'emploi
    # que tu as injecté dans la config (ou que tu reconstruis ici).
    dataset = ds

    bs = getattr(opt, "bs", getattr(opt, "batch_size", 1))
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=getattr(opt, "num_workers", 4),
        drop_last=False, collate_fn=getattr(dataset, "collate_fn", lambda x: list(zip(*x)))
    )
    return loader


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre deux ensembles de boxes (xyxy).
    Retours : matrice [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def compute_detection_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
    max_dets: int = 100,
) -> Dict[str, float]:
    """
    Métriques de détection simples, type AP@IoU + precision/recall/F1 globales.

    Hypothèses :
      - data_loader renvoie batches (images, targets)
        * images: list[Tensor] ou Tensor[B,3,H,W]
        * targets: list[dict] avec 'boxes' (N,4) xyxy et 'labels' (N,)
      - model(images) renvoie list[dict] au même format (boxes, labels, scores).

    On calcule :
      - AP@IoU (unique seuil, style VOC)
      - precision, recall, F1 globaux (toutes classes confondues)
    """
    import numpy as np

    model.eval()
    all_scores: List[float] = []
    all_tp: List[int] = []
    all_fp: List[int] = []
    n_gt = 0

    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch
            if isinstance(images, torch.Tensor):
                images = list(img.to(device) for img in images)
            else:
                images = [img.to(device) for img in images]

            outputs = model(images)

            if not isinstance(outputs, (list, tuple)):
                raise RuntimeError(
                    "[compute_detection_metrics] Le modèle doit renvoyer une liste de dicts "
                    "avec 'boxes', 'labels', 'scores'."
                )

            for pred, tgt in zip(outputs, targets):
                boxes_gt = tgt["boxes"].to(device)
                labels_gt = tgt["labels"].to(device)
                n_gt += boxes_gt.size(0)

                boxes_pred = pred["boxes"].to(device)
                scores_pred = pred["scores"].to(device)
                labels_pred = pred["labels"].to(device)

                # filtrage par score + max_dets
                keep = scores_pred >= score_thresh
                boxes_pred = boxes_pred[keep]
                scores_pred = scores_pred[keep]
                labels_pred = labels_pred[keep]

                if boxes_pred.size(0) > max_dets:
                    order = scores_pred.argsort(descending=True)[:max_dets]
                    boxes_pred = boxes_pred[order]
                    scores_pred = scores_pred[order]
                    labels_pred = labels_pred[order]

                if boxes_pred.numel() == 0:
                    continue

                matched_gt = torch.zeros(boxes_gt.size(0), dtype=torch.bool, device=device)

                for b_pred, s_pred, l_pred in zip(boxes_pred, scores_pred, labels_pred):
                    if boxes_gt.numel() == 0:
                        all_scores.append(float(s_pred.cpu()))
                        all_tp.append(0)
                        all_fp.append(1)
                        continue

                    same_class = labels_gt == l_pred
                    if not torch.any(same_class):
                        all_scores.append(float(s_pred.cpu()))
                        all_tp.append(0)
                        all_fp.append(1)
                        continue

                    ious = _box_iou_xyxy(b_pred[None, :], boxes_gt[same_class]).squeeze(0)
                    max_iou, idx_local = ious.max(dim=0)
                    idx_global = torch.nonzero(same_class, as_tuple=False)[idx_local, 0]

                    if max_iou >= iou_thresh and not matched_gt[idx_global]:
                        matched_gt[idx_global] = True
                        all_scores.append(float(s_pred.cpu()))
                        all_tp.append(1)
                        all_fp.append(0)
                    else:
                        all_scores.append(float(s_pred.cpu()))
                        all_tp.append(0)
                        all_fp.append(1)

    if n_gt == 0 or len(all_scores) == 0:
        return {
            "AP": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0,
            "n_gt": float(n_gt),
            "n_pred": float(len(all_scores)),
        }

    scores_np = np.array(all_scores, dtype=np.float32)
    tp_np = np.array(all_tp, dtype=np.int32)
    fp_np = np.array(all_fp, dtype=np.int32)

    order = np.argsort(-scores_np)
    tp_cum = np.cumsum(tp_np[order])
    fp_cum = np.cumsum(fp_np[order])

    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # AP (approx VOC / PR AUC)
    # on enforce un enveloppe monotone
    precisions_envelope = precisions.copy()
    for i in range(len(precisions_envelope) - 2, -1, -1):
        precisions_envelope[i] = max(precisions_envelope[i], precisions_envelope[i + 1])
    # intégrale discrète
    AP = float(np.trapz(precisions_envelope, recalls))

    # point unique au seuil fourni
    tp_total = tp_np.sum()
    fp_total = fp_np.sum()
    fn_total = max(n_gt - tp_total, 0)
    precision = float(tp_total / max(tp_total + fp_total, 1))
    recall = float(tp_total / max(tp_total + fn_total, 1))
    if precision + recall > 0:
        F1 = float(2 * precision * recall / (precision + recall))
    else:
        F1 = 0.0

    return {
        "AP": AP,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "n_gt": float(n_gt),
        "n_pred": float(len(all_scores)),
    }


def run_detection_on_camera(
    model: torch.nn.Module,
    device: torch.device,
    *,
    label_map: Optional[Dict[int, str]] = None,
    score_thresh: float = 0.5,
    cam_index: int = 0,
):
    """
    Boucle webcam pour un modèle de détection :

    - Modèle type TorchVision DETR : sortie = [ { 'boxes','scores','labels', ... } ]
    - Modèle type SimpleDETRHead   : sortie = (pred_logits, pred_boxes)
        * pred_logits : (B,Q,num_classes)
        * pred_boxes  : (B,Q,4) en cx,cy,w,h normalisés [0,1]
    """
    import cv2
    import numpy as np
    import time
    import torch

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index={cam_index}")

    model.eval()
    print(
        f"[run_detection_on_camera] Caméra {cam_index} ouverte. "
        f"Seuil score={score_thresh:.2f}. Appuyez sur 'q' pour quitter."
    )
    last_t = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            H, W = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
            img = img.to(device)

            with torch.no_grad():
                raw_out = model([img])

            # ------------------------------------------------------------------
            # 1) Cas TorchVision DETR : list[dict]
            # ------------------------------------------------------------------
            boxes = torch.empty((0, 4), device=device)
            scores = torch.empty((0,), device=device)
            labels = torch.empty((0,), device=device, dtype=torch.int64)

            if isinstance(raw_out, (list, tuple)) and len(raw_out) > 0 and isinstance(raw_out[0], dict):
                det = raw_out[0]
                boxes = det.get("boxes", boxes)
                scores = det.get("scores", scores)
                labels = det.get("labels", labels)

            # ------------------------------------------------------------------
            # 2) Cas SimpleDETRHead : (pred_logits, pred_boxes)
            # ------------------------------------------------------------------
            elif (
                isinstance(raw_out, (list, tuple))
                and len(raw_out) == 2
                and torch.is_tensor(raw_out[0])
                and torch.is_tensor(raw_out[1])
            ):
                pred_logits, pred_boxes = raw_out  # (B,Q,C), (B,Q,4)
                # On suppose B=1 pour la caméra
                logits = pred_logits[0]       # (Q,C)
                box_norm = pred_boxes[0]      # (Q,4) en cx,cy,w,h dans [0,1]

                # Scores + labels
                probs = logits.softmax(-1)    # (Q,C)
                scores, labels = probs.max(-1)  # (Q,), (Q,)

                # On considère label 0 comme "fond" -> on l'ignore si présent
                # (adapter si ta convention est différente)
                if labels.numel() > 0 and labels.min().item() == 0:
                    fg = labels > 0
                    scores = scores[fg]
                    box_norm = box_norm[fg]
                    labels = labels[fg]

                # Conversion cx,cy,w,h (normalisés) -> x1,y1,x2,y2 (pixels)
                if box_norm.numel() > 0:
                    cx = box_norm[:, 0] * W
                    cy = box_norm[:, 1] * H
                    bw = box_norm[:, 2] * W
                    bh = box_norm[:, 3] * H

                    x1 = cx - bw / 2.0
                    y1 = cy - bh / 2.0
                    x2 = cx + bw / 2.0
                    y2 = cy + bh / 2.0

                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            # ------------------------------------------------------------------
            # 3) Cas dict direct (au cas où)
            # ------------------------------------------------------------------
            elif isinstance(raw_out, dict):
                boxes = raw_out.get("boxes", boxes)
                scores = raw_out.get("scores", scores)
                labels = raw_out.get("labels", labels)

            else:
                # Modèle non supporté
                raise RuntimeError(
                    f"run_detection_on_camera: type de sortie non supporté ({type(raw_out)})"
                )

            # ------------------------------------------------------------------
            # Filtrage par score et passage en numpy
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # Filtrage par score et passage en numpy
            # ------------------------------------------------------------------
            if scores.numel() > 0:
                # (1) Pour debug : on ne filtre que par score,
                #     sans supprimer les labels == 0.
                keep = scores >= score_thresh

                # Debug : affiche quelques scores/labels bruts
                print(
                    f"[DEBUG] nb_queries={scores.numel()}, "
                    f"nb_keep={(keep.float().sum().item())}, "
                    f"top5_scores={scores.topk(min(5, scores.numel())).values.cpu().numpy()}"
                )

                boxes = boxes[keep].cpu().numpy().astype(np.float32)
                scores = scores[keep].cpu().numpy().astype(np.float32)
                labels = labels[keep].cpu().numpy().astype(np.int32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int32)

            vis = frame.copy()
            for b, s, lab in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b
                cv2.rectangle(
                    vis,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                if label_map is not None and lab in label_map:
                    txt = f"{label_map[lab]} {s:.2f}"
                else:
                    txt = f"id{lab} {s:.2f}"
                cv2.putText(
                    vis,
                    txt,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            cv2.putText(
                vis,
                f"{fps:.1f} FPS",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# =========================================================
#  FIXED: robust autoload of generator architecture + tokL support
#  (This definition overrides any earlier broken load_models.)
# =========================================================

from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple


def _load_train_cfg_hparams(wdir: PPath) -> _Dict[str, _Any]:
    """Load train_cfg.json (or hyperparameters.json/hparams.json) if present in weights_dir."""
    for name in ("train_cfg.json", "hyperparameters.json", "hparams.json"):
        p = PPath(wdir) / name
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                # some runs store hparams under static_hparams
                if isinstance(obj, dict) and isinstance(obj.get("static_hparams"), dict):
                    return obj["static_hparams"]
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def _infer_img_size(hp: _Dict[str, _Any], cfg: dict) -> int:
    for k in ("img_size", "crop_size", "load_size", "resize", "image_size"):
        v = hp.get(k, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    # fallback to cfg
    for k in ("img_size", "crop_size", "load_size"):
        v = cfg.get(k, None) if isinstance(cfg, dict) else None
        if isinstance(v, int) and v > 0:
            return int(v)
    return 256


def _infer_gen_kwargs(weights_dir: PPath, cfg: dict) -> _Dict[str, _Any]:
    hp = _load_train_cfg_hparams(weights_dir)

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    # token_dim
    token_dim = hp.get("token_dim", hp.get("hid_dim", model_cfg.get("token_dim", 256)))
    token_dim = int(token_dim) if isinstance(token_dim, int) and token_dim > 0 else 256

    arch_depth_delta = int(hp.get("arch_depth_delta", model_cfg.get("arch_depth_delta", 0) or 0))
    style_token_levels = int(hp.get("style_token_levels", model_cfg.get("style_token_levels", -1) or -1))
    unet_min_spatial = int(hp.get("unet_min_spatial", model_cfg.get("unet_min_spatial", 2) or 2))
    # NOTE: older experimental configs may contain "stop_down_at".
    # The current UNetGenerator API uses `unet_min_spatial` and an internal rule
    # (stop at >=4 when deepening). We intentionally ignore "stop_down_at" here.

    img_size = int(hp.get("img_size", _infer_img_size(hp, cfg)))

    norm_variant = str(hp.get("norm_variant", "legacy") or "legacy")
    extra_bot_resblocks = int(hp.get("extra_bot_resblocks", 0) or 0)

    return dict(
        token_dim=token_dim,
        arch_depth_delta=arch_depth_delta,
        style_token_levels=style_token_levels,
        img_size=img_size,
        unet_min_spatial=unet_min_spatial,
        norm_variant=norm_variant,
        extra_bot_resblocks=extra_bot_resblocks,
        use_res_skip_bot=bool(hp.get("use_res_skip_bot", False)),
        style_tokg_head_variant=str(hp.get("style_tokg_head_variant", "tokG_head") or "tokG_head"),
    )


def _read_state_any(path: PPath, dev: torch.device) -> _Dict[str, torch.Tensor]:
    path = PPath(path)
    # Robustness: if checkpoint was saved on CUDA but current machine has no CUDA,
    # always map storages to CPU to avoid:
    #   RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
    if isinstance(dev, torch.device) and dev.type == "cuda" and (not torch.cuda.is_available()):
        dev = torch.device("cpu")
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device=str(dev))
    obj = torch.load(path, map_location=dev)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # sometimes it's already a flat state dict but with extra keys
        if all(isinstance(k, str) for k in obj.keys()) and any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore
    return obj  # type: ignore


def _align_dp_prefix(sd: _Dict[str, torch.Tensor], model: torch.nn.Module) -> _Dict[str, torch.Tensor]:
    if not sd:
        return sd
    m_keys = list(model.state_dict().keys())
    model_has_mod = any(k.startswith("module.") for k in m_keys)
    ckpt_has_mod = any(k.startswith("module.") for k in sd.keys())
    if ckpt_has_mod and (not model_has_mod):
        return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    if (not ckpt_has_mod) and model_has_mod:
        return { (f"module.{k}" if not k.startswith("module.") else k): v for k, v in sd.items() }
    return sd


def _split_sup_from_ckpt(sd: _Dict[str, torch.Tensor]) -> _Tuple[_Dict[str, torch.Tensor], _Dict[str, torch.Tensor]]:
    """Split generator keys vs sup_heads keys if ckpt is a combined bundle."""
    gen_sd: _Dict[str, torch.Tensor] = {}
    sup_sd: _Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("sup_heads."):
            sup_sd[k.replace("sup_heads.", "", 1)] = v
        else:
            gen_sd[k] = v
    return gen_sd, sup_sd


def load_models(
    weights_dir: PPath,
    device: torch.device,
    cfg: dict,
    ckpt_gen: _Optional[str] = None,
    sup_ckpt: _Optional[str] = None,
    classes_json: _Optional[str] = None,
    sup_in_dim: _Optional[int] = None,
    *,
    strict_gen: bool = True,
    strict_sup: bool = True,
    ckpt_GA: _Optional[str] = None,
    ckpt_GB: _Optional[str] = None,
) -> _Tuple[torch.nn.Module, _Optional[torch.nn.Module], _Optional[dict]]:
    """Robust loader used by test.py.

    - Autoload generator architecture (arch_depth_delta/style_token_levels/img_size/etc.) from weights_dir config.
    - Supports separate GA/GB ckpts.
    - Loads SupHeads if provided or present inside generator ckpt.

    Returns: (G, SupHeads_or_None, task_classes_or_None)
    """
    from models.generator import UNetGenerator
    from models.sup_heads import SupHeads

    weights_dir = PPath(weights_dir)
    # CPU-only safety: never move modules to CUDA when CUDA is unavailable
    if device is None:
        device = torch.device("cpu")
    dev_str = str(device)
    if dev_str.startswith("cuda") and (not torch.cuda.is_available()):
        print(f"[WARN] test device={device} requested but CUDA is unavailable. Using CPU.")
        device = torch.device("cpu")

    # choose generator ckpt
    if ckpt_gen is None:
        # try cfg hints
        ckpt_name = None
        if isinstance(cfg, dict):
            ckpt_name = (cfg.get("model", {}) or {}).get("gen_ckpt_best") or (cfg.get("model", {}) or {}).get("gen_ckpt_last")
        if ckpt_name:
            ckpt_gen = str(weights_dir / ckpt_name)
        else:
            ckpt_gen = str(find_latest_ckpt(weights_dir))

    ckpt_gen_p = PPath(ckpt_gen)
    if not ckpt_gen_p.exists():
        raise FileNotFoundError(f"Generator ckpt not found: {ckpt_gen_p}")

    gen_kwargs = _infer_gen_kwargs(weights_dir, cfg)

    # read checkpoint first (to possibly infer norm variant / extra blocks)
    full_sd = _read_state_any(ckpt_gen_p, device)
    try:
        full_sd = _remap_keys(full_sd)
    except Exception:
        pass

    gen_sd, sup_sd_from_gen = _split_sup_from_ckpt(full_sd)

    # Infer compatibility if not stored in train_cfg.json
    keys = list(gen_sd.keys())
    if str(gen_kwargs.get("norm_variant", "legacy") or "legacy") == "legacy":
        if any(".inorm." in k or ".gnorm." in k for k in keys):
            gen_kwargs["norm_variant"] = "safe"
    extra = int(gen_kwargs.get("extra_bot_resblocks", 0) or 0)
    if any(k.startswith("res5.") for k in keys):
        extra = max(extra, 1)
    if any(k.startswith("res6.") for k in keys):
        extra = max(extra, 2)
    gen_kwargs["extra_bot_resblocks"] = extra

    # Infer legacy optional blocks (res_skip/res_bot) and style tokG head variant
    gen_kwargs["use_res_skip_bot"] = any(k.startswith("res_skip.") or k.startswith("res_bot.") for k in keys)
    if any(k.startswith("style_enc.tbot.") for k in keys):
        gen_kwargs["style_tokg_head_variant"] = "tbot"
    else:
        gen_kwargs["style_tokg_head_variant"] = "tokG_head"

    # build generator with final kwargs
    G = UNetGenerator(**gen_kwargs).to(device)
    gen_sd = _align_dp_prefix(gen_sd, G)

    # load generator
    try:
        G.load_state_dict(gen_sd, strict=strict_gen)
    except RuntimeError as e:
        if strict_gen:
            raise RuntimeError(f"[STRICT] Generator load failed for {ckpt_gen_p.name}:\n{e}")
        G.load_state_dict(gen_sd, strict=False)

    G.eval()

    # optional separate GA/GB
    def _load_one_gen(pth: str) -> torch.nn.Module:
        p = PPath(pth)
        if not p.exists():
            raise FileNotFoundError(f"Generator ckpt not found: {p}")
        Gx = UNetGenerator(**gen_kwargs).to(device)
        sd = _read_state_any(p, device)
        try:
            sd = _remap_keys(sd)
        except Exception:
            pass
        gen_sd2, _ = _split_sup_from_ckpt(sd)
        gen_sd2 = _align_dp_prefix(gen_sd2, Gx)
        try:
            Gx.load_state_dict(gen_sd2, strict=strict_gen)
        except RuntimeError as e:
            if strict_gen:
                raise RuntimeError(f"[STRICT] Generator load failed for {p.name}:\n{e}")
            Gx.load_state_dict(gen_sd2, strict=False)
        Gx.eval()
        return Gx

    if ckpt_GA:
        G.GA = _load_one_gen(ckpt_GA)
    if ckpt_GB:
        G.GB = _load_one_gen(ckpt_GB)

    # task classes (optional metadata: task -> list of class names or {classes:[...]})
    task_classes: dict | None = None
    if classes_json:
        try:
            task_classes = json.loads(PPath(classes_json).read_text())
        except Exception:
            task_classes = None

    def _tasks_from_task_classes(tc: dict) -> dict[str, int] | None:
        """Convert task_classes json to {task: num_classes}."""
        if not isinstance(tc, dict):
            return None
        out: dict[str, int] = {}
        for t, v in tc.items():
            if isinstance(v, int):
                out[str(t)] = int(v)
            elif isinstance(v, list):
                out[str(t)] = len(v)
            elif isinstance(v, dict):
                if "classes" in v and isinstance(v["classes"], list):
                    out[str(t)] = len(v["classes"])
                elif "num_classes" in v:
                    try:
                        out[str(t)] = int(v["num_classes"])
                    except Exception:
                        pass
        return out or None

    def _tasks_from_sup_state(sd: dict) -> dict[str, int] | None:
        """Infer {task:num_classes} from SupHeads state_dict keys."""
        if not isinstance(sd, dict):
            return None
        out: dict[str, int] = {}
        # expected key: classifiers.<task>.4.weight where last Linear is index 4
        for k, v in sd.items():
            if not isinstance(k, str):
                continue
            if not k.startswith("classifiers."):
                continue
            parts = k.split(".")
            if len(parts) < 4:
                continue
            task = parts[1]
            layer_idx = parts[2]
            param = parts[3]
            if layer_idx != "4" or param != "weight":
                continue
            if hasattr(v, "shape") and len(v.shape) >= 1:
                out[task] = int(v.shape[0])
        return out or None

    # load SupHeads if requested
    Sup = None
    if sup_ckpt or sup_sd_from_gen:
        # infer tasks from (1) cfg (2) classes_json (3) sup checkpoint/state_dict
        tasks_dict: dict[str, int] | None = None
        if isinstance(cfg, dict):
            tcfg = cfg.get("tasks") or (cfg.get("model", {}) or {}).get("tasks")
            if isinstance(tcfg, dict) and all(isinstance(v, int) for v in tcfg.values()):
                tasks_dict = {str(k): int(v) for k, v in tcfg.items()}
        if tasks_dict is None and task_classes is not None:
            tasks_dict = _tasks_from_task_classes(task_classes)

        # infer in_dim
        if sup_in_dim is None:
            # try feat_type from cfg/hparams
            hp = _load_train_cfg_hparams(weights_dir)
            feat_type = hp.get("sup_feat_type", (cfg.get("model", {}) or {}).get("sup_feat_type", None))
            if isinstance(feat_type, str) and hasattr(G, "sup_in_dim_for"):
                try:
                    sup_in_dim = int(G.sup_in_dim_for(feat_type))
                except Exception:
                    sup_in_dim = None
        if sup_in_dim is None:
            # safe fallback
            sup_in_dim = int(gen_kwargs.get("token_dim", 256))

        # If still unknown, infer tasks from sup weights before instantiation
        if tasks_dict is None:
            if sup_ckpt:
                sup_p0 = PPath(sup_ckpt)
                if sup_p0.is_dir():
                    sup_p0 = find_latest_ckpt(sup_p0)
                sd0 = _read_state_any(sup_p0, device)
                if isinstance(sd0, dict) and "sup_heads" in sd0 and isinstance(sd0["sup_heads"], dict):
                    sd0 = sd0["sup_heads"]
                if isinstance(sd0, dict) and "state_dict" in sd0 and isinstance(sd0["state_dict"], dict):
                    sd0 = sd0["state_dict"]
                if isinstance(sd0, dict) and any(k.startswith("sup_heads.") for k in sd0.keys()):
                    sd0 = {k.replace("sup_heads.", "", 1): v for k, v in sd0.items()}
                tasks_dict = _tasks_from_sup_state(sd0)
            else:
                tasks_dict = _tasks_from_sup_state(sup_sd_from_gen)

        if tasks_dict is None:
            raise RuntimeError(
                "SupHeads requested (tsne_use_supheads/per_task/sup_predict) but tasks could not be inferred. "
                "Provide --classes_json (task->classes) or ensure cfg contains tasks mapping."
            )

        # Instantiate SupHeads with correct signature: SupHeads(tasks, in_dim, ...)
        Sup = SupHeads(tasks_dict, int(sup_in_dim)).to(device)

        # load state
        if sup_ckpt:
            sup_p = PPath(sup_ckpt)
            if sup_p.is_dir():
                sup_p = find_latest_ckpt(sup_p)
            sd = _read_state_any(sup_p, device)
            # bundles may store under sup_heads
            if isinstance(sd, dict) and "sup_heads" in sd and isinstance(sd["sup_heads"], dict):
                sd = sd["sup_heads"]
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            if isinstance(sd, dict) and any(k.startswith("sup_heads.") for k in sd.keys()):
                # strip prefix
                sd = {k.replace("sup_heads.", "", 1): v for k, v in sd.items()}
            sd = _align_dp_prefix(sd, Sup)
            try:
                Sup.load_state_dict(sd, strict=strict_sup)
            except RuntimeError as e:
                if strict_sup:
                    raise RuntimeError(f"[STRICT] SupHeads load failed for {sup_p.name}:\n{e}")
                Sup.load_state_dict(sd, strict=False)
        else:
            sd = _align_dp_prefix(sup_sd_from_gen, Sup)
            try:
                Sup.load_state_dict(sd, strict=strict_sup)
            except RuntimeError as e:
                if strict_sup:
                    raise RuntimeError(f"[STRICT] SupHeads load failed from generator ckpt:\n{e}")
                Sup.load_state_dict(sd, strict=False)

        Sup.eval()

    return G, Sup, task_classes
