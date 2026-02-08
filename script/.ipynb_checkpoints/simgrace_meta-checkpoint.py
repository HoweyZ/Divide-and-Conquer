# semiSupervised/simgrace_meta.py
import os
import copy
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torch.nn.functional as F
from data.data_loading import DataLoader, load_dataset
from semiSupervised.parser import get_parser
from semiSupervised.evaluate_embedding import evaluate_embedding
from semiSupervised.mask_generator import MaskSimclr, SampleSelector
import json
from datetime import datetime
import re

# =========================
# Logging / IO
# =========================

def setup_run_dir_and_logger(args):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", str(getattr(args, "exp_name", "default")), f"seed{args.seed}", ts)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run.log")

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    import sys
    f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)

    print(f"[LOG] run_dir={run_dir}")
    print(f"[LOG] log_path={log_path}")
    return run_dir


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2, sort_keys=True)


def spaces_to_serializable(spaces: dict):
    def _convert(x):
        if isinstance(x, set):
            return sorted(list(x))
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_convert(v) for v in x]
        return x

    out = _convert(spaces)

    stats = {}
    if isinstance(spaces, dict):
        for k, v in spaces.items():
            if isinstance(v, set):
                stats[k] = len(v)
        hs = spaces.get("_head_sets", None)
        if isinstance(hs, dict):
            stats["_head_sets"] = {kk: (len(vv) if isinstance(vv, set) else None) for kk, vv in hs.items()}

    if isinstance(out, dict):
        out["_stats"] = stats
    return out


def save_topk_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "score"])
        for n, s in rows:
            w.writerow([n, float(s)])


def freeze_params_by_space(mask_simclr_module, spaces: dict, freeze_space: str):
   
    freeze_space = str(freeze_space).strip().upper()
    if freeze_space in ("", "NONE", "NO", "0"):
        freeze_set = set()
    elif freeze_space == "ALL":
        freeze_set = set(spaces["A"]) | set(spaces["B"]) | set(spaces["C"]) | set(spaces["D"])
    elif freeze_space in ("A", "B", "C", "D"):
        freeze_set = set(spaces[freeze_space])
    else:
        raise ValueError(f"freeze_space must be one of none/A/B/C/D/all, got {freeze_space}")

    frozen, trainable = 0, 0
    for name, p in mask_simclr_module.named_parameters():
        if name in freeze_set:
            p.requires_grad = False
            frozen += p.numel()
        else:
            p.requires_grad = True
            trainable += p.numel()

    print(f"[ABLATION-FREEZE] freeze_space={freeze_space} frozen_params={frozen} trainable_params={trainable}")
    return freeze_set


# =========================
# QKV utilities
# =========================

def apply_topk_mask_to_heatmap(H: np.ndarray, k: int = None, ratio: float = None, mode: str = "per_row"):
    H = np.asarray(H, dtype=np.float32)
    rows, dim = H.shape
    if (k is None) and (ratio is None):
        return H

    if k is None:
        ratio = float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        k = int(round(dim * ratio))
        k = max(1, k)

    k = int(k)
    k = max(1, min(dim, k))

    out = np.zeros_like(H, dtype=np.float32)

    if mode == "global":
        flat = np.abs(H).reshape(-1)
        if flat.size == 0:
            return out
        kk = max(1, min(flat.size, k))
        idx = np.argpartition(flat, -kk)[-kk:]
        out.reshape(-1)[idx] = H.reshape(-1)[idx]
        return out

    for r in range(rows):
        a = np.abs(H[r])
        idx = np.argpartition(a, -k)[-k:]
        out[r, idx] = H[r, idx]
    return out


def apply_percentile_binary_mask_to_heatmap(H: np.ndarray, keep_percentile: float = 85.0, mode: str = "global"):
    H = np.asarray(H, dtype=np.float32)
    out = np.zeros_like(H, dtype=np.float32)

    keep_percentile = float(keep_percentile)
    keep_percentile = max(0.0, min(100.0, keep_percentile))

    A = np.abs(H)

    def _binary_by_thresh(x, thr):
        return (x >= thr).astype(np.float32)

    if mode == "per_row":
        for r in range(A.shape[0]):
            vals = A[r].reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if float(np.max(vals)) <= 0.0:
                continue
            thr = float(np.percentile(vals, keep_percentile))
            if thr <= 0.0:
                thr = float(np.nextafter(0, 1))
            out[r] = _binary_by_thresh(A[r], thr)
        return out

    vals = A.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    if float(np.max(vals)) <= 0.0:
        return out
    thr = float(np.percentile(vals, keep_percentile))
    if thr <= 0.0:
        thr = float(np.nextafter(0, 1))
    out = _binary_by_thresh(A, thr)
    return out


_Q_RE = re.compile(r"\.attention\.Q\.(weight|bias)$")
_K_RE = re.compile(r"\.attention\.K\.(weight|bias)$")
_V_RE = re.compile(r"\.attention\.V\.(weight|bias)$")


def _qkv_tag_from_param_name(name: str):
    if _Q_RE.search(name):
        return "Q"
    if _K_RE.search(name):
        return "K"
    if _V_RE.search(name):
        return "V"
    return None


def _layer_key_from_param_name(name: str):
    marker = ".attention."
    if marker not in name:
        return None
    return name.split(marker)[0]


# =========================
# Plot 
# =========================

def plot_qkv_2x2_grid_png(
    path,
    heatmaps: dict,
    dim: int,
    row_names=("Q", "K", "V"),
    order=("Only_structure", "Only_semantic", "Overlap", "Other"),
    titles=None,
    cmap="Greens",
    dpi: int = 600,
    gap_lw: float = 0.6,
    use_log1p: bool = True,
    shared_norm: bool = False,
    unify_colorbar: bool = True,
    colorbar_single: bool = True,
    vmax_percentile: float = 100.0,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import numpy as np
        import matplotlib.ticker as mticker
        from matplotlib import rcParams
    except Exception as e:
        print(f"[HEATMAP] matplotlib not available, skip png. err={e}")
        return

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    rcParams['font.size'] = 9

    if titles is None:
        titles = {
            "Only_structure": "Structure",
            "Only_semantic": "Semantic",
            "Overlap": "Overlap",
            "Other": "Other"
        }

    keys = list(order)

    mats2d = {}
    for k in keys:
        H = np.asarray(heatmaps[k], dtype=np.float32)
        if H.shape != (len(row_names), dim):
            raise ValueError(f"heatmap[{k}].shape={H.shape}, expected {(len(row_names), dim)}")
        if use_log1p:
            H = np.log1p(np.maximum(H, 0.0))
        mats2d[k] = H

    def _safe_vmax_from_flat(flat: np.ndarray, pct: float):
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return 1e-6
        vmax = float(np.max(flat)) if float(pct) >= 100.0 else float(np.percentile(flat, float(pct)))
        if not np.isfinite(vmax) or vmax <= 0:
            return 1e-6
        return vmax

    if colorbar_single:
        unify_colorbar = True

    shared_norm_obj = None
    if unify_colorbar or shared_norm:
        flat = np.concatenate([mats2d[k].reshape(-1) for k in keys], axis=0)
        vmax = _safe_vmax_from_flat(flat, vmax_percentile)
        shared_norm_obj = Normalize(vmin=0.0, vmax=vmax)

    def panel_norm(H):
        if shared_norm_obj is not None:
            return shared_norm_obj
        vmax = float(np.nanmax(H)) if np.isfinite(np.nanmax(H)) else 1e-6
        if vmax <= 0:
            vmax = 1e-6
        return Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[0.15, 1, 1, 0.08],
        height_ratios=[1, 1],
        wspace=0.15,
        hspace=0.35,
        left=0.08, right=0.95, top=0.93, bottom=0.12
    )

    ax_label_top = fig.add_subplot(gs[0, 0])
    ax_label_bot = fig.add_subplot(gs[1, 0])

    ax00 = fig.add_subplot(gs[0, 1])
    ax01 = fig.add_subplot(gs[0, 2], sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 1], sharex=ax00)
    ax11 = fig.add_subplot(gs[1, 2], sharey=ax10, sharex=ax01)

    if colorbar_single:
        cax = fig.add_subplot(gs[:, 3])
    else:
        cax_top = fig.add_subplot(gs[0, 3])
        cax_bot = fig.add_subplot(gs[1, 3])

    def draw_one(ax, H, title, show_xlabel=False):
        norm = panel_norm(H)
        ax.imshow(H, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_title(title, pad=6)
        ax.set_yticks([])

        if show_xlabel:
            step = 4 if dim <= 32 else (8 if dim <= 64 else (16 if dim <= 128 else 32))
            xticks = list(range(0, dim, step))
            if dim - 1 not in xticks:
                xticks.append(dim - 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks], fontsize=7)
            ax.set_xlabel("Dimension", fontsize=8, labelpad=3)
            ax.tick_params(axis='x', which='both', length=0, width=0)
        else:
            ax.set_xticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if gap_lw > 0:
            ax.set_xticks(np.arange(H.shape[1]) - 0.5, minor=True)
            ax.set_yticks(np.arange(H.shape[0]) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=gap_lw)
            ax.tick_params(which="minor", size=0)

        return norm

    n00 = draw_one(ax00, mats2d[keys[0]], titles.get(keys[0], keys[0]), show_xlabel=False)
    n01 = draw_one(ax01, mats2d[keys[1]], titles.get(keys[1], keys[1]), show_xlabel=False)
    n10 = draw_one(ax10, mats2d[keys[2]], titles.get(keys[2], keys[2]), show_xlabel=True)
    n11 = draw_one(ax11, mats2d[keys[3]], titles.get(keys[3], keys[3]), show_xlabel=True)

    for ax_label in [ax_label_top, ax_label_bot]:
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(len(row_names), 0)
        ax_label.axis('off')
        for i, label in enumerate(row_names):
            ax_label.text(
                0.6, i + 0.5, label,
                ha='right', va='center',
                fontsize=10,
                fontweight='semibold',
                color='#2d5016',
                transform=ax_label.transData
            )

    if colorbar_single:
        from matplotlib.colors import Normalize
        if shared_norm_obj is None:
            vmax = max(float(n00.vmax), float(n01.vmax), float(n10.vmax), float(n11.vmax))
            vmax = max(vmax, 1e-6)
            norm_cb = Normalize(0.0, vmax)
        else:
            norm_cb = shared_norm_obj

        sm = ScalarMappable(norm=norm_cb, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label("Importance" + (" (log1p)" if use_log1p else ""), fontsize=8, labelpad=2)
        cb.ax.tick_params(labelsize=7, length=2, width=0.5)
        cb.formatter = mticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
        cb.outline.set_linewidth(0.5)
    else:
        from matplotlib.colors import Normalize
        if shared_norm_obj is not None:
            sm_top = ScalarMappable(norm=shared_norm_obj, cmap=cmap)
            sm_bot = ScalarMappable(norm=shared_norm_obj, cmap=cmap)
        else:
            sm_top = ScalarMappable(norm=Normalize(0.0, max(n00.vmax, n01.vmax)), cmap=cmap)
            sm_bot = ScalarMappable(norm=Normalize(0.0, max(n10.vmax, n11.vmax)), cmap=cmap)

        for cax_i, sm in [(cax_top, sm_top), (cax_bot, sm_bot)]:
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax_i, orientation="vertical")
            cb.set_label("Importance" + (" (log1p)" if use_log1p else ""), fontsize=8, labelpad=2)
            cb.ax.tick_params(labelsize=7, length=2, width=0.5)
            cb.formatter = mticker.ScalarFormatter(useMathText=True)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
            cb.outline.set_linewidth(0.5)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)


# =========================
# Dataset / padding utilities
# =========================

def _str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


def _dataset_kwargs_from_args(args):
    kw = {}
    if hasattr(args, "include_down_adj"):
        kw["include_down_adj"] = bool(args.include_down_adj)
    else:
        kw["include_down_adj"] = _str2bool(getattr(args, "use_coboundaries", "true"))

    if hasattr(args, "max_ring_size"):
        kw["max_ring_size"] = args.max_ring_size
    if hasattr(args, "use_edge_features"):
        kw["use_edge_features"] = args.use_edge_features
    if hasattr(args, "simple_features"):
        kw["simple_features"] = args.simple_features
    return kw


def load_dataset_from_args(name, args):
    return load_dataset(
        name,
        max_dim=int(getattr(args, "max_dim", 2)),
        init_method=getattr(args, "init_method", "mean"),
        n_jobs=int(getattr(args, "preproc_jobs", 2)),
        **_dataset_kwargs_from_args(args),
    )


def compute_gt_in_dim(args):
    names = set([args.dataset])
    if hasattr(args, "mix_datasets") and args.mix_datasets:
        for s in args.mix_datasets.split(","):
            s = s.strip()
            if s:
                names.add(s)

    dim_map = {}
    for name in sorted(names):
        ds = load_dataset_from_args(name, args)
        dim_map[name] = int(ds.num_features_in_dim(0))

    gt_in_dim = max(dim_map.values()) if dim_map else int(load_dataset_from_args(args.dataset, args).num_features_in_dim(0))
    print(f"[DEBUG] graph_transformer gt_in_dim={gt_in_dim} dim_map={dim_map}")
    return gt_in_dim, dim_map


def pad_batch_node_features_inplace(batch, target_dim: int):
    if not hasattr(batch, "cochains") or len(batch.cochains) == 0:
        return batch
    c0 = batch.cochains[0]
    if not hasattr(c0, "x") or c0.x is None:
        return batch

    x = c0.x
    if x.dim() != 2:
        return batch

    f = int(x.size(-1))
    if f == target_dim:
        return batch
    if f > target_dim:
        c0.x = x[:, :target_dim]
        return batch

    pad = target_dim - f
    c0.x = F.pad(x, (0, pad), mode="constant", value=0.0)
    return batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_subsample_indices(y, ratio, seed=0):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    chosen = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        n = max(1, int(len(cls_idx) * ratio))
        rng.shuffle(cls_idx)
        chosen.append(cls_idx[:n])
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return chosen


def _get_item_label(item):
    if hasattr(item, "y") and item.y is not None:
        return int(item.y)
    if hasattr(item, "cochains") and len(item.cochains) > 0 and hasattr(item.cochains[0], "y"):
        return int(item.cochains[0].y)
    raise AttributeError("Cannot find label in item (expected item.y or item.cochains[0].y).")


def _get_batch_label(batch, device):
    if hasattr(batch, "y") and batch.y is not None:
        y = batch.y
    else:
        y = batch.cochains[0].y
    if y.dim() > 1:
        y = y.view(-1)
    return y.long().to(device)


def infer_num_classes_from_dataset(ds):
    y_np = np.array([_get_item_label(ds[i]) for i in range(len(ds))], dtype=np.int64)
    return int(np.max(y_np)) + 1


def load_single_loader(args, dataset_name: str, ratio: float, shuffle: bool = True):
    ds = load_dataset_from_args(dataset_name, args)
    y_np = np.array([_get_item_label(ds[i]) for i in range(len(ds))], dtype=np.int64)
    keep = stratified_subsample_indices(y_np, ratio=ratio, seed=args.seed)

    items = [ds[int(i)] for i in keep]
    num_classes = int(np.max(y_np[keep])) + 1 if len(keep) > 0 else infer_num_classes_from_dataset(ds)
    ld = DataLoader(items, batch_size=args.batch_size, shuffle=shuffle)
    return ld, num_classes


class MixedBatchLoader:
    def __init__(self, loaders, shuffle_loaders_each_epoch=True):
        self.loaders = loaders
        self.shuffle_loaders_each_epoch = shuffle_loaders_each_epoch

    def __iter__(self):
        order = list(range(len(self.loaders)))
        if self.shuffle_loaders_each_epoch:
            random.shuffle(order)
        iters = {i: iter(self.loaders[i]) for i in order}
        alive = set(order)

        while alive:
            for i in list(order):
                if i not in alive:
                    continue
                try:
                    yield next(iters[i])
                except StopIteration:
                    alive.remove(i)

    def __len__(self):
        total = 0
        for ld in self.loaders:
            try:
                total += len(ld)
            except TypeError:
                pass
        return total


def load_mixed_loader(args, ratio, shuffle=True):
    names = [s.strip() for s in args.mix_datasets.split(",") if s.strip()]
    assert len(names) == 8, f"expected 8 datasets, got {len(names)}: {names}"

    loaders = []
    all_labels = []

    for name in names:
        ds = load_dataset_from_args(name, args)

        y_np = np.array([_get_item_label(ds[i]) for i in range(len(ds))])
        keep = stratified_subsample_indices(y_np, ratio=ratio, seed=args.seed)

        items = []
        labels = []
        for i in keep:
            item = ds[int(i)]
            items.append(item)
            labels.append(_get_item_label(item))

        all_labels.extend(labels)

        ld = DataLoader(items, batch_size=args.batch_size, shuffle=shuffle)
        loaders.append(ld)

    num_classes = int(np.max(all_labels)) + 1
    mixed_loader = MixedBatchLoader(loaders, shuffle_loaders_each_epoch=shuffle)
    return mixed_loader, num_classes


class MetaMask(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.mask_simclr = MaskSimclr(dataset, args)
        self.cell_mask_model = SampleSelector(input_dim=self.mask_simclr.embedding_dim)
        self.args = args


# =========================
# Pretrain / Train / Eval
# =========================

def supervised_pretrain_and_save(model: MetaMask, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    loader, num_classes = load_mixed_loader(args, ratio=args.pretrain_ratio, shuffle=True)

    emb_dim = int(model.mask_simclr.embedding_dim)
    sup_head = nn.Linear(emb_dim, num_classes).to(device)

    opt = optim.Adam(
        list(model.mask_simclr.parameters()) + list(sup_head.parameters()),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
    )
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.pretrain_epochs):
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device) if hasattr(batch, "to") else batch
            if args.model == "graph_transformer":
                pad_batch_node_features_inplace(batch, int(args.gt_in_dim))

            z = model.mask_simclr.encode(batch)
            y = _get_batch_label(batch, device)

            loss = ce(sup_head(z), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))

        print(f"[PRETRAIN] epoch={epoch+1:03d}/{args.pretrain_epochs} loss={total/max(1,n):.4f}")

    ckpt = {
        "mask_simclr": model.mask_simclr.state_dict(),
        "emb_dim": emb_dim,
        "mix_datasets": args.mix_datasets,
        "args": vars(args),
    }

    os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
    torch.save(ckpt, args.ckpt_path)
    print(f"[PRETRAIN] saved -> {args.ckpt_path}")


def load_pretrained_if_exists(model: MetaMask, ckpt_path: str, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.mask_simclr.load_state_dict(ckpt["mask_simclr"], strict=True)
        print(f"[LOAD] loaded pretrained encoder from {ckpt_path}")
        return True
    print(f"[LOAD] no ckpt found at {ckpt_path} (continue without pretrained weights)")
    return False


def apply_space_gradient_weights(mask_simclr_module, spaces, args):
    A, B, C, D = spaces["A"], spaces["B"], spaces["C"], spaces["D"]
    for name, p in mask_simclr_module.named_parameters():
        if (p.grad is None) or (not p.requires_grad):
            continue

        if name in A:
            w = float(getattr(args, "w_A", 1.0))
        elif name in B:
            w = float(getattr(args, "w_B", 1.0))
        elif name in C:
            w = float(getattr(args, "w_C", 1.0))
        else:
            w = float(getattr(args, "w_D", 1.0))

        p.grad.mul_(w)


def apply_heatmap_dim_gradient_weights(mask_simclr_module, heatmaps_raw: dict, args):
   
    strength = float(getattr(args, "heatmap_ft_strength", 1.0))
    wmin = float(getattr(args, "heatmap_ft_min", 0.2))
    wmax = float(getattr(args, "heatmap_ft_max", 5.0))
    eps = 1e-12

    needed = ["Only_structure", "Only_semantic", "Overlap", "Other"]
    for k in needed:
        if k not in heatmaps_raw:
            raise KeyError(f"[AUTO-FT] heatmaps_raw missing key={k}, got keys={list(heatmaps_raw.keys())}")

    H = (
        1.0 * np.asarray(heatmaps_raw["Overlap"], dtype=np.float32) +
        1.0 * np.asarray(heatmaps_raw["Only_structure"], dtype=np.float32) +
        1.0 * np.asarray(heatmaps_raw["Only_semantic"], dtype=np.float32) +
        0.2 * np.asarray(heatmaps_raw["Other"], dtype=np.float32)
    )
    if H.ndim != 2 or H.shape[0] != 3:
        raise ValueError(f"[AUTO-FT] bad heatmap shape: {H.shape}, expected (3, dim)")

    W = []
    for r in range(3):
        v = H[r]
        m = float(np.mean(v)) if np.isfinite(np.mean(v)) else 0.0
        if m <= 0.0:
            w = np.ones_like(v, dtype=np.float32)
        else:
            vnorm = v / (m + eps)
            w = 1.0 + strength * (vnorm - 1.0)
        w = np.clip(w, wmin, wmax).astype(np.float32)
        W.append(w)

    WQ = torch.tensor(W[0], dtype=torch.float32)
    WK = torch.tensor(W[1], dtype=torch.float32)
    WV = torch.tensor(W[2], dtype=torch.float32)

    def _weight_for_tag(tag: str):
        if tag == "Q":
            return WQ
        if tag == "K":
            return WK
        if tag == "V":
            return WV
        return None

    dim = int(WQ.numel())

    for name, p in mask_simclr_module.named_parameters():
        if p.grad is None or (not p.requires_grad):
            continue
        tag = _qkv_tag_from_param_name(name)
        if tag is None:
            continue
        w = _weight_for_tag(tag)
        if w is None:
            continue

        g = p.grad
        if g.dim() == 1 and g.numel() == dim:
            p.grad.mul_(w.to(device=g.device))
        elif g.dim() == 2 and g.size(0) == dim:
            p.grad.mul_(w.to(device=g.device).view(dim, 1))


def apply_heatmap_dim_gradient_weights_mixed(mask_simclr_module, heatmaps1_raw: dict, heatmaps2_raw: dict, args):
    
    a1 = float(getattr(args, "heatmap_mix_first", 0.8))
    a2 = float(getattr(args, "heatmap_mix_second", 0.2))

    strength = float(getattr(args, "heatmap_ft_strength", 1.0))
    wmin = float(getattr(args, "heatmap_ft_min", 0.2))
    wmax = float(getattr(args, "heatmap_ft_max", 5.0))
    eps = 1e-12

    needed = ["Only_structure", "Only_semantic", "Overlap", "Other"]
    for k in needed:
        if k not in heatmaps1_raw or k not in heatmaps2_raw:
            raise KeyError(f"[MIX-FT] missing key={k}")

    def _make_W(heatmaps_raw: dict):
        H = (
            1.0 * np.asarray(heatmaps_raw["Overlap"], dtype=np.float32) +
            1.0 * np.asarray(heatmaps_raw["Only_structure"], dtype=np.float32) +
            1.0 * np.asarray(heatmaps_raw["Only_semantic"], dtype=np.float32) +
            0.2 * np.asarray(heatmaps_raw["Other"], dtype=np.float32)
        )
        if H.ndim != 2 or H.shape[0] != 3:
            raise ValueError(f"[MIX-FT] bad heatmap shape: {H.shape}, expected (3, dim)")

        W = []
        for r in range(3):
            v = H[r]
            m = float(np.mean(v)) if np.isfinite(np.mean(v)) else 0.0
            if m <= 0.0:
                w = np.ones_like(v, dtype=np.float32)
            else:
                vnorm = v / (m + eps)
                w = 1.0 + strength * (vnorm - 1.0)
            w = np.clip(w, wmin, wmax).astype(np.float32)
            W.append(w)
        return W

    W1 = _make_W(heatmaps1_raw)
    W2 = _make_W(heatmaps2_raw)
    Wmix = [np.clip(a1 * W1[i] + a2 * W2[i], wmin, wmax).astype(np.float32) for i in range(3)]

    WQ = torch.tensor(Wmix[0], dtype=torch.float32)
    WK = torch.tensor(Wmix[1], dtype=torch.float32)
    WV = torch.tensor(Wmix[2], dtype=torch.float32)

    def _weight_for_tag(tag: str):
        if tag == "Q": return WQ
        if tag == "K": return WK
        if tag == "V": return WV
        return None

    dim = int(WQ.numel())

    for name, p in mask_simclr_module.named_parameters():
        if p.grad is None or (not p.requires_grad):
            continue
        tag = _qkv_tag_from_param_name(name)
        if tag is None:
            continue
        w = _weight_for_tag(tag)
        if w is None:
            continue

        g = p.grad
        if g.dim() == 1 and g.numel() == dim:
            p.grad.mul_(w.to(device=g.device))
        elif g.dim() == 2 and g.size(0) == dim:
            p.grad.mul_(w.to(device=g.device).view(dim, 1))


# =========================
# Head selection + overlap
# =========================

def _get_num_heads(model: MetaMask, args):
    nh = getattr(args, "num_heads", None)
    if nh is None:
        nh = getattr(args, "n_heads", None)
    if nh is not None:
        return int(nh)

    for obj in [model.mask_simclr, getattr(model.mask_simclr, "encoder", None)]:
        if obj is None:
            continue
        for attr in ["num_heads", "n_heads", "heads"]:
            if hasattr(obj, attr):
                try:
                    return int(getattr(obj, attr))
                except Exception:
                    pass

    raise AttributeError("无法推断 num_heads：请显式提供 args.num_heads (或 args.n_heads)。")


def _head_scores_from_qkv_grad(param_name: str, g: torch.Tensor, dim: int, num_heads: int):
    tag = _qkv_tag_from_param_name(param_name)
    if tag is None:
        return {}

    layer_key = _layer_key_from_param_name(param_name)
    if layer_key is None:
        return {}

    if g is None or (not torch.is_tensor(g)):
        return {}

    gg = g.detach()
    if gg.is_sparse:
        gg = gg.to_dense()
    gg = gg.float()

    head_dim = dim // num_heads
    if head_dim * num_heads != dim:
        raise ValueError(f"dim={dim} 不能被 num_heads={num_heads} 整除。")

    out = {}

    if gg.dim() == 1 and gg.numel() == dim:
        for h in range(num_heads):
            sl = gg[h * head_dim: (h + 1) * head_dim]
            out[(layer_key, h)] = float(torch.norm(sl, p=2).item())
        return out

    if gg.dim() == 2 and gg.size(0) == dim:
        for h in range(num_heads):
            sl = gg[h * head_dim: (h + 1) * head_dim, :]
            out[(layer_key, h)] = float(torch.norm(sl, p=2).item())
        return out

    return {}


def probe_parameter_spaces(model: MetaMask, args, num_classes, run_dir=None, dataset_name: str = None):
  
    if dataset_name is None:
        dataset_name = args.dataset

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    probe_data_mode = str(getattr(args, "probe_data_mode", "target")).lower()
    probe_ratio = float(getattr(args, "probe_ratio", 1.0))

    if probe_data_mode == "mixed":
        loader, _ = load_mixed_loader(args, ratio=probe_ratio, shuffle=True)
        print(f"[PROBE] data_mode=mixed ratio={probe_ratio}")
    else:
        loader, _ = load_single_loader(args, dataset_name=dataset_name, ratio=probe_ratio, shuffle=True)
        print(f"[PROBE] data_mode=target dataset={dataset_name} ratio={probe_ratio}")

    batches = int(getattr(args, "probe_batches", 20))
    try:
        total_batches = len(loader)
    except Exception:
        total_batches = None

    planned = batches if total_batches is None else min(batches, total_batches)
    approx_samples = planned * int(args.batch_size)

    print(
        f"[PROBE] probe_batches={batches} total_loader_batches={total_batches} "
        f"planned_used_batches={planned} approx_used_samples<={approx_samples}"
    )

    params = [(n, p) for n, p in model.mask_simclr.named_parameters() if p.requires_grad]
    names = [n for n, _ in params]
    tensors = [p for _, p in params]

    emb_dim = int(model.mask_simclr.embedding_dim)
    sup_head = nn.Linear(emb_dim, num_classes).to(device)
    ce = nn.CrossEntropyLoss()

    sum_head_con = defaultdict(float)
    sum_head_ce = defaultdict(float)

    used = 0
    num_heads = _get_num_heads(model, args)

    dbg_qkv_total = 0
    dbg_qkv_with_layerkey = 0

    for batch in loader:
        batch = batch.to(device) if hasattr(batch, "to") else batch
        used += 1
        if used > batches:
            break

        if args.model == "graph_transformer":
            pad_batch_node_features_inplace(batch, int(args.gt_in_dim))

        data1 = copy.deepcopy(batch)
        data2 = copy.deepcopy(batch)
        x1 = model.mask_simclr(data1)
        x2 = model.mask_simclr(data2)
        loss_con = model.mask_simclr.simclr_loss(x1, x2)

        grads_con = torch.autograd.grad(
            loss_con, tensors, retain_graph=True, create_graph=False, allow_unused=True
        )

        z = model.mask_simclr.encode(batch)
        y = _get_batch_label(batch, device)
        loss_ce = ce(sup_head(z), y)

        grads_ce = torch.autograd.grad(
            loss_ce, tensors, retain_graph=False, create_graph=False, allow_unused=True
        )

        for n, gc, ge in zip(names, grads_con, grads_ce):
            if _qkv_tag_from_param_name(n) is None:
                continue

            dbg_qkv_total += 1
            if _layer_key_from_param_name(n) is not None:
                dbg_qkv_with_layerkey += 1

            if gc is not None:
                for hk, sc in _head_scores_from_qkv_grad(n, gc, dim=emb_dim, num_heads=num_heads).items():
                    sum_head_con[hk] += sc

            if ge is not None:
                for hk, se in _head_scores_from_qkv_grad(n, ge, dim=emb_dim, num_heads=num_heads).items():
                    sum_head_ce[hk] += se

    denom = max(1, min(used, batches))

    head_keys = sorted(set(list(sum_head_con.keys()) + list(sum_head_ce.keys())))
    if len(head_keys) == 0:
        raise RuntimeError()

    I_head_con = {hk: sum_head_con.get(hk, 0.0) / denom for hk in head_keys}
    I_head_ce = {hk: sum_head_ce.get(hk, 0.0) / denom for hk in head_keys}

    select_mode = str(getattr(args, "head_select_mode", "percentile"))
    per_layer = bool(getattr(args, "head_select_per_layer", True))

    k_head_con = int(getattr(args, "topk_head_k_con", 0))
    k_head_ce = int(getattr(args, "topk_head_k_ce", 0))
    ratio_con = float(getattr(args, "topk_head_ratio_con", 0.2))
    ratio_ce = float(getattr(args, "topk_head_ratio_ce", 0.2))

    pct_con = float(getattr(args, "overlap_pct_con", 70.0))
    pct_ce = float(getattr(args, "overlap_pct_ce", 70.0))

    auto_relax = bool(getattr(args, "overlap_auto_relax", True))
    relax_step = float(getattr(args, "overlap_relax_step", 5.0))
    relax_min = float(getattr(args, "overlap_relax_min", 0.0))

    layer_keys = sorted(set([lk for (lk, _) in head_keys]))

    def _select_heads_percentile(score_map: dict, pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        selected = set()

        if per_layer:
            for lk in layer_keys:
                vals = [score_map.get((lk, h), 0.0) for h in range(num_heads)]
                vmax = float(np.max(vals)) if len(vals) else 0.0
                if vmax <= 0.0:
                    continue
                thr = float(np.percentile(np.array(vals, dtype=np.float32), pct))
                if thr <= 0.0:
                    thr = float(np.nextafter(0, 1))
                for h in range(num_heads):
                    if score_map.get((lk, h), 0.0) >= thr:
                        selected.add((lk, h))
            return selected

        vals = np.array([score_map.get(hk, 0.0) for hk in head_keys], dtype=np.float32)
        vmax = float(np.max(vals)) if vals.size else 0.0
        if vmax <= 0.0:
            return set()
        thr = float(np.percentile(vals, pct))
        if thr <= 0.0:
            thr = float(np.nextafter(0, 1))
        return set([hk for hk in head_keys if score_map.get(hk, 0.0) >= thr])

    def _select_heads_topk(score_map: dict, k: int, ratio: float):
        selected = set()
        if per_layer:
            kk = int(k) if int(k) > 0 else max(1, int(round(num_heads * float(ratio))))
            kk = max(1, min(num_heads, kk))
            for lk in layer_keys:
                items = [((lk, h), score_map.get((lk, h), 0.0)) for h in range(num_heads)]
                items.sort(key=lambda x: x[1], reverse=True)
                selected |= set([hk for hk, _ in items[:kk]])
            return selected

        N = len(head_keys)
        kk = int(k) if int(k) > 0 else max(1, int(round(N * float(ratio))))
        kk = max(1, min(N, kk))
        items = sorted([(hk, score_map.get(hk, 0.0)) for hk in head_keys], key=lambda x: x[1], reverse=True)
        return set([hk for hk, _ in items[:kk]])

    def _pick_sets(p_con, p_ce):
        if select_mode == "topk":
            s_con = _select_heads_topk(I_head_con, k_head_con, ratio_con)
            s_ce = _select_heads_topk(I_head_ce, k_head_ce, ratio_ce)
        else:
            s_con = _select_heads_percentile(I_head_con, p_con)
            s_ce = _select_heads_percentile(I_head_ce, p_ce)
        return s_con, s_ce

    cur_pct_con, cur_pct_ce = pct_con, pct_ce
    top_heads_con, top_heads_ce = _pick_sets(cur_pct_con, cur_pct_ce)
    overlap_heads = top_heads_con.intersection(top_heads_ce)

    if auto_relax and len(overlap_heads) == 0 and select_mode != "topk":
        while len(overlap_heads) == 0 and (cur_pct_con > relax_min or cur_pct_ce > relax_min):
            cur_pct_con = max(relax_min, cur_pct_con - relax_step)
            cur_pct_ce = max(relax_min, cur_pct_ce - relax_step)
            top_heads_con, top_heads_ce = _pick_sets(cur_pct_con, cur_pct_ce)
            overlap_heads = top_heads_con.intersection(top_heads_ce)

    only_heads_con = top_heads_con - overlap_heads
    only_heads_ce = top_heads_ce - overlap_heads

    A, B, C = set(), set(), set()

    for n in names:
        if _qkv_tag_from_param_name(n) is None:
            continue
        lk = _layer_key_from_param_name(n)
        if lk is None:
            continue

        layer_heads = {(lk, h) for h in range(num_heads)}

        if len(layer_heads & overlap_heads) > 0:
            C.add(n)
        elif len(layer_heads & only_heads_con) > 0:
            A.add(n)
        elif len(layer_heads & only_heads_ce) > 0:
            B.add(n)

    D = set(names) - (A | B | C)

    spaces = {"A": A, "B": B, "C": C, "D": D}
    spaces["_head_sets"] = {
        "overlap": set(overlap_heads),
        "only_con": set(only_heads_con),
        "only_ce": set(only_heads_ce),
        "top_con": set(top_heads_con),
        "top_ce": set(top_heads_ce),
        "all": set(head_keys),
    }
    spaces["_num_heads"] = int(num_heads)
    spaces["_emb_dim"] = int(emb_dim)
    spaces["_probe_dataset"] = str(dataset_name)
    spaces["_probe_data_mode"] = str(probe_data_mode)

    meta = {
        "probe_dataset": str(dataset_name),
        "probe_data_mode": str(probe_data_mode),
        "probe_ratio": float(probe_ratio),
        "probe_batches_used": int(denom),
        "num_params": int(len(names)),
        "num_heads": int(num_heads),
        "layers_seen": int(len(layer_keys)),
        "qkv_params_seen": int(dbg_qkv_total),
        "qkv_params_with_layer_key": int(dbg_qkv_with_layerkey),
        "select_mode": str(select_mode),
        "per_layer": bool(per_layer),
        "overlap_threshold": {
            "overlap_pct_con_init": float(pct_con),
            "overlap_pct_ce_init": float(pct_ce),
            "overlap_pct_con_used": float(cur_pct_con),
            "overlap_pct_ce_used": float(cur_pct_ce),
            "auto_relax": bool(auto_relax),
            "relax_step": float(relax_step),
            "relax_min": float(relax_min),
        },
        "heads_count": {
            "top_heads_con": int(len(top_heads_con)),
            "top_heads_ce": int(len(top_heads_ce)),
            "overlap_heads": int(len(overlap_heads)),
        },
        "sizes": {"A": len(A), "B": len(B), "C": len(C), "D": len(D)},
        "overlap_def": "overlap_heads = selected_con ∩ selected_ce",
    }

    print(
        f"[PROBE-HEAD] dataset={dataset_name} mode={probe_data_mode} ratio={probe_ratio} "
        f"batches={denom} layers={len(layer_keys)} num_heads={num_heads} "
        f"qkv(layer_key)={dbg_qkv_with_layerkey}/{dbg_qkv_total} "
        f"pct_used(con,ce)=({cur_pct_con:.1f},{cur_pct_ce:.1f}) "
        f"top_con={len(top_heads_con)} top_ce={len(top_heads_ce)} overlap={len(overlap_heads)} "
        f"|A|={len(A)} |B|={len(B)} |C|={len(C)} |D|={len(D)}"
    )

    if run_dir is not None:
        save_json(os.path.join(run_dir, "probe_meta.json"), meta)
        save_json(
            os.path.join(run_dir, "overlap_heads.json"),
            {"overlap_heads": [{"layer_key": hk[0], "head": int(hk[1])} for hk in sorted(list(overlap_heads))]},
        )
        save_topk_csv(
            os.path.join(run_dir, "top_heads_con.csv"),
            [(f"{hk[0]}::head{hk[1]}", 1.0) for hk in sorted(list(top_heads_con))],
        )
        save_topk_csv(
            os.path.join(run_dir, "top_heads_ce.csv"),
            [(f"{hk[0]}::head{hk[1]}", 1.0) for hk in sorted(list(top_heads_ce))],
        )

    return spaces, meta, [], [], []



def compute_qkv_heatmaps_by_param_space_fisher(model, args, num_classes: int, spaces: dict, dim: int, dataset_name: str = None):
 
    if dataset_name is None:
        dataset_name = args.dataset

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    if args.model != "graph_transformer":
        raise ValueError("none QKV")

    heatmap_data_mode = str(getattr(args, "heatmap_data_mode", "target")).lower()

    heatmap_ratio_arg = getattr(args, "heatmap_ratio", None)
    heatmap_ratio = float(heatmap_ratio_arg) if heatmap_ratio_arg is not None else float(getattr(args, "probe_ratio", 1.0))

    if heatmap_data_mode == "mixed":
        loader, _ = load_mixed_loader(args, ratio=heatmap_ratio, shuffle=True)
        print(f"[HEATMAP-FISHER] data_mode=mixed ratio={heatmap_ratio}")
    else:
        loader, _ = load_single_loader(args, dataset_name=dataset_name, ratio=heatmap_ratio, shuffle=True)
        print(f"[HEATMAP-FISHER] data_mode=target dataset={dataset_name} ratio={heatmap_ratio}")

    emb_dim = int(model.mask_simclr.embedding_dim)
    assert emb_dim == dim, f"no cons"

    head_sets = spaces.get("_head_sets", None)
    num_heads = int(spaces.get("_num_heads", getattr(args, "num_heads", 0) or getattr(args, "n_heads", 0) or 0))
    if head_sets is None or num_heads <= 0:
        raise RuntimeError("no infor")

    overlap_heads = set(head_sets["overlap"])
    only_heads_con = set(head_sets["only_con"])
    only_heads_ce = set(head_sets["only_ce"])

    head_dim = dim // num_heads
    if head_dim * num_heads != dim:
        raise ValueError(f"//no")

    def head_grad_to_dim_vector_first(g: torch.Tensor, head_id: int):
        if g is None or (not torch.is_tensor(g)):
            return None
        gg = g.detach()
        if gg.is_sparse:
            gg = gg.to_dense()
        gg = gg.float()

        out = torch.zeros((dim,), dtype=torch.float32)
        s = head_id * head_dim
        e = (head_id + 1) * head_dim

        if gg.dim() == 1 and gg.numel() == dim:
            out[s:e] = gg[s:e].abs().cpu()
            return out

        if gg.dim() == 2 and gg.size(0) == dim:
            sl = gg[s:e, :]
            out[s:e] = torch.norm(sl, p=2, dim=1).cpu()
            return out

        return None

    def head_grad_to_dim_vector_fisher_diag(g: torch.Tensor, head_id: int):
        
        if g is None or (not torch.is_tensor(g)):
            return None
        gg = g.detach()
        if gg.is_sparse:
            gg = gg.to_dense()
        gg = gg.float()

        out = torch.zeros((dim,), dtype=torch.float32)
        s = head_id * head_dim
        e = (head_id + 1) * head_dim

        if gg.dim() == 1 and gg.numel() == dim:
            out[s:e] = (gg[s:e] * gg[s:e]).cpu()
            return out

        if gg.dim() == 2 and gg.size(0) == dim:
            sl = gg[s:e, :]  # (head_dim, in)
            out[s:e] = torch.mean(sl * sl, dim=1).cpu()
            return out

        return None

    # collect qkv params
    qkv_params, qkv_names, qkv_tags, qkv_layer_keys = [], [], [], []
    for n, p in model.mask_simclr.named_parameters():
        if not p.requires_grad:
            continue
        tag = _qkv_tag_from_param_name(n)
        if tag is None:
            continue
        lk = _layer_key_from_param_name(n)
        if lk is None:
            continue
        qkv_names.append(n)
        qkv_params.append(p)
        qkv_tags.append(tag)
        qkv_layer_keys.append(lk)

    if len(qkv_params) == 0:
        raise RuntimeError(" ")

    A = spaces.get("A", set())
    B = spaces.get("B", set())
    C = spaces.get("C", set())
    D = spaces.get("D", set())
    space_sets = {"Only_structure": A, "Only_semantic": B, "Overlap": C, "Other": D}

    row_keys = ["Q", "K", "V"]
    heat1 = {sp: torch.zeros((3, dim), dtype=torch.float32) for sp in space_sets.keys()}
    heat2 = {sp: torch.zeros((3, dim), dtype=torch.float32) for sp in space_sets.keys()}
    counts = {sp: {"Q": 0, "K": 0, "V": 0, "qkv_params": len(qkv_params)} for sp in space_sets.keys()}

    sup_head = nn.Linear(emb_dim, num_classes).to(device)
    ce = nn.CrossEntropyLoss()
    heatmap_batches = int(getattr(args, "heatmap_batches", 8))

    batch_count = 0
    for batch in loader:
        batch = batch.to(device) if hasattr(batch, "to") else batch
        batch_count += 1
        if batch_count > heatmap_batches:
            break

        pad_batch_node_features_inplace(batch, int(args.gt_in_dim))

        data1 = copy.deepcopy(batch)
        data2 = copy.deepcopy(batch)
        x1 = model.mask_simclr(data1)
        x2 = model.mask_simclr(data2)
        loss_con = model.mask_simclr.simclr_loss(x1, x2)

        z = model.mask_simclr.encode(batch)
        y = _get_batch_label(batch, device)
        loss_ce = ce(sup_head(z), y)

        grads_con = torch.autograd.grad(loss_con, qkv_params, retain_graph=True, create_graph=False, allow_unused=True)
        grads_ce = torch.autograd.grad(loss_ce, qkv_params, retain_graph=False, create_graph=False, allow_unused=True)

        for lk, tag, gc, ge in zip(qkv_layer_keys, qkv_tags, grads_con, grads_ce):
            ridx = row_keys.index(tag)

            for h in range(num_heads):
                hk = (lk, h)

                vc1 = head_grad_to_dim_vector_first(gc, h)
                ve1 = head_grad_to_dim_vector_first(ge, h)

                vc2 = head_grad_to_dim_vector_fisher_diag(gc, h)
                ve2 = head_grad_to_dim_vector_fisher_diag(ge, h)

                if hk in overlap_heads:
                    if vc1 is not None:
                        heat1["Overlap"][ridx] += vc1
                        counts["Overlap"][tag] += 1
                    if ve1 is not None:
                        heat1["Overlap"][ridx] += ve1
                        counts["Overlap"][tag] += 1
                    if vc2 is not None:
                        heat2["Overlap"][ridx] += vc2
                    if ve2 is not None:
                        heat2["Overlap"][ridx] += ve2

                elif hk in only_heads_con:
                    if vc1 is not None:
                        heat1["Only_structure"][ridx] += vc1
                        counts["Only_structure"][tag] += 1
                    if vc2 is not None:
                        heat2["Only_structure"][ridx] += vc2

                elif hk in only_heads_ce:
                    if ve1 is not None:
                        heat1["Only_semantic"][ridx] += ve1
                        counts["Only_semantic"][tag] += 1
                    if ve2 is not None:
                        heat2["Only_semantic"][ridx] += ve2

                else:
                    if vc1 is not None:
                        heat1["Other"][ridx] += vc1
                        counts["Other"][tag] += 1
                    if ve1 is not None:
                        heat1["Other"][ridx] += ve1
                        counts["Other"][tag] += 1

                    if vc2 is not None:
                        heat2["Other"][ridx] += vc2
                    if ve2 is not None:
                        heat2["Other"][ridx] += ve2

    normalize_mode = str(getattr(args, "heatmap_normalize_mode", "mean"))
    eps = 1e-12
    if normalize_mode in ("mean", "per_head"):
        for sp in heat1.keys():
            for tag_idx, tag in enumerate(["Q", "K", "V"]):
                denom = float(counts[sp].get(tag, 0))
                if denom > 0:
                    heat1[sp][tag_idx] /= (denom + eps)
                    heat2[sp][tag_idx] /= (denom + eps)

        if normalize_mode == "per_head":
            n_overlap = max(1, len(overlap_heads))
            n_only_con = max(1, len(only_heads_con))
            n_only_ce = max(1, len(only_heads_ce))
            all_heads = set(head_sets.get("all", []))
            n_other = max(1, len(all_heads - overlap_heads - only_heads_con - only_heads_ce))

            heat1["Overlap"] /= float(n_overlap)
            heat1["Only_structure"] /= float(n_only_con)
            heat1["Only_semantic"] /= float(n_only_ce)
            heat1["Other"] /= float(n_other)

            heat2["Overlap"] /= float(n_overlap)
            heat2["Only_structure"] /= float(n_only_con)
            heat2["Only_semantic"] /= float(n_only_ce)
            heat2["Other"] /= float(n_other)

    heatmaps1_raw = {sp: H.numpy() for sp, H in heat1.items()}
    heatmaps2_raw = {sp: H.numpy() for sp, H in heat2.items()}

  
    heatmap_binary = bool(getattr(args, "heatmap_binary", False))
    keep_pct = float(getattr(args, "heatmap_keep_percentile", 85.0))
    pct_mode = str(getattr(args, "heatmap_percentile_mode", "global"))

    topk_dim_k = getattr(args, "heatmap_topk_dim_k", None)
    topk_dim_ratio = getattr(args, "heatmap_topk_dim_ratio", 0.8)
    topk_mode = getattr(args, "heatmap_topk_mode", "per_row")

    heatmaps_masked = {
        sp: apply_topk_mask_to_heatmap(heatmaps1_raw[sp], k=topk_dim_k, ratio=topk_dim_ratio, mode=topk_mode)
        for sp in heatmaps1_raw.keys()
    }

    if heatmap_binary:
        heatmaps_vis = {
            sp: apply_percentile_binary_mask_to_heatmap(heatmaps_masked[sp], keep_percentile=keep_pct, mode=pct_mode)
            for sp in heatmaps_masked.keys()
        }
    else:
        heatmaps_vis = heatmaps_masked

    def _stats_map(hraw: dict):
        stats = {}
        for sp, H in hraw.items():
            HH = np.asarray(H, dtype=np.float32)
            stats[sp] = {"sum": float(np.sum(HH)), "mean": float(np.mean(HH)), "max": float(np.max(HH))}
        return stats

    meta = {
        "dataset": str(dataset_name),
        "heatmap_data_mode": str(heatmap_data_mode),
        "heatmap_ratio": float(heatmap_ratio),
        "heatmap_batches": int(batch_count),
        "second_order": {
            "type": "diag_fisher",
            "definition": "F ≈ E[g^2] (per-parameter), aggregated to per-output-dim",
        },
        "stats_first_order": _stats_map(heatmaps1_raw),
        "stats_second_order": _stats_map(heatmaps2_raw),
    }

    print("[HEATMAP-FISHER][STATS first]", json.dumps(meta["stats_first_order"], ensure_ascii=False))
    print("[HEATMAP-FISHER][STATS second]", json.dumps(meta["stats_second_order"], ensure_ascii=False))
    return heatmaps_vis, heatmaps1_raw, heatmaps2_raw, meta


def save_heatmap_csv(path, H, row_names):
    dim = H.shape[1]
    header = ["row"] + [f"dim_{i}" for i in range(dim)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r, name in enumerate(row_names):
            w.writerow([name] + [float(x) for x in H[r].tolist()])


# =========================
# Train / Eval helpers
# =========================

def weighted_train(model: MetaMask, args, spaces, num_classes, dataset_name: str, heatmaps_for_grad_1=None, heatmaps_for_grad_2=None, freeze_space="none"):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    train_data_mode = str(getattr(args, "train_data_mode", "target")).lower()
    train_ratio = float(getattr(args, "train_ratio", getattr(args, "probe_ratio", 1.0)))

    if train_data_mode == "mixed":
        loader, _ = load_mixed_loader(args, ratio=train_ratio, shuffle=True)
        print(f"[TRAIN] data_mode=mixed ratio={train_ratio}")
    else:
        loader, _ = load_single_loader(args, dataset_name=dataset_name, ratio=train_ratio, shuffle=True)
        print(f"[TRAIN] data_mode=target dataset={dataset_name} ratio={train_ratio}")

    emb_dim = int(model.mask_simclr.embedding_dim)
    sup_head = nn.Linear(emb_dim, num_classes).to(device)
    ce = nn.CrossEntropyLoss()

    freeze_set = freeze_params_by_space(model.mask_simclr, spaces, freeze_space=str(freeze_space))

    trainable_params = [p for p in model.mask_simclr.parameters() if p.requires_grad]

    opt = optim.Adam(
        trainable_params + list(sup_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_epochs = int(getattr(args, "epochs", 20))
    lambda_con = float(getattr(args, "lambda_con", 1.0))
    lambda_ce = float(getattr(args, "lambda_ce", 1.0))

    for epoch in range(train_epochs):
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device) if hasattr(batch, "to") else batch
            if args.model == "graph_transformer":
                pad_batch_node_features_inplace(batch, int(args.gt_in_dim))

            data1 = copy.deepcopy(batch)
            data2 = copy.deepcopy(batch)
            x1 = model.mask_simclr(data1)
            x2 = model.mask_simclr(data2)
            loss_con = model.mask_simclr.simclr_loss(x1, x2)

            z = model.mask_simclr.encode(batch)
            y = _get_batch_label(batch, device)
            loss_ce = ce(sup_head(z), y)

            loss = lambda_con * loss_con + lambda_ce * loss_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()

            auto_heatmap_ft = _str2bool(getattr(args, "auto_heatmap_ft", True))
            if auto_heatmap_ft and (heatmaps_for_grad_1 is not None) and (heatmaps_for_grad_2 is not None):
                apply_heatmap_dim_gradient_weights_mixed(model.mask_simclr, heatmaps_for_grad_1, heatmaps_for_grad_2, args)
            elif auto_heatmap_ft and (heatmaps_for_grad_1 is not None):
                apply_heatmap_dim_gradient_weights(model.mask_simclr, heatmaps_for_grad_1, args)
            else:
                apply_space_gradient_weights(model.mask_simclr, spaces, args)

            opt.step()

            total += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))

        print(f"[TRAIN] epoch={epoch+1:03d}/{train_epochs} loss={total/max(1,n):.4f}")


@torch.no_grad()
def extract_embeddings(model: MetaMask, dataset, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    zs, ys = [], []

    for batch in loader:
        batch = batch.to(device) if hasattr(batch, "to") else batch
        if args.model == "graph_transformer":
            pad_batch_node_features_inplace(batch, int(args.gt_in_dim))

        z = model.mask_simclr.encode(batch).detach().cpu()
        y = _get_batch_label(batch, device).detach().cpu()
        zs.append(z)
        ys.append(y)

    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()
    return Z, Y


def run_test(model: MetaMask, args):
    target_ds = load_dataset_from_args(args.dataset, args)
    X, y = extract_embeddings(model, target_ds, args)
    res = evaluate_embedding(X, y)
    print("[TEST] evaluate_embedding:", res)
    return res


def visualize_test_binary_2d(
    model: MetaMask,
    args,
    run_dir: str,
    method: str = "tsne",
    max_points: int = 5000,
):
    import numpy as np

    method = str(method).strip().lower()
    assert method in ("tsne", "umap"), f"method must be 'tsne' or 'umap', got {method}"

    ds = load_dataset_from_args(args.dataset, args)
    Z, Y = extract_embeddings(model, ds, args)

    np.save(os.path.join(run_dir, "embeddings_test_Z.npy"), Z)
    np.save(os.path.join(run_dir, "embeddings_test_Y.npy"), Y)

    rng = np.random.RandomState(int(args.seed))
    N = Z.shape[0]
    if (max_points is not None) and (N > int(max_points)):
        idx = rng.choice(N, size=int(max_points), replace=False)
        Zs, Ys = Z[idx], Y[idx]
    else:
        Zs, Ys = Z, Y

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=int(getattr(args, "tsne_perplexity", 30)),
            learning_rate="auto",
            init="pca",
            random_state=int(args.seed),
        )
        X2 = reducer.fit_transform(Zs)
    else:
        try:
            import umap
        except Exception as e:
            raise RuntimeError(" ") from e

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(getattr(args, "umap_n_neighbors", 15)),
            min_dist=float(getattr(args, "umap_min_dist", 0.1)),
            metric=str(getattr(args, "umap_metric", "euclidean")),
            random_state=int(args.seed),
        )
        X2 = reducer.fit_transform(Zs)

    csv_path = os.path.join(run_dir, f"{method}_test_binary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label"])
        for (x, y), lab in zip(X2, Ys):
            w.writerow([float(x), float(y), int(lab)])
    print(f"[{method.upper()}-TEST] saved -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[{method.upper()}-TEST] matplotlib not available, skip png. err={e}")
        return

    png_path = os.path.join(run_dir, f"{method}_test_binary.png")
    fig, ax = plt.subplots(figsize=(7, 6))

    Ys = Ys.astype(int)
    m0 = (Ys == 0)
    m1 = (Ys == 1)

    ax.scatter(X2[m0, 0], X2[m0, 1], s=10, alpha=0.9, c="#1f77b4")
    ax.scatter(X2[m1, 0], X2[m1, 1], s=10, alpha=0.9, c="#F28500")

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[{method.upper()}-TEST] saved -> {png_path}")


# =========================
# main
# =========================

def main():
    parser = get_parser()
    args = parser.parse_args()

    run_dir = setup_run_dir_and_logger(args)
    save_json(os.path.join(run_dir, "config.json"), vars(args))

    import sys
    print("[DEBUG] sys.argv:", sys.argv)
    print("[DEBUG] do_pretrain:", args.do_pretrain)
    print("[DEBUG] ckpt_path:", args.ckpt_path)
    print("[DEBUG] simgrace_meta file:", __file__)
    print("[DEBUG] cwd:", os.getcwd())

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    base_ds = load_dataset_from_args(args.dataset, args)

    if args.model == "graph_transformer":
        args.gt_in_dim, dim_map = compute_gt_in_dim(args)
        save_json(os.path.join(run_dir, "feature_dims.json"), {"gt_in_dim": int(args.gt_in_dim), "dim_map": dim_map})

    model = MetaMask(base_ds, args).to(device)

    if args.do_pretrain:
        supervised_pretrain_and_save(model, args)
        load_pretrained_if_exists(model, args.ckpt_path, device)
    else:
        load_pretrained_if_exists(model, args.ckpt_path, device)

    spaces = None
    num_classes = None
    heatmaps_for_grad_1 = None
    heatmaps_for_grad_2 = None

    if args.do_probe:
        _, num_classes = load_single_loader(args, dataset_name=args.dataset, ratio=1.0, shuffle=False)

        spaces, probe_meta, _, _, _ = probe_parameter_spaces(
            model, args, num_classes, run_dir=run_dir, dataset_name=args.dataset
        )

        save_json(os.path.join(run_dir, "spaces.json"), spaces_to_serializable(spaces))
        save_json(os.path.join(run_dir, "probe_meta.json"), probe_meta)

        dim = int(model.mask_simclr.embedding_dim)

  
        heatmaps_vis, heatmaps1_raw, heatmaps2_raw, meta = compute_qkv_heatmaps_by_param_space_fisher(
            model, args, num_classes, spaces, dim=dim, dataset_name=args.dataset
        )
        heatmaps_for_grad_1 = heatmaps1_raw
        heatmaps_for_grad_2 = heatmaps2_raw

        save_json(os.path.join(run_dir, "heatmap_qkv_by_space_fisher_meta.json"), meta)

        use_log1p = not bool(getattr(args, "heatmap_binary", True))
        cmap = "Greens"

        grid_png_path = os.path.join(run_dir, "heatmap_qkv_ABCD_2x2.png")
        plot_qkv_2x2_grid_png(
            grid_png_path,
            heatmaps=heatmaps_vis,
            dim=dim,
            row_names=("Q", "K", "V"),
            order=("Only_structure", "Only_semantic", "Overlap", "Other"),
            use_log1p=use_log1p,
            cmap=cmap,
            colorbar_single=True,
            unify_colorbar=True,
            vmax_percentile=99.0,
        )
        print(f"[HEATMAP-QKV] saved 2x2 grid -> {grid_png_path}")

        for tag, heatmaps in [("first_raw", heatmaps1_raw), ("second_raw", heatmaps2_raw)]:
            for sp, H in heatmaps.items():
                npy_path = os.path.join(run_dir, f"heatmap_qkv_{sp}_{tag}.npy")
                csv_path = os.path.join(run_dir, f"heatmap_qkv_{sp}_{tag}.csv")
                np.save(npy_path, H)
                save_heatmap_csv(csv_path, H, row_names=("Q", "K", "V"))
                print(f"[HEATMAP-QKV:{sp}][{tag}] saved -> {npy_path}")
                print(f"[HEATMAP-QKV:{sp}][{tag}] saved -> {csv_path}")

    auto_heatmap_ft = _str2bool(getattr(args, "auto_heatmap_ft", True))
    assert auto_heatmap_ft, "auto_heatmap_ft=False so heatmap FT is disabled"

    if args.do_train:
        if spaces is None or num_classes is None:
            _, num_classes = load_single_loader(args, dataset_name=args.dataset, ratio=1.0, shuffle=False)
            spaces, _, _, _, _ = probe_parameter_spaces(model, args, num_classes, run_dir=None, dataset_name=args.dataset)

      
        if auto_heatmap_ft and ((heatmaps_for_grad_1 is None) or (heatmaps_for_grad_2 is None)):
            dim = int(model.mask_simclr.embedding_dim)
            _, heatmaps1_raw, heatmaps2_raw, _ = compute_qkv_heatmaps_by_param_space_fisher(
                model, args, num_classes, spaces, dim=dim, dataset_name=args.dataset
            )
            heatmaps_for_grad_1 = heatmaps1_raw
            heatmaps_for_grad_2 = heatmaps2_raw

        freeze_space = getattr(args, "freeze_space", "none")

        weighted_train(
            model, args, spaces, num_classes,
            dataset_name=args.dataset,
            heatmaps_for_grad_1=heatmaps_for_grad_1,
            heatmaps_for_grad_2=heatmaps_for_grad_2,
            freeze_space=freeze_space
        )

    if args.do_test:
        vis_method = str(getattr(args, "emb_vis_method", "tsne"))
        visualize_test_binary_2d(model, args, run_dir=run_dir, method=vis_method, max_points=2000)

        res = run_test(model, args)
        save_json(os.path.join(run_dir, "test_metrics.json"), res if isinstance(res, dict) else {"result": res})
        print(f"[LOG] saved test metrics -> {os.path.join(run_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
