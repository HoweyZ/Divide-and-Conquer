import os
import torch
from torch_geometric.datasets import TUDataset

def pyg_graph_to_txt_lines(data):
    n = int(data.num_nodes)
    y = int(data.y.item()) if getattr(data, "y", None) is not None else 0

    # Build undirected adjacency list (0-based)
    adj = [[] for _ in range(n)]
    if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
        ei = data.edge_index.cpu()
        for u, v in ei.t().tolist():
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].append(v)
                adj[v].append(u)  # make it undirected

        # dedup + sort
        for u in range(n):
            adj[u] = sorted(set(adj[u]))

    # Determine discrete node tags
    # - If x exists and looks like categorical/one-hot => argmax as tag
    # - Otherwise fallback to degree as tag
    tags = [0] * n
    x = getattr(data, "x", None)
    if isinstance(x, torch.Tensor) and x.dim() == 2 and x.size(0) == n and x.numel() > 0:
        x = x.cpu()
        tags = torch.argmax(x, dim=1).tolist()
    else:
        tags = [len(adj[u]) for u in range(n)]

    lines = []
    lines.append(f"{n} {y}")
    for u in range(n):
        neigh = adj[u]
        deg = len(neigh)
        row = [str(int(tags[u])), str(deg)] + [str(v) for v in neigh]
        lines.append(" ".join(row))
    return lines

def export_one(dataset_name: str, pyg_root: str, out_root: str):
    ds = TUDataset(root=pyg_root, name=dataset_name)

    out_dir = os.path.join(out_root, dataset_name, "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}.txt")

    with open(out_path, "w") as f:
        f.write(str(len(ds)) + "\n")
        for data in ds:
            for line in pyg_graph_to_txt_lines(data):
                f.write(line + "\n")

    print(f"[OK] {dataset_name}: wrote {out_path} | num_graphs={len(ds)}")

def export_all(
    pyg_root="/root/CellCLAT-main/.pyg_tu_cache",
    out_root="/root/CellCLAT-main/datasets",
):
    # Mapping from your table names to PyG TU dataset names when needed
    datasets = [
        "NCI1",
        "PROTEINS",
        "MUTAG",
        "NCI109",
        "DD",
        "AIDS",
        "COX2_MD",
        "Mutagenicity",
        "MCF-7",
        "PTC_FM",
        "PTC_FR",
        "PTC_MM",
        "PTC_MR"
    ]

    for name in datasets:
        export_one(name, pyg_root=pyg_root, out_root=out_root)

if __name__ == "__main__":
    export_all()
