"""
visualize_bfs.py

Visual diagnostics for the BFS-Inspired Neural Network (BFSNet).

What this script provides
-------------------------
1) Runs a forward pass through BFSNet (with return_trace=True) to collect:
   - num_nodes_per_depth
   - spawn_counts_sum
   - active_mask_sums
2) Computes derived metrics:
   - avg_children_per_example/depth = spawn_counts_sum / num_examples
   - avg_children_per_active_parent/depth = spawn_counts_sum / active_mask_sums
3) Visualizes:
   A) A layered "schematic" BFS tree based on num_nodes_per_depth
      (edges are schematic; counts per depth are accurate to model trace)
   B) Bar charts for all trace metrics and derived metrics

Updates in this version
-----------------------
- Added --dataset choices: toy, mnist, fashionmnist, text_bow.
- When a dataset is selected, auto-infer input_dim (and num_classes)
  from the loader and construct BFSNet accordingly.
- Manual flags (--input_dim, --output_dim) remain; a summary shows both
  inferred and effective dims for sanity checking.

Author: William McKeon
Date: 2025-08-21
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from bfs_model import BFSNet

# Optional support for data loading.
# Try utils.data_utils first (your layout), then fall back to top-level data_utils.
_HAS_DATA_UTILS = False
try:
    from utils.data_utils import get_dataloaders, set_seed as du_set_seed  # type: ignore
    _HAS_DATA_UTILS = True
except Exception:
    try:
        from data_utils import get_dataloaders, set_seed as du_set_seed  # type: ignore
        _HAS_DATA_UTILS = True
    except Exception:
        _HAS_DATA_UTILS = False

# Optional graph drawing
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#                             Helper: Seeding & Data                          #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """Deterministic CPU seeding (mirrors usage in other scripts)."""
    torch.manual_seed(seed)
    if _HAS_DATA_UTILS:
        du_set_seed(seed)


def build_input_from_args(
    args: argparse.Namespace
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[int], Optional[int]]:
    """
    Build an input batch X according to CLI flags.

    Returns
    -------
    X : torch.Tensor [B, D]
    dataset_info : Optional[(X_full, y_full)]
        If a dataset was loaded, returns (one small batch of inputs, labels)
        for reference; otherwise None.
    inferred_input_dim : Optional[int]
        If dataset mode is used, the inferred input dimension; else None.
    inferred_num_classes : Optional[int]
        If dataset mode is used, the inferred number of classes; else None.
    """
    if args.dataset is None:
        # Manual token mode (toy-style)
        D = args.input_dim
        if args.token.upper() == "A":
            x = torch.tensor([[1.0] * D])
        elif args.token.upper() == "B":
            x = torch.tensor([[-1.0] * D])
        else:
            raise ValueError("Unknown token. Use 'A' or 'B' when --dataset is not provided.")
        return x, None, None, None

    # Dataset mode via data_utils
    if not _HAS_DATA_UTILS:
        raise ImportError("data_utils.py not found, but --dataset was specified. "
                          "Either add data_utils.py (or utils/data_utils.py) or omit --dataset and use --token.")

    name = args.dataset.lower().strip()
    if name == "toy":
        train_loader, _, D, C = get_dataloaders(
            "toy",
            batch_size=args.viz_batch_size,
            seed=args.seed,
            toy_input_dim=args.input_dim,               # manual override for toy
            toy_samples_per_class=args.samples_per_class,
        )
    elif name == "mnist":
        train_loader, _, D, C = get_dataloaders(
            "mnist",
            batch_size=args.viz_batch_size,
            seed=args.seed,
            mnist_flatten=True,
            mnist_download=True,
        )
    elif name == "fashionmnist":
        train_loader, _, D, C = get_dataloaders(
            "fashionmnist",
            batch_size=args.viz_batch_size,
            seed=args.seed,
            mnist_flatten=True,         # same interface as mnist in data_utils
            mnist_download=True,
        )
    elif name == "text_bow":
        train_loader, _, D, C = get_dataloaders(
            "text_bow",
            batch_size=args.viz_batch_size,
            seed=args.seed,
            text_vocab_size=args.text_vocab_size,
            text_samples_per_class=args.text_samples_per_class,
        )
    else:
        raise ValueError("Unknown --dataset. Use one of: toy, mnist, fashionmnist, text_bow.")

    xb, yb = next(iter(train_loader))  # small batch
    idx = max(0, min(args.sample_index, xb.size(0) - 1))
    x = xb[idx:idx + 1, :]  # [1, D]
    return x, (xb, yb), int(D), int(C)


# --------------------------------------------------------------------------- #
#                        Trace → Derived Metrics & Tree                       #
# --------------------------------------------------------------------------- #

def compute_derived_metrics(
    trace: Dict[str, torch.Tensor],
    num_examples: int,
) -> Dict[str, List[float]]:
    """
    Compute intuition-friendly per-depth metrics.
    """
    out: Dict[str, List[float]] = {}

    nodes = trace["num_nodes_per_depth"].detach().cpu().float()         # [<= D+1]
    spawns = trace["spawn_counts_sum"].detach().cpu().float()           # [<= D]
    active = trace["active_mask_sums"].detach().cpu().float()           # [<= D+1]

    # Avg children per example per depth (use total examples that produced spawns)
    avg_children_per_ex = spawns / max(1, num_examples)                 # [<= D]
    # Avg children per active parent per depth (robust to pruning)
    denom = torch.clamp(active[:-1], min=1e-9) if active.numel() > 1 else active  # parents at each expansion depth
    if denom.numel() == spawns.numel():
        avg_children_per_active = spawns / denom
    else:
        # Pad/truncate safely
        K = min(spawns.numel(), denom.numel())
        avg_children_per_active = spawns[:K] / torch.clamp(denom[:K], min=1e-9)

    out["num_nodes_per_depth"] = nodes.tolist()
    out["spawn_counts_sum"] = spawns.tolist()
    out["active_mask_sums"] = active.tolist()
    out["avg_children_per_example"] = avg_children_per_ex.tolist()
    out["avg_children_per_active_parent"] = avg_children_per_active.tolist()
    return out


def build_schematic_layers(num_nodes_per_depth: List[int]) -> List[List[int]]:
    """
    Build a per-depth list of node IDs (schematic). IDs are sequential.

    Example:
        [1, 2, 5] → depth 0: [0]
                    depth 1: [1,2]
                    depth 2: [3,4,5,6,7]
    """
    layers: List[List[int]] = []
    nid = 0
    for count in num_nodes_per_depth:
        layer = list(range(nid, nid + int(count)))
        layers.append(layer)
        nid += int(count)
    return layers


def build_schematic_edges(layers: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Make simple breadth-wise edges between consecutive layers.
    Each parent connects to roughly ceil(len(next_layer)/len(cur_layer)) children,
    distributing remainder across earliest parents.
    """
    edges: List[Tuple[int, int]] = []
    for d in range(len(layers) - 1):
        parents = layers[d]
        children = layers[d + 1]
        if not parents or not children:
            continue
        base = len(children) // len(parents)
        rem = len(children) % len(parents)
        idx = 0
        for i, p in enumerate(parents):
            take = base + (1 if i < rem else 0)
            for _ in range(take):
                if idx >= len(children):
                    break
                edges.append((p, children[idx]))
                idx += 1
    return edges


# --------------------------------------------------------------------------- #
#                                   Plotting                                  #
# --------------------------------------------------------------------------- #

def _draw_tree_matplotlib(ax, layers: List[List[int]], edges: List[Tuple[int, int]]) -> None:
    """
    Simple layered layout without networkx: scatter nodes and draw lines.
    """
    # Compute coordinates: depth -> y, index within layer -> x
    y_gap = 1.6
    x_gap = 1.2

    coords: Dict[int, Tuple[float, float]] = {}
    for d, layer in enumerate(layers):
        n = max(1, len(layer))
        xs = [x_gap * (i - (n - 1) / 2.0) for i in range(n)]
        y = -d * y_gap
        for node_id, x in zip(layer, xs):
            coords[node_id] = (x, y)

    # Edges
    for u, v in edges:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1.0)

    # Nodes
    xs = [coords[i][0] for layer in layers for i in layer]
    ys = [coords[i][1] for layer in layers for i in layer]
    ax.scatter(xs, ys, s=200)

    # Labels (node IDs)
    for node_id, (x, y) in coords.items():
        ax.text(x, y + 0.15, str(node_id), ha="center", va="bottom", fontsize=8)

    ax.set_title("Schematic BFS Tree (counts per depth)")
    ax.axis("off")


def plot_trace_and_tree(
    metrics: Dict[str, List[float]],
    show_tree: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Render the schematic tree and metric charts.
    """
    num_nodes = metrics["num_nodes_per_depth"]
    spawns = metrics["spawn_counts_sum"]
    active = metrics["active_mask_sums"]
    avg_ex = metrics["avg_children_per_example"]
    avg_act = metrics["avg_children_per_active_parent"]

    # Prepare tree layers/edges
    layers = build_schematic_layers([int(round(x)) for x in num_nodes])
    edges = build_schematic_edges(layers)

    # Figure layout:
    # If tree is requested, use 2x3; else, 1x3 charts only
    if show_tree:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0])
        ax_tree = fig.add_subplot(gs[0, :])
        ax_nodes = fig.add_subplot(gs[1, 0])
        ax_spawns = fig.add_subplot(gs[1, 1])
        ax_active = fig.add_subplot(gs[1, 2])
    else:
        fig, (ax_nodes, ax_spawns, ax_active) = plt.subplots(1, 3, figsize=(14, 4))

    if show_tree:
        # If networkx exists, use it; else use a simple layered plot
        if _HAS_NX:
            G = nx.DiGraph()
            for layer in layers:
                for nid in layer:
                    G.add_node(nid, depth=[i for i, L in enumerate(layers) if nid in L][0])
            G.add_edges_from(edges)
            # Layered positions
            pos = {}
            for d, layer in enumerate(layers):
                n = max(1, len(layer))
                xs = [i - (n - 1) / 2.0 for i in range(n)]
                for nid, x in zip(layer, xs):
                    pos[nid] = (x, -d)
            nx.draw(G, pos, with_labels=True, node_size=700, arrows=False, ax=ax_tree)
            ax_tree.set_title("Schematic BFS Tree (counts per depth)")
            ax_tree.axis("off")
        else:
            _draw_tree_matplotlib(ax_tree, layers, edges)

    # Bar charts
    x_nodes = list(range(len(num_nodes)))
    ax_nodes.bar(x_nodes, num_nodes)
    ax_nodes.set_title("num_nodes_per_depth")
    ax_nodes.set_xlabel("depth")
    ax_nodes.set_ylabel("nodes")

    x_spawns = list(range(len(spawns)))
    ax_spawns.bar(x_spawns, spawns)
    ax_spawns.set_title("spawn_counts_sum (per depth)")
    ax_spawns.set_xlabel("expansion depth")
    ax_spawns.set_ylabel("children spawned")

    x_active = list(range(len(active)))
    ax_active.bar(x_active, active)
    ax_active.set_title("active_mask_sums (per depth)")
    ax_active.set_xlabel("depth")
    ax_active.set_ylabel("active parents (sum)")

    # Add derived metrics as a super-title annotation
    derived_text = (
        f"avg_children_per_example/depth: {['{:.3f}'.format(v) for v in avg_ex]}   |   "
        f"avg_children_per_active_parent/depth: {['{:.3f}'.format(v) for v in avg_act]}"
    )
    fig.suptitle(derived_text, fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[visualize_bfs] Saved figure to: {save_path}")
    else:
        plt.show()


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def build_model_with_dims(
    args: argparse.Namespace,
    effective_input_dim: int,
    effective_output_dim: int
) -> BFSNet:
    """
    Instantiate BFSNet using the *effective* dimensions that will actually be used.
    """
    model = BFSNet(
        input_dim=effective_input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=effective_output_dim,
        max_depth=args.max_depth,
        max_children=args.max_children,
        sibling_embed=not args.no_sibling_embed,
        use_pruning=args.use_pruning,
        pruning_mode=args.pruning_mode,
        pruning_threshold=args.pruning_threshold,
        branch_temperature=args.branch_temperature,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BFSNet expansion and metrics.")
    # Model dims (manual overrides; auto-infer when dataset is selected)
    parser.add_argument("--input_dim", type=int, default=8, help="Manual input feature size (used when --dataset is None, or to override).")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden size.")
    parser.add_argument("--output_dim", type=int, default=2, help="Manual output size (classes).")

    # BFS structure & gates
    parser.add_argument("--max_depth", type=int, default=2, help="Number of BFS expansion steps.")
    parser.add_argument("--max_children", type=int, default=3, help="Max children per node (branch gate chooses 0..K).")
    parser.add_argument("--branch_temperature", type=float, default=0.8, help="Gumbel-Softmax temperature.")
    parser.add_argument("--no_sibling_embed", action="store_true", help="Disable sibling embedding.")

    # Pruning options
    parser.add_argument("--use_pruning", action="store_true", help="Enable pruning during expansion.")
    parser.add_argument("--pruning_mode", type=str, default="learned", choices=["learned", "threshold"],
                        help="Pruning mode if enabled.")
    parser.add_argument("--pruning_threshold", type=float, default=1e-3, help="Threshold for threshold-based pruning.")

    # Data / input control
    parser.add_argument("--dataset", type=str, default=None,
                        choices=[None, "toy", "mnist", "fashionmnist", "text_bow"],
                        help="If set, sample a vector from the dataset to visualize.")
    parser.add_argument("--token", type=str, default="A", help="When --dataset is None: choose 'A' or 'B'.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index within the sampled batch to visualize.")
    parser.add_argument("--viz_batch_size", type=int, default=16, help="Batch size for sampling from dataset.")

    # Dataset-specific knobs
    parser.add_argument("--samples_per_class", type=int, default=100, help="(toy) samples per class.")
    parser.add_argument("--text_vocab_size", type=int, default=256, help="(text_bow) vocabulary size.")
    parser.add_argument("--text_samples_per_class", type=int, default=200, help="(text_bow) samples per class.")

    # Run control
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the figure instead of showing.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Build input vector(s) and infer dims if dataset mode is used
    x, ds_info, inferred_D, inferred_C = build_input_from_args(args)
    B, D_from_x = x.shape

    # Resolve effective dims for the model
    # Use inferred dimensions from dataset, or fall back to manual args
    if inferred_D is not None:
        effective_input_dim = inferred_D
        effective_output_dim = inferred_C if inferred_C is not None else args.output_dim
    else:
        # Manual token mode
        effective_input_dim = args.input_dim
        effective_output_dim = args.output_dim

    # Construct model with effective dims
    model = build_model_with_dims(args, effective_input_dim, effective_output_dim)

    # Print quick summary
    print(model.summary())
    if inferred_D is not None:
        print(f"[dims] inferred: input_dim={inferred_D}, num_classes={inferred_C} | "
              f"effective (used to build model): input_dim={effective_input_dim}, output_dim={effective_output_dim}")
    else:
        print(f"[dims] manual mode: input_dim={effective_input_dim}, output_dim={effective_output_dim}")

    # Forward with trace
    with torch.no_grad():
        out, trace = model(x, return_trace=True)

    print("Input batch shape:", x.shape)
    print("Output shape:", out.shape)
    print("Trace:")
    for k, v in trace.items():
        print(f"  {k}: {v}")

    # Metrics + figure
    metrics = compute_derived_metrics(trace, num_examples=B)
    save_path = args.save.strip() or None
    plot_trace_and_tree(metrics, show_tree=True, save_path=save_path)


if __name__ == "__main__":
    main()