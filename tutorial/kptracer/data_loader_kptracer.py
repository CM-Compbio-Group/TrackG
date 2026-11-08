"""data_loader_kptracer.py
Dataset-specific data loader for the KP-Tracer analysis (tumor family 3454_Lkb1_All).

Everything is computed from the raw inputs in ``tutorial/data/kptracer/`` (KP-Tracer, Yang et al. 2022, Cell):
    3454_Lkb1_All_counts.h5ad : raw UMI counts (``adata_processed.combined.h5ad`` .X, 4000 genes) of the 1846 cells
                                of the lineage tree, in tree-leaf order, with the cell labels Tumor and SubTumor
    3454_Lkb1_All_tree.nwk    : the lineage tree
Expression: genes detected in at least one cell -> counts per 10^4 -> log1p.
Kernels (scaled to unit diagonal): lineage = number of edges shared by the root-to-cell paths of two cells (depth of
their lowest common ancestor); location = 1 for two cells of the same SubTumor (tumor location), 0 otherwise.
The generic methods (gene filtering, standardization, kernel concatenation) are imported from Track_G.

Usage
-----
>>> from data_loader_kptracer import load_kptracer_data
>>> data = load_kptracer_data()                     # Track-G inputs: 1 % genes, K = [K_lineage^1/2 | K_location^1/2]
>>> data = load_kptracer_data(min_expr_frac=0.05, use_sqrt=False)
"""

from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy import sparse

from Track_G import (
    build_kernel_matrix,
    filter_and_standardize_expression,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "kptracer"
TREE_ID = "3454_Lkb1_All"

# Kernel order defines the K column blocks and the A row blocks: lineage tree, then location (SubTumor).
KERNEL_NAMES = np.array(["Sigma_phylo", "Sigma_subtumor"])


def lca_kernel(tree, cells) -> np.ndarray:
    """Lineage kernel: K[i, j] = number of edges shared by the root->i and root->j paths (every edge has length 1)."""
    parent = {child: clade for clade in tree.find_clades() for child in clade.clades}
    edge = {node: e for e, node in enumerate(parent)}          # one edge above every non-root node
    leaf = {str(t.name): t for t in tree.get_terminals()}
    rows, cols = [], []
    for i, cell in enumerate(cells):
        node = leaf[cell]
        while node in parent:                                   # walk up to the root
            rows.append(i)
            cols.append(edge[node])
            node = parent[node]
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells), len(edge)))   # cells x edges
    return (B @ B.T).toarray()


def unit_diagonal(S: np.ndarray) -> np.ndarray:
    """Scale a covariance to unit diagonal (correlation matrix)."""
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def load_kptracer_data(min_expr_frac: float = 0.01, min_gene_std: float = 1e-6, frobenius_normalize: bool = False,
                       use_sqrt: bool = True) -> Dict[str, Any]:
    """Load the raw KP-Tracer inputs and build the Track-G inputs.

    min_expr_frac : keep genes expressed (> 0) in more than this fraction of cells, e.g. 0.01 = 1 % (default,
        672 genes), 0 = no expression filter.
    min_gene_std : minimum per-gene std (numpy ddof=0) to retain a gene. Default 1e-6 removes near-constant genes.
    frobenius_normalize : divide each kernel by its Frobenius norm before the concatenation (default False).
    use_sqrt : use the symmetric square root K^1/2 of each kernel (default True = Track-G); False uses K itself.
    """
    tree = Phylo.read(DATA_DIR / f"{TREE_ID}_tree.nwk", "newick")
    counts = ad.read_h5ad(DATA_DIR / f"{TREE_ID}_counts.h5ad")
    cell_names = counts.obs_names.astype(str).to_numpy()          # tree-leaf order
    cell_metadata = counts.obs[["Tumor", "SubTumor"]].copy()

    # Log expression, as in the KP-Tracer preprocessing: genes in >= 1 cell, counts per 10^4, log1p (float32).
    X = counts.X.toarray().astype(np.float32)
    keep = X.sum(0) > 0
    X = X[:, keep]
    X = np.log1p(X / (X.sum(1, keepdims=True) / np.float32(1e4)))   # = scanpy normalize_total(1e4) + log1p
    log_exp = pd.DataFrame(X, index=cell_names, columns=counts.var_names[keep].astype(str))

    expression = filter_and_standardize_expression(log_exp, min_expr_frac, min_gene_std)
    print(f"Total genes retained after filtering: {len(expression['valid_genes'])} / {log_exp.shape[1]} "
          f"(std>{min_gene_std:.2g}, expressed in > {min_expr_frac:.1%} of cells)")

    # Kernels with unit diagonal, stored as float32 like the preprocessed kernels of the paper.
    location = pd.get_dummies(cell_metadata["SubTumor"].astype(str), dtype=float).to_numpy()
    kernels_raw = {
        "Sigma_phylo": unit_diagonal(lca_kernel(tree, cell_names)),
        "Sigma_subtumor": unit_diagonal(location @ location.T),
    }
    kernels_raw = {name: K.astype(np.float32).astype(float) for name, K in kernels_raw.items()}
    K, kernels_for_cca = build_kernel_matrix(kernels_raw, KERNEL_NAMES, frobenius_normalize, use_sqrt)
    n_cells, p_gene = expression["X2_np"].shape
    assert K.shape == (n_cells, len(KERNEL_NAMES) * n_cells)

    return dict(
        cell_names=cell_names, kernels_raw=kernels_raw, kernels_for_cca=kernels_for_cca, K=K,
        cell_metadata=cell_metadata, tree=tree, log_exp=log_exp, n_cells=n_cells, p_gene=p_gene,
        KERNEL_NAMES=KERNEL_NAMES.copy(), min_expr_frac=float(min_expr_frac), min_gene_std=float(min_gene_std),
        frobenius_normalize=bool(frobenius_normalize), use_sqrt=bool(use_sqrt),
        **expression,
    )
