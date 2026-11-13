"""kernel_construction.py
Utilities for building kernel matrices from common biological data types.

Two constructors are provided:

    kernel_from_newick      : lineage kernel from a Newick tree file (LCA depth).
    kernel_from_categorical : co-membership kernel from a categorical label vector.

All kernels are returned as (n, n) float64 ndarrays with unit diagonal where applicable.
They can be passed directly to ``build_kernel_matrix`` in Track_G for use in CCA.

Usage
-----
>>> from kernel_construction import kernel_from_newick, kernel_from_categorical
>>> K_lineage  = kernel_from_newick("tree.nwk", cell_names)
>>> K_location = kernel_from_categorical(cell_metadata["SubTumor"])
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from Bio import Phylo
from scipy import sparse

def kernel_from_newick(
    path: str | Path,
    cell_names: Sequence[str],
    normalize: bool = True,
) -> np.ndarray:
    """Build a lineage kernel from a Newick tree file.

    K[i, j] = number of edges shared by the root-to-cell paths of cells i and j
    (i.e. the depth of their lowest common ancestor; every edge has length 1).

    Parameters
    ----------
    path : path to the Newick tree file.
    cell_names : ordered list of cell names; the kernel rows/columns follow this order.
    normalize : if True (default), scale to unit diagonal (correlation matrix).

    Returns
    -------
    K : (n, n) ndarray, dtype float64.
    """
    tree = Phylo.read(Path(path), "newick")

    parent = {child: clade for clade in tree.find_clades() for child in clade.clades}
    edge = {node: e for e, node in enumerate(parent)}
    leaf = {str(t.name): t for t in tree.get_terminals()}

    missing = [c for c in cell_names if c not in leaf]
    if missing:
        raise ValueError(f"{len(missing)} cell(s) not found in tree: {missing[:5]}")

    n = len(cell_names)
    rows, cols = [], []
    for i, cell in enumerate(cell_names):
        node = leaf[cell]
        while node in parent:
            rows.append(i)
            cols.append(edge[node])
            node = parent[node]

    B = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, len(edge)),
    )
    K = (B @ B.T).toarray().astype(float)

    if normalize:
        d = np.sqrt(np.diag(K))
        K = K / np.outer(d, d)

    return K

def kernel_from_categorical(
    labels: pd.Series | np.ndarray,
) -> np.ndarray:
    """Build a kernel from a categorical variable.

    K[i, j] = 1 if cells i and j share the same category, 0 otherwise.
    The diagonal is always 1 (each cell belongs to exactly one category).

    Parameters
    ----------
    labels : (n,) categorical vector; a pandas Series or a 1-D array of labels.

    Returns
    -------
    K : (n, n) ndarray, dtype float64.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

    B = pd.get_dummies(labels).to_numpy().astype(float)
    return B @ B.T
