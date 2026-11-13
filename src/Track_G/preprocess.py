"""preprocess.py
Prepare log1p-normalized expression and kernel matrices for PA and CCA.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .gene_filtering import filter_and_standardize_expression
from .CCA import build_kernel_matrix

def prepare_for_cca(
    log_expr: pd.DataFrame,
    kernels: dict[str, np.ndarray],
    kernel_names: Sequence[str],
    min_expr_frac: float = 0.01,
    min_gene_std: float = 1e-6,
    frobenius_normalize: bool = False,
    use_sqrt: bool = True,
    cast_float32: bool = False,
) -> dict:
    """Preprocess log1p-normalized expression and a set of kernels for PA and CCA.

    Parameters
    ----------
    log_expr : (n, p) DataFrame of log1p-normalized gene expression (cells x genes).
    kernels : dict of (n, n) kernel matrices, e.g. from kernel_from_newick or kernel_from_categorical.
    kernel_names : ordered list of keys into ``kernels`` that defines the column-block order of K.
    min_expr_frac : keep genes expressed in more than this fraction of cells (default 0.01 = 1%).
    min_gene_std : minimum per-gene std to retain a gene; removes near-constant genes (default 1e-6).
    frobenius_normalize : divide each kernel by its Frobenius norm before concatenation (default False).
    use_sqrt : use the symmetric square root K^1/2 of each kernel (default True, as in Track-G).

    Returns
    -------
    dict with keys:
        X2_np            : (n, q) float64 array, filtered and standardized expression.
        valid_genes      : list of retained gene names.
        K                : (n, m*n) concatenated kernel matrix, input to CCA.
        kernels_for_cca  : dict of processed per-kernel matrices (after sqrt / Frobenius normalization).
        n_cells, p_gene  : dimensions of X2_np.
    """
    # expression filtering and standardization
    expression = filter_and_standardize_expression(log_expr, min_expr_frac, min_gene_std)
    print(f"Genes retained: {len(expression['valid_genes'])} / {log_expr.shape[1]} "
          f"(std>{min_gene_std:.2g}, expressed in >{min_expr_frac:.1%} of cells)")

    # kernel concatenation
    if cast_float32:
        kernels = {name: K.astype(np.float32).astype(float) for name, K in kernels.items()}
    kernels = {name: np.asarray(kernels[name], dtype=float) for name in kernel_names}
    K, kernels_for_cca = build_kernel_matrix(kernels, kernel_names, frobenius_normalize, use_sqrt)

    n_cells, p_gene = expression["X2_np"].shape
    assert K.shape == (n_cells, len(kernel_names) * n_cells), \
        f"K shape {K.shape} inconsistent with {n_cells} cells and {len(kernel_names)} kernels"

    return dict(
        K=K,
        kernels_for_cca=kernels_for_cca,
        n_cells=n_cells,
        p_gene=p_gene,
        **expression,
    )
