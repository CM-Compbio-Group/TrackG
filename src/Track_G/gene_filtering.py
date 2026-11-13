"""gene_filtering.py
Gene filtering and standardization of a cells x genes log-expression table.

Usage
-----
>>> from Track_G import filter_genes, filter_and_standardize_expression
>>> valid_genes, log_exp_filtered, gene_frac_expr = filter_genes(log_exp)
>>> valid_genes, log_exp_filtered, gene_frac_expr = filter_genes(log_exp, min_expr_frac=0.05, min_gene_std=1e-4)
>>> expr = filter_and_standardize_expression(log_exp)        # expr["X2_np"]: standardized filtered genes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def filter_genes(
    log_exp: pd.DataFrame,
    *,
    min_expr_frac: float = 0.01,
    min_gene_std: float = 1e-6,
    verbose: bool = True,
) -> Tuple[pd.Index, pd.DataFrame, pd.Series]:
    """Filter genes by expression fraction and standard deviation.

    Parameters
    ----------
    log_exp:
        cells x genes log-expression DataFrame (already aligned to desired cell order).
    min_expr_frac:
        Minimum fraction of cells expressing a gene (> 0). 0.01 = 1%.
        Pass 0.0 to skip.
    min_gene_std:
        Minimum per-gene std (numpy ddof=0). Default 1e-6 removes near-constant genes.
        Pass 0.0 to skip.

    Returns
    -------
    valid_genes:      pd.Index of retained gene names.
    log_exp_filtered: pd.DataFrame cells x filtered-genes.
    gene_frac_expr:   pd.Series, fraction of expressing cells for every input gene.
    """
    if not 0.0 <= min_expr_frac < 1.0:
        raise ValueError(f"min_expr_frac must be in [0, 1); received {min_expr_frac}.")
    if not np.isfinite(min_gene_std) or min_gene_std < 0.0:
        raise ValueError(f"min_gene_std must be a finite non-negative number; received {min_gene_std}.")

    gene_frac_expr = (log_exp > 0).sum(axis=0) / len(log_exp)
    gene_stds = log_exp.values.std(axis=0)  # numpy ddof=0

    keep = np.ones(log_exp.shape[1], dtype=bool)
    if min_gene_std > 0.0:
        keep &= gene_stds > min_gene_std
    if min_expr_frac > 0.0:
        keep &= gene_frac_expr.values > min_expr_frac

    valid_genes = log_exp.columns[keep]
    log_exp_filtered = log_exp[valid_genes]

    if verbose:
        print(f"Genes retained (std>{min_gene_std:.2g} & expr>{min_expr_frac*100:.1f}%): {len(valid_genes)}")

    return valid_genes, log_exp_filtered, gene_frac_expr


def filter_and_standardize_expression(log_exp: pd.DataFrame, min_expr_frac: float = 0.01, min_gene_std: float = 1e-6) -> Dict[str, Any]:
    """Gene filter + per-gene standardization of a cells x genes expression table."""
    log_exp = log_exp.astype(float)

    valid_genes, log_exp_filtered, gene_frac_expr = filter_genes(
        log_exp, min_expr_frac=min_expr_frac, min_gene_std=min_gene_std
    )

    # Gene standardization (zero-mean, unit-variance across cells)
    X_vals = log_exp_filtered.values
    X2_np = (X_vals - X_vals.mean(axis=0)) / X_vals.std(axis=0)

    return dict(X2_np=X2_np, log_exp_filtered=log_exp_filtered, valid_genes=valid_genes,
                gene_frac_expr=gene_frac_expr)
