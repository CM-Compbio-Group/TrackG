"""ALPINE (covariate-guided NMF) on a cells x genes log-expression table: simulations, KP-Tracer, C. elegans.

``run_alpine`` filters the genes with ``Track_G.gene_filtering.filter_genes`` (the same genes as
the CCA), clips the expression at 0, fits ALPINE with one or more categorical cell covariates and returns its
factor matrices as labelled DataFrames.  Gene scores, ranks and FP curves are computed by the caller.

>>> from alpine_baseline import run_alpine
>>> out = run_alpine(log_exp, cell_metadata["SubTumor"], n_covariate_components=9)
>>> out["W"]["SubTumor"]      # genes x 9 guided (SubTumor-specific) components
>>> out["W"]["unguided"]      # genes x 1 unguided (shared) component
>>> pd.to_pickle(out, path)   # the result is a dict of DataFrames

Defaults follow the collaborator's KPtracer_alpine_subtumor.py (2026-10-21): 1 unguided + 9 guided components,
lam = 1e3, alpha_W = 0, Frobenius loss, seed 42, genes expressed in > 1 % of cells with std > 1e-6.
ALPINE rescales every W column to sum to 1, so the size of a component is carried by H, not W.
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"   # repo src/: the Track_G package
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from Track_G import gene_filtering  # noqa: E402  (the gene filter shared with Track-G)


def filter_expression(log_exp, min_expr_frac: float = 0.01, min_gene_std: float = 1e-6) -> pd.DataFrame:
    """cells x genes table -> the genes kept by ``Track_G.gene_filtering.filter_genes`` (std with ddof = 0).

    An array is accepted too; its cells and genes are then named by position ("0", "1", ...).
    """
    log_exp = log_exp if isinstance(log_exp, pd.DataFrame) else pd.DataFrame(np.asarray(log_exp))
    log_exp = log_exp.astype(float)
    log_exp.index, log_exp.columns = log_exp.index.astype(str), log_exp.columns.astype(str)
    if log_exp.isna().any().any():
        raise ValueError("log_exp contains NaN; fill it first (the KP-Tracer scripts use fillna(0)).")
    _, filtered, _ = gene_filtering.filter_genes(
        log_exp, min_expr_frac=min_expr_frac, min_gene_std=min_gene_std, verbose=False)
    return filtered


def _nonnegative(filtered: pd.DataFrame) -> np.ndarray:
    X = filtered.to_numpy(dtype=np.float32)
    if (X < 0).any():
        warnings.warn(f"{(X < 0).mean():.1%} of the expression entries are negative and set to 0 (NMF needs X >= 0).")
    return np.maximum(X, 0)


def run_alpine(log_exp, covariates, n_covariate_components: Union[int, Sequence[int]] = 9, n_components: int = 1,
               *, lam: Union[float, Sequence[float]] = 1e3, loss_type: str = "frobenius", alpha_W: float = 0.0,
               max_iter: int = None, device: str = "auto", random_state: int = 42,
               min_expr_frac: float = 0.01, min_gene_std: float = 1e-6, verbose: bool = False) -> Dict[str, Any]:
    """Fit ALPINE and return its factor matrices.

    log_exp : cells x genes log-expression (not standardized), a DataFrame indexed by cell name.
    covariates : categorical cell labels: a Series (one covariate, its name is the key) or a DataFrame (one
        column per covariate), indexed by the cell names of ``log_exp``; a plain array is taken in row order.
    n_covariate_components : guided components per covariate (one int, or one per covariate).
    n_components : unguided components, shared across covariates (ALPINE needs at least 1).
    lam : weight of each covariate's label-prediction loss (one float, or one per covariate).
    max_iter : None = ALPINE chooses it from the elbow of a 200-iteration warm-up run.
    device : "auto" = CUDA if available, else CPU (KP-Tracer, 672 genes: ~15 s on CPU).
    min_expr_frac, min_gene_std : gene filter (``Track_G.gene_filtering.filter_genes``); 0 skips a criterion.

    Returns a dict
      W            {covariate: genes x guided components, "unguided": genes x unguided components}, columns sum to 1
      H            {same keys: cells x components}
      B            {covariate: label x guided components}
      gene_scores  {covariate: genes x label}, ALPINE's ``get_covariate_gene_scores`` (W H Y^T / label size)
      loss_history DataFrame of the losses per iteration of the final fit
      genes, cells Index of the filtered genes / the cells;  settings  dict of the run settings
    """
    import torch
    from alpine import ALPINE

    filtered = filter_expression(log_exp, min_expr_frac, min_gene_std)
    genes, cells = filtered.columns, filtered.index
    if isinstance(covariates, (pd.Series, pd.DataFrame)):
        cov = covariates.to_frame() if isinstance(covariates, pd.Series) else covariates.copy()
        cov.index = cov.index.astype(str)
        cov = cov.reindex(cells)
    else:
        cov = pd.DataFrame({"label": np.asarray(covariates)}, index=cells)
    if cov.isna().any().any():
        raise ValueError("Some cells of log_exp have no covariate label.")
    keys = [str(k) for k in cov.columns]
    k_cov = [int(n_covariate_components)] * len(keys) if np.isscalar(n_covariate_components) \
        else [int(n) for n in n_covariate_components]
    lams = [float(lam)] * len(keys) if np.isscalar(lam) else [float(v) for v in lam]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(random_state)                            # as set_all_seeds in the collaborator's script
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    adata = ad.AnnData(
        X=_nonnegative(filtered),
        obs=pd.DataFrame({k: pd.Categorical(cov[k].astype(str).to_numpy()) for k in keys}, index=cells),
        var=pd.DataFrame(index=genes))
    model = ALPINE(n_components=int(n_components), n_covariate_components=k_cov, lam=lams, alpha_W=float(alpha_W),
                   loss_type=loss_type, device=device, random_state=int(random_state))
    model.fit(adata, covariate_keys=keys, max_iter=None if max_iter is None else int(max_iter), verbose=verbose)
    M = model.get_decomposed_matrices()

    # ALPINE orders Ws / Hs as [covariate 1, ..., covariate m, unguided]; Bs as [covariate 1, ..., covariate m].
    blocks = keys + ["unguided"]
    comp = {b: [f"{b}_comp_{i + 1}" for i in range(n)] for b, n in zip(blocks, k_cov + [int(n_components)])}
    labels = {k: [s[len(k) + 1:] if s.startswith(k + "_") else s for s in model.fe.encoded_labels[k]] for k in keys}
    W = {b: pd.DataFrame(w, index=genes, columns=comp[b]) for b, w in zip(blocks, M["Ws"])}
    H = {b: pd.DataFrame(h.T, index=cells, columns=comp[b]) for b, h in zip(blocks, M["Hs"])}
    B = {k: pd.DataFrame(b, index=labels[k], columns=comp[k]) for k, b in zip(keys, M["Bs"])}
    gene_scores = {k: df.set_axis(labels[k], axis=1) for k, df in model.get_covariate_gene_scores().items()}

    settings = dict(method="ALPINE", n_cells=len(cells), n_genes=len(genes), min_expr_frac=float(min_expr_frac),
                    min_gene_std=float(min_gene_std), covariates=keys, n_covariate_components=k_cov,
                    n_components=int(n_components), lam=lams, alpha_W=float(alpha_W), loss_type=loss_type,
                    max_iter=int(model.max_iter), device=device, random_state=int(random_state),
                    gene_filter_file=gene_filtering.__file__)
    return dict(W=W, H=H, B=B, gene_scores=gene_scores, loss_history=model.loss_history.copy(),
                genes=genes, cells=cells, settings=settings)
