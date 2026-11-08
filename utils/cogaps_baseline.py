"""CoGAPS (Bayesian NMF, PyCoGAPS) on a cells x genes log-expression table: simulations, KP-Tracer, C. elegans.

``run_cogaps`` filters the genes with ``Track_G.gene_filtering.filter_genes`` (the same genes as
the CCA), clips the expression at 0, runs CoGAPS and returns the amplitude (gene) and pattern (cell) matrices as
labelled DataFrames.  CoGAPS is unsupervised: a pattern is not tied to any kernel / covariate, so the caller maps
patterns to phylo / subtumor / ... before computing gene scores, ranks and FP curves.

>>> from cogaps_baseline import run_cogaps
>>> out = run_cogaps(log_exp, n_patterns=J)    # J = number of CCA components
>>> out["A"]                  # genes x patterns (Amean, gene amplitudes)
>>> out["P"]                  # cells x patterns (Pmean)
>>> pd.to_pickle(out, path)   # the result is a dict of DataFrames; cache it, a full run takes long

``n_patterns`` has no default: it is set to the CCA's number of components (J), not to the 18 of the collaborator's
KPTracer_cogaps.py. Other defaults follow that script: 50,000 iterations, seed 42, sparse optimization.
PyCoGAPS = GitHub master bc12674 in /home/yjunfeng/software/pycogaps, built into review-eecs545 with OpenMP
(setup.py: -fopenmp -D__GAPS_OPENMP__).
Deliberate differences from the script:
- The data are passed to CoGAPS as genes x cells (its own layout, transposeData=False).  The script's
  transposeData=True on a cells x genes matrix returns a broken pattern matrix in this PyCoGAPS (entries ~ -1e28 on
  a rank-3 test matrix), while genes x cells recovers the true factors (|r| = 1.00).
- Single (standard) CoGAPS only.  The script's ``setDistributedParams(nSets=7)`` had no effect (``distributed``
  was never set), so it also ran a single chain; both distributed modes of this PyCoGAPS return non-finite /
  negative A ("genome-wide") or P ("single-cell") entries on the same test matrix.
- ``n_threads > 1`` runs CoGAPS's asynchronous (OpenMP) sampler on that many threads; ``n_threads = 1`` (default)
  is the sequential sampler the script ran, unchanged. KP-Tracer, 672 genes, 9 patterns, 1,000 iterations:
  sequential 275 s, asynchronous 1 / 2 / 4 threads 312 / 174 / 98 s (same chi-square). Asynchronous runs are not
  bit-identical to the sequential sampler.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

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


def run_cogaps(log_exp, n_patterns: int, *, n_iterations: int = 50000, seed: int = 42,
               use_sparse_optimization: bool = True, n_threads: int = 1, min_expr_frac: float = 0.01,
               min_gene_std: float = 1e-6, messages: bool = False, output_frequency: int = 1000) -> Dict[str, Any]:
    """Run CoGAPS and return its factor matrices.

    log_exp : cells x genes log-expression (not standardized), a DataFrame indexed by cell name.
    n_patterns : number of patterns (nPatterns), required; use the CCA's J.
    n_iterations, seed, use_sparse_optimization : CoGAPS parameters (nIterations, seed, sparse optimization).
    n_threads : 1 = sequential sampler; > 1 = asynchronous sampler on n_threads OpenMP threads (faster).
    min_expr_frac, min_gene_std : gene filter (``Track_G.gene_filtering.filter_genes``); 0 skips a criterion.
    messages : print CoGAPS progress every ``output_frequency`` iterations.

    Returns a dict
      A, A_sd      genes x patterns: amplitude matrix (posterior mean) and its posterior SD
      P, P_sd      cells x patterns: pattern matrix (posterior mean) and its posterior SD
      mean_chisq   final mean chi-square;  chisq_history  chi-square every ``output_frequency`` iterations
      genes, cells Index of the filtered genes / the cells;  settings  dict of the run settings
    """
    with contextlib.redirect_stdout(io.StringIO()):     # PyCoGAPS prints a banner on import
        from PyCoGAPS.parameters import CoParams, setParams
        from PyCoGAPS.pycogaps_main import CoGAPS

    filtered = filter_expression(log_exp, min_expr_frac, min_gene_std)
    genes, cells = filtered.columns, filtered.index
    adata = ad.AnnData(X=np.ascontiguousarray(_nonnegative(filtered).T),   # genes x cells
                       obs=pd.DataFrame(index=genes), var=pd.DataFrame(index=cells))
    params = CoParams(adata)
    setParams(params, {"nIterations": int(n_iterations), "seed": int(seed), "nPatterns": int(n_patterns),
                       "useSparseOptimization": bool(use_sparse_optimization)})
    out = contextlib.nullcontext() if messages else contextlib.redirect_stdout(io.StringIO())
    with out:
        threads = {} if int(n_threads) <= 1 else dict(nThreads=int(n_threads), asynchronousUpdates=True)
        result = CoGAPS(adata, params, messages=messages, outputFrequency=int(output_frequency), **threads)

    def _by_name(M, names):
        M = pd.DataFrame(M).astype(float)
        M.index, M.columns = M.index.astype(str), M.columns.astype(str)
        if not M.index.is_unique or set(M.index) != set(names):
            raise RuntimeError("CoGAPS returned rows that do not match the input genes / cells.")
        return M.reindex(names)
    A, A_sd = _by_name(result.obs, genes), _by_name(result.uns["asd"], genes)
    P, P_sd = _by_name(result.var, cells), _by_name(result.uns["psd"], cells)
    for name, M in (("A", A), ("P", P)):                 # garbage such as -1e28 = a broken PyCoGAPS code path
        if not (np.isfinite(M.to_numpy()).all() and (M.to_numpy() >= 0).all()):
            raise RuntimeError(f"CoGAPS returned a non-finite or negative {name} matrix.")

    uns = result.uns
    settings = dict(method="CoGAPS", n_cells=len(cells), n_genes=len(genes), min_expr_frac=float(min_expr_frac),
                    min_gene_std=float(min_gene_std), n_patterns=int(n_patterns), n_iterations=int(n_iterations),
                    seed=int(seed), use_sparse_optimization=bool(use_sparse_optimization), n_threads=int(n_threads),
                    sampler="sequential" if int(n_threads) <= 1 else "asynchronous",
                    total_running_time_s=uns.get("totalRunningTime"),
                    gene_filter_file=gene_filtering.__file__)
    return dict(A=A, P=P, A_sd=A_sd, P_sd=P_sd, mean_chisq=uns.get("meanChiSq"),
                chisq_history=np.asarray(uns.get("chisqHistory", []), float),
                genes=genes, cells=cells, settings=settings)
