# Parallel analysis
"""Parallel analysis (PA) for choosing the number of multi-kernel CCA components.

The CCA decomposes the cross-covariance ``Y = K^T X``, where ``K = [K_1 ... K_m]`` (n x mn) is the
concatenation of the (normalized) cell kernels and ``X`` (n x p) is the gene-standardized expression.
PA compares the ordered singular values of ``Y`` with the same-order quantile of a permutation null
and stops at the first failure (sequential Buja-Eyuboglu rule).

Three nulls are available via the ``null`` argument:

``"entrywise"`` (default)
    Shuffle all entries of ``Y = K^T X`` jointly.  Conservative noise-floor stopping rule.
``"row_permute"``
    Permute the cells (rows) of ``X``.  Keeps ``K K^T`` and gene-gene covariance.
``"expression"``
    Permute every gene of ``X`` independently (Horn PA on expression self-rank; ignores kernels).

Usage
-----
>>> from Track_G import run_parallel_analysis, plot_parallel_analysis
>>> pa = run_parallel_analysis(X2_np, K)             # entrywise null by default
>>> pa = run_parallel_analysis(X2_np, K, null="row_permute")   # users may configure other permutation e.g. row permutation if they have good reasons
>>> pa.J                                             # number of components to fit
>>> pa.result                                        # dict with singular_values, threshold, etc.
>>> pa.summary                                       # one-row DataFrame
>>> fig = plot_parallel_analysis(pa, path="parallel_analysis.png")

Each per-null result is a dict with keys ``kind, singular_values, threshold, null_singular_values,
above_threshold, n_components, n_above, quantile, n_permutations, at_bound``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg

NULLS = ("entrywise", "row_permute", "expression")
DESCRIPTIONS = {
    "entrywise": "cross-covariance (K^T X), global entry-wise null",
    "row_permute": "cross-covariance (K^T X), row-permute null",
    "expression": "expression self-rank (X), per-gene null",
}
PLOT_TITLES = {
    "entrywise": "Cross-cov PA (K^T X), global entry-wise null",
    "row_permute": "Cross-cov PA (K^T X), row-permute null",
    "expression": "Expression self-rank PA (X)",
}


def _sequential_select(observed, threshold):
    """Stop at the first ordered singular value that fails its null quantile."""
    above = observed > threshold
    fail = np.flatnonzero(~above)
    selected = int(fail[0]) if fail.size else len(observed)
    return selected, above


def _result(kind, observed, null, quantile, n_permutations, m):
    threshold = np.quantile(null, quantile, axis=0)
    selected, above = _sequential_select(observed, threshold)
    return {"kind": kind, "singular_values": observed, "threshold": threshold,
            "null_singular_values": null, "above_threshold": above, "n_components": selected,
            "n_above": int(above.sum()), "quantile": float(quantile),
            "n_permutations": int(n_permutations), "at_bound": selected == m}


def _cross_cov_spectrum(Z, G, m):
    """Top-m singular values of K^T Z, computed as sqrt(eig(Z^T G Z)) with G = K K^T."""
    vals = np.clip(linalg.eigvalsh(Z.T @ (G @ Z)), 0.0, None)[::-1]
    return np.sqrt(vals[:m])


def pa_entrywise(X_std, K, n_permutations=1000, quantile=0.95, max_components=60, seed=1234):
    """Cross-covariance PA with the global entry-wise null (all entries of Y = K^T X shuffled)."""
    X_std = np.asarray(X_std, float)
    n, p = X_std.shape
    m = min(max_components, p, n)
    observed = _cross_cov_spectrum(X_std, K @ K.T, m)
    Y = K.T @ X_std                                          # (mn, p)

    def top_svals(M):
        a, b = M.shape
        vals = np.clip(linalg.eigvalsh(M @ M.T if a <= b else M.T @ M), 0.0, None)[::-1]
        out = np.zeros(m)
        k = min(m, len(vals))
        out[:k] = np.sqrt(vals[:k])
        return out

    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, m))
    for b in range(n_permutations):
        null[b] = top_svals(rng.permuted(Y, axis=None))
    return _result("global_entrywise", observed, null, quantile, n_permutations, m)


def pa_row_permute(X_std, K, n_permutations=1000, quantile=0.95, max_components=60, seed=1234):
    """Cross-covariance PA with the row-permute null (cells of X permuted, one permutation per replicate)."""
    X_std = np.asarray(X_std, float)
    n, p = X_std.shape
    m = min(max_components, p, n)
    G = K @ K.T                                              # (n, n), computed once
    observed = _cross_cov_spectrum(X_std, G, m)
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, m))
    for b in range(n_permutations):
        null[b] = _cross_cov_spectrum(X_std[rng.permutation(n)], G, m)
    return _result("cross_covariance", observed, null, quantile, n_permutations, m)


def pa_expression(X_std, n_permutations=1000, quantile=0.95, max_components=60, seed=1234):
    """Horn PA on X alone: every gene permuted independently (expression self-rank)."""
    X = np.asarray(X_std, float)
    X = X - X.mean(axis=0, keepdims=True)
    n, p = X.shape
    m = min(max_components, p, n - 1)
    observed = linalg.svdvals(X, check_finite=False)[:m]
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, m))
    for b in range(n_permutations):
        null[b] = linalg.svdvals(rng.permuted(X, axis=0), check_finite=False)[:m]
    return _result("expression", observed, null, quantile, n_permutations, m)


# Names used by the notebooks before this module existed.
global_entrywise_pa = pa_entrywise
cross_covariance_pa = pa_row_permute
expression_pa = pa_expression


@dataclass
class PAResult:
    """Output of run_parallel_analysis.  ``J`` is the number of components to fit."""
    J: int                                   # selected by `selector`, clipped to [1, j_cap]
    j_cap: Optional[int]
    null: str
    result: dict 
    summary: pd.DataFrame = field(repr=False)
    settings: dict = field(repr=False)

    def __repr__(self):
        J_raw = self.result["n_components"]
        cap = f", capped from {J_raw}" if self.J != J_raw else ""
        return f"PAResult(J={self.J} by {self.null} null{cap})"


def run_parallel_analysis(X_std, K, null: str = "entrywise",
                          n_permutations: int = 1000, quantile: float = 0.95, max_components: int = 60,
                          seed: int = 1234, j_cap: Optional[int] = 20, verbose: bool = True) -> PAResult:
    """Run PA with one null distribution and return the selected J.

    Parameters
    ----------
    X_std : (n, p) gene-standardized expression.
    K : (n, m*n) concatenated, normalized kernels (the CCA input).
    null : which null to use; one of "entrywise" (default), "row_permute", "expression".
    j_cap : upper bound on ``J`` (runtime guard); ``None`` for no cap.  ``J`` is at least 1.
    """
    X_std = np.asarray(X_std, float)
    K = np.asarray(K, float)
    if K.shape[0] != X_std.shape[0]:
        raise ValueError(f"K has {K.shape[0]} rows but X has {X_std.shape[0]} cells")
    kw = dict(n_permutations=n_permutations, quantile=quantile, max_components=max_components, seed=seed)

    if null not in NULLS:
        raise ValueError(f"unknown null '{null}'; choose from {NULLS}")
    funcs = {"entrywise": lambda: pa_entrywise(X_std, K, **kw),
             "row_permute": lambda: pa_row_permute(X_std, K, **kw),
             "expression": lambda: pa_expression(X_std, **kw)}
    result = funcs[null]()

    J_raw = int(result["n_components"])
    J = max(J_raw, 1) if j_cap is None else int(min(max(J_raw, 1), j_cap))
    summary = pd.DataFrame([{
        "null": null,
        "PA variant": DESCRIPTIONS[null],
        "selected_J": result["n_components"],
        "n_above_threshold": result["n_above"],
        "at_scan_bound": result["at_bound"]
    }])
    settings = dict(null=null, j_cap=j_cap, **kw)
    if verbose:
        print(f"PA ({null} null) selects J = {J_raw}")
        if J != J_raw:
            print(f"NOTE: J capped/raised to {J} (j_cap={j_cap}); raise j_cap to fit more.")
    return PAResult(J=J, j_cap=j_cap, null=null,
                    result=result, summary=summary, settings=settings)


def plot_parallel_analysis(pa: PAResult, path=None):
    """One panel per null: observed spectrum, permutation threshold, null median and the selected J.

    The selector's panel comes first.  Returns the figure (and saves it if `path` is given).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.3, 4.6), constrained_layout=True)
    r = pa.result
    v, t = np.asarray(r["singular_values"]), np.asarray(r["threshold"])
    x = np.arange(1, len(v) + 1)
    sel = r["n_components"]
    ax.plot(x, v, "o-", color="#234F78", lw=1.8, ms=3.5, label="Observed")        
    ax.plot(x, t, "s--", color="#B55648", lw=1.4, ms=2.5, label=f"Permutation {r['quantile']:.0%} threshold")
    ax.plot(x, np.median(np.asarray(r["null_singular_values"]), axis=0), color="#88939E", lw=1.1,
            label="Null median")
    ax.axvline(sel + 0.5, color="#288268", ls="--", lw=1.6, label=f"PA J = {sel}")
    title = PLOT_TITLES[pa.null]
    ax.set(xlabel="Singular-value index", ylabel="Singular value", title=title)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
