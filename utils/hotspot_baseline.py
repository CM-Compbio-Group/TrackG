"""Hotspot (local autocorrelation of each gene on a cell KNN graph): simulations, KP-Tracer, C. elegans.

``run_hotspot`` filters the genes with ``Track_G.gene_filtering.filter_genes`` (the same genes as
the CCA), builds the cell KNN graph from a tree, a kernel, a distance matrix or a latent embedding, and returns
the per-gene autocorrelation table (and optionally the gene modules).  Ranks and FP curves are done by the caller.

>>> from hotspot_baseline import run_hotspot
>>> out = run_hotspot(log_exp, tree=TREE_NEWICK)                  # KP-Tracer: lineage tree (newick path)
>>> out = run_hotspot(log_exp, kernel=kernels["phylo"])           # simulation: cells x cells kernel
>>> out["autocorrelation"]    # genes x [C, Z, Pval, FDR]

Defaults follow the collaborator's Hotspot_KPtracer_Phylo.py (2026-10-21): model "normal", unweighted KNN graph
with 30 neighbours, 4 jobs, FDR < 0.05, modules with min_gene_threshold = 30, core_only = True.
One addition: ``random_state``.  Hotspot's tree KNN counts edges (branch lengths are ignored) and breaks the many
distance ties with the unseeded global ``np.random``; on KP-Tracer two unseeded runs give Z with Spearman ~0.90.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def kernel_to_distance(kernel) -> np.ndarray:
    """Kernel-induced distance d_ij = sqrt(K_ii + K_jj - 2 K_ij) (Euclidean distance in the feature space)."""
    K = np.asarray(kernel, dtype=float)
    d = np.diag(K)
    return np.sqrt(np.clip(d[:, None] + d[None, :] - 2.0 * K, 0.0, None))


def run_hotspot(log_exp, *, tree=None, kernel=None, distances=None, latent=None, model: str = "normal",
                n_neighbors: int = 30, weighted_graph: bool = False, jobs: int = 4, fdr_threshold: float = 0.05,
                modules: bool = False, min_gene_threshold: int = 30, core_only: bool = True,
                n_modules: Optional[int] = None, random_state: int = 0,
                min_expr_frac: float = 0.01, min_gene_std: float = 1e-6) -> Dict[str, Any]:
    """Run Hotspot and return the per-gene autocorrelation statistics.

    log_exp : cells x genes log-expression, a DataFrame indexed by cell name.
    Exactly one cell-similarity input (rows / leaves in the cell names or the row order of ``log_exp``):
      tree      : ete3 tree or newick path (read with format=1); pruned to the cells of ``log_exp`` with branch
                  lengths kept, as in the collaborator's script (a passed tree is copied, not modified).
                  Hotspot does not support ``weighted_graph=True`` with a tree.
      kernel    : cells x cells similarity, used through ``kernel_to_distance``.
      distances : cells x cells distance matrix.
      latent    : cells x d embedding (Euclidean KNN).
    model : Hotspot null model: "normal" (depth-adjusted normal), "none" (pre-standardized data), "danb", "bernoulli".
    modules : also compute local correlations of the FDR-significant genes, Hotspot's gene modules and their
        per-cell module scores.
    n_modules : with ``modules``, a fixed number of modules instead of Hotspot's threshold rule: Hotspot's own
        average-linkage tree of the local correlations (``hs.linkage``) is cut into ``n_modules`` clusters
        (``scipy.cluster.hierarchy.fcluster``, criterion "maxclust"); every significant gene is then in a module.
        None (default) keeps Hotspot's modules (``min_gene_threshold``, ``core_only``).
    random_state : seed of the global ``np.random`` before the KNN graph (tree-distance ties are broken randomly).
    min_expr_frac, min_gene_std : gene filter (``Track_G.gene_filtering.filter_genes``); 0 skips a criterion.

    Returns a dict
      autocorrelation   genes x [C, Z, Pval, FDR] in the filtered-gene order (Z = Hotspot's ranking statistic)
      significant       Index of the genes with FDR < ``fdr_threshold``
      local_correlations, modules, module_genes, module_scores   genes x genes Z, module per gene (-1 =
                        unassigned), {module: genes}, cells x modules scores; only if ``modules=True``
      genes, cells      Index of the filtered genes / the cells;  settings  dict of the run settings
    """
    import hotspot

    if sum(x is not None for x in (tree, kernel, distances, latent)) != 1:
        raise ValueError("Pass exactly one of tree, kernel, distances, latent.")
    filtered = filter_expression(log_exp, min_expr_frac, min_gene_std)
    genes, cells = filtered.columns, filtered.index
    adata = ad.AnnData(X=filtered.to_numpy(dtype=float), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))

    def _cells_by_cells(M, name):
        if isinstance(M, pd.DataFrame):
            M = M.set_axis(M.index.astype(str), axis=0).set_axis(M.columns.astype(str), axis=1).loc[cells, cells]
        M = np.asarray(M, dtype=float)
        if M.shape != (len(cells), len(cells)):
            raise ValueError(f"{name} must be cells x cells ({len(cells)} x {len(cells)}), got {M.shape}.")
        return M

    hs_kwargs = {}
    if tree is not None:
        import ete3
        tree = ete3.Tree(str(tree), format=1) if isinstance(tree, (str, Path)) else copy.deepcopy(tree)
        leaves = {leaf.name for leaf in tree.iter_leaves()}
        missing = [c for c in cells if c not in leaves]
        if missing:
            raise ValueError(f"{len(missing)} cells are not leaves of the tree, e.g. {missing[:3]}.")
        tree.prune(cells.tolist(), preserve_branch_length=True)
        hs_kwargs["tree"] = tree
    elif latent is not None:
        lat = latent.set_axis(latent.index.astype(str)).loc[cells] if isinstance(latent, pd.DataFrame) else latent
        adata.obsm["latent"] = np.asarray(lat, dtype=float)
        hs_kwargs["latent_obsm_key"] = "latent"
    else:
        D = _cells_by_cells(distances, "distances") if kernel is None \
            else kernel_to_distance(_cells_by_cells(kernel, "kernel"))
        adata.obsp["distances"] = D
        hs_kwargs["distances_obsp_key"] = "distances"

    hs = hotspot.Hotspot(adata=adata, model=model, **hs_kwargs)
    np.random.seed(random_state)
    hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=int(n_neighbors))
    autocorrelation = hs.compute_autocorrelations(jobs=int(jobs)).reindex(genes)
    autocorrelation.index.name = "gene"
    significant = autocorrelation.index[autocorrelation["FDR"] < fdr_threshold]

    out = dict(autocorrelation=autocorrelation, significant=significant)
    if modules and len(significant) < 2:                 # nothing to correlate: no modules
        out.update(local_correlations=None, modules=pd.Series(-1, index=genes, name="Module"), module_genes={},
                   module_scores=pd.DataFrame(index=cells))
    elif modules:
        out["local_correlations"] = hs.compute_local_correlations(significant, jobs=int(jobs))
        out["modules"] = hs.create_modules(min_gene_threshold=int(min_gene_threshold), core_only=core_only,
                                           fdr_threshold=fdr_threshold)
        if n_modules is not None:                    # same tree, cut into a fixed number of modules
            from scipy.cluster.hierarchy import fcluster
            cut = fcluster(hs.linkage, t=int(n_modules), criterion="maxclust")
            hs.modules = pd.Series(cut, index=out["modules"].index, name="Module")
            out["modules"] = hs.modules
        assigned = out["modules"][out["modules"] != -1]
        out["module_genes"] = {f"Module_{m}": g.index.tolist() for m, g in assigned.groupby(assigned)}
        scores = hs.calculate_module_scores() if len(assigned) else pd.DataFrame(index=hs.adata.obs_names)
        out["module_scores"] = scores.rename(columns=lambda m: f"Module_{m}").reindex(cells)
    graph = "tree" if tree is not None else "latent" if latent is not None else "kernel" if kernel is not None \
        else "distances"
    out.update(genes=genes, cells=cells, settings=dict(
        method="Hotspot", n_cells=len(cells), n_genes=len(genes), min_expr_frac=float(min_expr_frac),
        min_gene_std=float(min_gene_std), graph=graph, model=model, n_neighbors=int(n_neighbors),
        weighted_graph=bool(weighted_graph), random_state=int(random_state), jobs=int(jobs),
        fdr_threshold=float(fdr_threshold),
        n_significant=len(significant), modules=bool(modules),
        min_gene_threshold=int(min_gene_threshold) if modules else None, core_only=bool(core_only) if modules else None,
        n_modules=int(n_modules) if modules and n_modules is not None else None,
        gene_filter_file=gene_filtering.__file__))
    return out
