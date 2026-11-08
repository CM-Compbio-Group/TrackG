"""evaluation_kptracer.py
Figures and evaluation of a Track-G fit on KP-Tracer (tumor family 3454_Lkb1_All).
The two cellular attributes are the lineage tree and the location (SubTumor label).

Negative controls (paper, Supplementary "Conditional permutation tests for KP-Tracer negative controls"): genes whose
expression has no structure along one attribute once the other attribute is fixed. Moran's I of every gene on a graph
of the attribute is compared with a conditional permutation null (B = 1000 permutations,
p = (1 + #{b: I_b > I}) / (B + 1)):
    lineage  : kNN graph of the lineage kernel (>= 30 neighbours per cell and all sister leaves); the expression is
               permuted among cells of the same location. Negatives: p > 0.5.
    location : graph connecting cells of the same location; the expression is permuted among sister leaves (cells
               with the same parent), which keeps the lineage exactly. Negatives: p > 0.5 among the testable genes
               (>= 25 % of the gene's variance lies in cells whose location the null can change).
A gene ranking for an attribute should place few of its negative controls at the top (``plot_fp_curves``).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import sparse
from scipy.stats import rankdata

ATTRIBUTE = {"Sigma_phylo": "Lineage", "Sigma_subtumor": "Location"}   # kernel name -> attribute in figures
TRACKG_COLOR = "#2b6ca3"
plt.rcParams["pdf.fonttype"] = 42                                        # editable text in the PDF figures


# ----------------------------------------------------------------------------- tree helpers
def sister_groups(tree, cell_names):
    """Cells grouped by their immediate parent in the tree (sister leaves have identical lineage-kernel rows)."""
    parent = {child: clade for clade in tree.find_clades() for child in clade.clades}
    leaf = {str(t.name): t for t in tree.get_terminals()}
    pid = pd.Series([id(parent[leaf[c]]) if (c in leaf and leaf[c] in parent) else f"solo_{c}" for c in cell_names])
    return [np.flatnonzero((pid == g).to_numpy()) for g in pd.unique(pid)]


def tree_layout(tree, cell_names):
    """Row order of the cells along the tree leaves + x (depth) / y positions of every node for drawing the tree."""
    keep = set(map(str, cell_names))
    leaves = [t for t in tree.get_terminals() if str(t.name) in keep]
    y = {t: i + 1 for i, t in enumerate(leaves)}
    x = tree.depths() if max(tree.depths().values()) else tree.depths(unit_branch_lengths=True)

    def set_y(clade):
        if clade.is_terminal():
            return y.get(clade)
        ys = [v for v in (set_y(c) for c in clade.clades) if v is not None]
        if ys:
            y[clade] = sum(ys) / len(ys)
            return y[clade]
        return None

    set_y(tree.root)
    pos = {c: i for i, c in enumerate(cell_names)}
    return np.array([pos[str(t.name)] for t in leaves]), x, y


def draw_tree(ax, tree, x, y, lw=0.3, color="#444444"):
    """Draw the tree with the root on the left and the leaves at y = 1..n."""
    def rec(clade, parent_x):
        if clade not in y:
            return
        ax.hlines(y[clade], parent_x, x[clade], color=color, lw=lw)
        kids = [c for c in clade.clades if c in y]
        if kids:
            ax.vlines(x[clade], min(y[c] for c in kids), max(y[c] for c in kids), color=color, lw=lw)
        for c in kids:
            rec(c, x[clade])
    rec(tree.root, 0)
    ax.set_axis_off()


# ----------------------------------------------------------------------------- negative controls
def moran_i(X, A):
    """Moran's I of every column of X (cells x genes) on the weighted cell graph A."""
    Xc = X - X.mean(0, keepdims=True)
    den = np.sum(Xc * Xc, axis=0)
    num = np.sum(Xc * (A @ Xc), axis=0)
    return (X.shape[0] / A.sum()) * np.divide(num, den, out=np.full(X.shape[1], np.nan), where=den > 0)


def permutation_pvalues(X, A, groups, n_perm=1000, seed=4201):
    """Observed Moran's I and upper-tail p-values; each permutation shuffles the cells within every group."""
    obs = moran_i(X, A)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(X.shape[1])
    for _ in range(n_perm):
        order = np.arange(X.shape[0])
        for ix in groups:
            order[ix] = rng.permutation(ix)
        exceed += moran_i(X[order], A) > obs
    return obs, (1 + exceed) / (n_perm + 1)


def knn_graph(K, k):
    """Symmetric kNN graph of a kernel with weights max(K, 0); k = neighbours per cell (int or one per cell).
    Ties are broken deterministically (stable sort, lower cell index first)."""
    K = np.array(K, dtype=float)
    np.fill_diagonal(K, -np.inf)
    n = K.shape[0]
    k = np.broadcast_to(np.asarray(k, dtype=int), (n,))
    order = np.argsort(-K, axis=1, kind="stable")
    r = np.repeat(np.arange(n), k)
    c = np.concatenate([order[i, :k[i]] for i in range(n)])
    A = sparse.csr_matrix((np.maximum(K[r, c], 0.0), (r, c)), shape=K.shape)
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def lineage_negative_controls(X2_np, K_lineage, tree, cell_names, location, genes, n_perm=1000, k_min=30,
                              seed=4201, p_neg=0.5):
    """Lineage structure beyond location: Moran's I on the lineage kNN graph, null = permute within each location."""
    n_sisters = np.zeros(len(cell_names), int)
    for ix in sister_groups(tree, cell_names):
        n_sisters[ix] = len(ix) - 1
    A = knn_graph(K_lineage, np.maximum(k_min, n_sisters))          # all sister leaves are neighbours
    labels = np.asarray(location, dtype=str)
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    I, p = permutation_pvalues(X2_np, A, groups, n_perm, seed)
    return pd.DataFrame({"moran_I": I, "p": p, "negative": p > p_neg}, index=pd.Index(genes, name="gene"))


def location_negative_controls(X2_np, tree, cell_names, location, genes, n_perm=1000, seed=4201, p_neg=0.5,
                               min_share=0.25):
    """Location structure beyond lineage: Moran's I on the same-location graph, null = permute among sister leaves."""
    labels = np.asarray(location, dtype=str)
    A = sparse.csr_matrix((labels[:, None] == labels[None, :]).astype(float))
    A.setdiag(0)
    A.eliminate_zeros()
    sisters = sister_groups(tree, cell_names)
    I, p = permutation_pvalues(X2_np, A, [ix for ix in sisters if len(ix) > 1], n_perm, seed)
    movable = np.zeros(len(labels), bool)                            # cells whose location the null can change
    for ix in sisters:
        movable[ix] = len(set(labels[ix])) > 1
    Xc = X2_np - X2_np.mean(0)
    testable = (Xc[movable] ** 2).sum(0) / (Xc ** 2).sum(0) >= min_share
    return pd.DataFrame({"moran_I": I, "p": p, "testable": testable, "negative": (p > p_neg) & testable},
                        index=pd.Index(genes, name="gene"))


# ----------------------------------------------------------------------------- FP curves
def rank_order(score):
    """Gene order of a ranking: average ranks for ties, then a stable sort."""
    return np.argsort(rankdata(-np.asarray(score, float), method="average"), kind="stable")


def negatives_in_top(scores, negatives, Ks=(20, 50, 100, 200)):
    """Number of negative controls among the top-K genes of each attribute's ranking (fewer is better)."""
    rows = {}
    for attr, neg in negatives.items():
        neg = neg.reindex(scores.index).to_numpy(bool)
        order = rank_order(scores[attr])
        rows[attr] = {f"top {K}": int(neg[order[:K]].sum()) for K in Ks} | {"negatives": int(neg.sum())}
    return pd.DataFrame(rows).T


def plot_fp_curves(scores, negatives, top=200, path=None):
    """Cumulative negative controls along the Track-G ranking of each attribute, with the random expectation."""
    fig, axes = plt.subplots(1, len(negatives), figsize=(6.0 * len(negatives), 4.4), constrained_layout=True)
    for ax, (attr, neg) in zip(np.atleast_1d(axes), negatives.items()):
        neg = neg.reindex(scores.index).to_numpy(bool)
        score = scores[attr].to_numpy()
        cum = np.cumsum(neg[rank_order(score)])
        n = int((score > 0).sum())                                   # the curve stops at the last gene with score > 0
        ax.plot(np.arange(1, n + 1), cum[:n], color=TRACKG_COLOR, lw=2.8, label="Track-G")
        if n < top:
            ax.plot(n, cum[n - 1], "o", color=TRACKG_COLOR, ms=5)
        Kx = np.arange(1, top + 1)
        ax.plot(Kx, Kx * neg.mean(), color="#888", ls=":", lw=1.2, label="random ranking")
        ax.set(xlim=(0, top), ylim=(0, max(cum[top - 1], top * neg.mean()) * 1.1 + 0.5),
               xlabel="Gene rank", ylabel="Num. false positives", title=attr)
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- component figures
def dominant_order(p):
    """Components grouped by their dominating attribute (in kernel order), then by index."""
    return np.argsort(p.to_numpy().argmax(0), kind="stable")


def plot_relatedness(p, path=None):
    """Heatmap of the relative relatedness p_kj of every component to every attribute (columns sum to 1)."""
    order = dominant_order(p)
    p_show = p.iloc[:, order].rename(index=ATTRIBUTE)
    fig, ax = plt.subplots(figsize=(0.65 * p.shape[1] + 1.5, 2.6))
    sns.heatmap(p_show, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "relative relatedness $p_{kj}$"})
    for b in np.flatnonzero(np.diff(p.to_numpy().argmax(0)[order])) + 1:
        ax.axvline(b, color="white", ls="--", lw=1.3)
    ax.set(title="Relative relatedness of each component to the two cellular attributes",
           xlabel="Components (ordered by dominating attribute)", ylabel="Cellular attribute")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def location_colors(location):
    """One colour per location (tab20), in sorted order."""
    names = sorted(pd.unique(np.asarray(location, dtype=str)))
    return {g: plt.cm.tab20(i % 20) for i, g in enumerate(names)}


def _minmax(M):
    """Scale every column to [0, 1] (as in the paper's KA heatmaps)."""
    M = np.asarray(M, float)
    rng = M.max(0) - M.min(0)
    return np.divide(M - M.min(0), rng, out=np.zeros_like(M), where=rng > 0)


def _short(name):
    return str(name).replace("3454_Lkb1_", "")


def plot_scores_by_location(scores, location, title="KA", path=None):
    """Cell scores (cells x components, e.g. KA of the location-related components) with cells ordered by location."""
    labels = np.asarray(location, dtype=str)
    order = np.argsort(labels, kind="stable")
    colors = location_colors(labels)
    fig = plt.figure(figsize=(0.55 * scores.shape[1] + 3.0, 7.5))
    gs = GridSpec(1, 3, width_ratios=[0.25, 0.55 * scores.shape[1], 0.15], wspace=0.04)
    ax_c, ax_h, ax_b = (fig.add_subplot(gs[i]) for i in range(3))
    ax_c.imshow(np.array([colors[g] for g in labels[order]])[:, None], aspect="auto", interpolation="none")
    ax_c.set_axis_off()
    im = ax_h.imshow(_minmax(scores.to_numpy())[order], aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                     interpolation="none")
    ax_h.set_xticks(range(scores.shape[1]))
    ax_h.set_xticklabels(scores.columns, rotation=90)
    ax_h.set_yticks([])
    ax_h.set_title(title)
    fig.colorbar(im, cax=ax_b, label="scaled score")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], [_short(g) for g in colors],
               title="Location", loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_scores_by_tree(scores, tree, cell_names, location, title="KA", path=None):
    """Cell scores (cells x components, e.g. KA of the lineage-related components) with cells ordered by the tree."""
    order, x, y = tree_layout(tree, cell_names)
    labels = np.asarray(location, dtype=str)
    colors = location_colors(labels)
    n = len(order)
    fig = plt.figure(figsize=(0.55 * scores.shape[1] + 4.5, 7.5))
    gs = GridSpec(1, 4, width_ratios=[2.0, 0.25, 0.55 * scores.shape[1], 0.15], wspace=0.04)
    ax_t, ax_c, ax_h, ax_b = (fig.add_subplot(gs[i]) for i in range(4))
    draw_tree(ax_t, tree, x, y)
    ax_t.set_ylim(n + 0.5, 0.5)                                  # first leaf on top, as in the heat map
    extent = [-0.5, scores.shape[1] - 0.5, n + 0.5, 0.5]
    ax_c.imshow(np.array([colors[g] for g in labels[order]])[:, None], aspect="auto", interpolation="none",
                extent=[0, 1, n + 0.5, 0.5])
    ax_c.set_axis_off()
    im = ax_h.imshow(_minmax(scores.to_numpy())[order], aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                     interpolation="none", extent=extent)
    ax_h.set_xticks(range(scores.shape[1]))
    ax_h.set_xticklabels(scores.columns, rotation=90)
    ax_h.set_yticks([])
    ax_h.set_title(title)
    fig.colorbar(im, cax=ax_b, label="scaled score")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], [_short(g) for g in colors],
               title="Location", loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_gene_expression(log_exp, genes, tree, location, title="", path=None):
    """Expression (z-score per gene, clipped at +/-3) of a gene set: cells ordered by location (left) and by tree (right)."""
    cell_names = log_exp.index.to_numpy(dtype=str)
    Z = log_exp[list(genes)].to_numpy(float)
    Z = np.clip((Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0), -3, 3)
    labels = np.asarray(location, dtype=str)
    colors = location_colors(labels)
    loc_order = np.argsort(labels, kind="stable")
    tree_order, x, y = tree_layout(tree, cell_names)
    n, g = len(cell_names), len(genes)
    fig = plt.figure(figsize=(2 * 0.3 * g + 5.5, 7.5))
    gs = GridSpec(1, 6, width_ratios=[0.25, 0.3 * g, 1.6, 0.25, 0.3 * g, 0.15], wspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    kw = dict(aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="none", extent=[-0.5, g - 0.5, n + 0.5, 0.5])
    for ax_c, ax_h, order, ttl in ((axes[0], axes[1], loc_order, "cells ordered by location"),
                                   (axes[3], axes[4], tree_order, "cells ordered by lineage tree")):
        ax_c.imshow(np.array([colors[c] for c in labels[order]])[:, None], aspect="auto", interpolation="none",
                    extent=[0, 1, n + 0.5, 0.5])
        ax_c.set_axis_off()
        im = ax_h.imshow(Z[order], **kw)
        ax_h.set_xticks(range(g))
        ax_h.set_xticklabels(genes, rotation=90, fontsize=8)
        ax_h.set_yticks([])
        ax_h.set_title(ttl, fontsize=10)
    draw_tree(axes[2], tree, x, y)
    axes[2].set_ylim(n + 0.5, 0.5)
    fig.colorbar(im, cax=axes[5], label="expression z-score")
    fig.suptitle(title)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], [_short(c) for c in colors],
               title="Location", loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig
