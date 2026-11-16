"""evaluation_simulation.py
Comparison of a Track-G fit with the true modules of a simulated data set, and the figures of the simulation
(paper: Results, simulations; Supplementary: details of simulation). The simulation and the Track-G steps are
written out in ``tutorial/simulation.ipynb``.

Every true module is matched to one component by the Hungarian algorithm on |Pearson(h_j, XW component)|, then
scored by
    component activity recovery : |Pearson| of the matched pair
    module gene set recovery    : Jaccard between the 40 specific genes and the top 40 genes of the component
                                  (shared genes removed from the ranking)
    attribute assignment        : p_true - p_other, the relative relatedness of the matched component to the module's
                                  attribute minus that to the other attribute (> 0 = correct)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LABEL = {"phylo": "lineage", "cell_type": "cell type"}    # simulator topic -> attribute (kernel) name in the notebook
ACOL = {"lineage": "#17a2b8", "cell type": "#e8b923"}     # attribute colours (paper Fig. Simulation C)
TRACKG_COLOR = "#2b6ca3"
plt.rcParams["pdf.fonttype"] = 42                         # editable text in the PDF figures


def setting_label(setting):
    """'struct_low__iid_high' -> 'unseen attr. contribution low, pure noise high'."""
    return "unseen attr. contribution {}, pure noise {}".format(*(p.split("_")[-1] for p in setting.split("__")))


# ----------------------------------------------------------------------------- comparison with the truth
def hungarian(H, cells):
    """|corr| of every true module (columns of H) with every component, and the one-to-one matching {module: comp}."""
    P, Q = H - H.mean(0), cells - cells.mean(0)
    R = np.abs((P / np.maximum(np.linalg.norm(P, axis=0), 1e-12)).T
               @ (Q / np.maximum(np.linalg.norm(Q, axis=0), 1e-12)))
    rows, cols = linear_sum_assignment(-R)
    return R, dict(zip(rows, cols))


def module_names(topics):
    """lineage 1, lineage 2, cell type 1, cell type 2."""
    topics = np.asarray(topics).astype(str)
    return [f"{LABEL[t]} {int(np.sum(topics[:i + 1] == t))}" for i, t in enumerate(topics)]


def match_to_truth(sim, fit, out, gene_ids):
    """One row per true module (recovery metrics of its matched component) + the |corr| matrix for the heat maps.

    sim: simulate_paired output; fit: fit_witten_nonnegative_cca output; out: summarize_cca(fit, data), with the
    kernels named as in LABEL ("lineage", "cell type"); gene_ids: gene names of the simulated genes (rows of sim["B"]).
    """
    H, topics = sim["H"], np.asarray(sim["topics"]).astype(str)
    ext = np.flatnonzero(fit["extracted_record"])                                       # extracted components
    comps = list(out["observed_component_expression"].columns[ext])
    XW, p = out["observed_component_expression"][comps], out["kernel_attr"][comps]     # cell scores, p_kj
    W2 = pd.DataFrame(fit["W2_matrix"][:, ext], index=out["gene_kernel"].index.astype(str), columns=comps)
    r, nz, kept, gene_ids = H.shape[1], sim["B"] > 0, set(W2.index), np.asarray(gene_ids)
    specific = [set(gene_ids[nz[:, i] & (nz.sum(1) == 1)]) & kept for i in range(r)]   # genes of module i only
    shared = set(gene_ids[nz.sum(1) == r]) & kept                                       # genes of every module
    R, match = hungarian(H, XW.to_numpy(float)) if comps else (np.zeros((r, 0)), {})
    names, rows = module_names(topics), []
    for i in range(r):
        attr = LABEL[topics[i]]
        row = dict(module=names[i], attribute=attr, component=None, abs_r_cell=0.0, jaccard=0.0, p_diff=np.nan)
        if i in match:
            c = comps[match[i]]
            ranked = W2[c].sort_values(ascending=False, kind="stable").index
            top = [g for g in ranked if g not in shared][:len(specific[i])]
            overlap = len(set(top) & specific[i])
            row.update(component=c, abs_r_cell=R[i, match[i]], jaccard=overlap / len(set(top) | specific[i]),
                       p_diff=p.loc[attr, c] - p[c].drop(attr).max())
        rows.append(row)
    label = dict(p.idxmax().items())                                         # dominating attribute of each component
    return rows, dict(R=R, match=match, comps=comps, label=label, names=names)


# ----------------------------------------------------------------------------- figures
def _settings(values):
    """Settings in the order of their first appearance (the order of the notebook's loop)."""
    return list(dict.fromkeys(values))


def _setting_ticks(ax, settings, size=14):
    """Tick labels "<unseen level>\\n<pure-noise level>" with the row key left of the axis (paper Fig. Simulation)."""
    ax.set_xticklabels(["{}\n{}".format(*(p.split("_")[-1] for p in k.split("__"))) for k in settings],
                       fontsize=size, linespacing=1.2)
    off = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
    ax.annotate("unseen attr.\npure noise", xy=(0, 0), xycoords="axes fraction", xytext=(-8, -off),
                textcoords="offset points", ha="right", va="top", fontsize=size, linespacing=1.2)


def _boxes(ax, values, x, color, width, rng):
    bp = ax.boxplot([values], positions=[x], widths=0.85 * width, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.3))
    bp["boxes"][0].set(facecolor=color, alpha=0.35, edgecolor=color)
    ax.scatter(x + rng.uniform(-0.3, 0.3, len(values)) * width, values, s=16, color=color, edgecolor="k", lw=0.3,
               zorder=3)


def plot_recovery(res, path=None):
    """(A) component activity recovery and (B) module gene set recovery: mean over the modules, one point per seed."""
    settings = _settings(res["setting"])
    per_seed = res.groupby(["setting", "seed"], observed=True)[["abs_r_cell", "jaccard"]].mean().reset_index()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, (col, ylab, title) in zip(axes, [("abs_r_cell", "Pearson correlation", "Component activity recovery"),
                                             ("jaccard", "Jaccard similarity", "Module gene set recovery")]):
        for g, setting in enumerate(settings):
            _boxes(ax, per_seed.loc[per_seed.setting == setting, col].to_numpy(), g, TRACKG_COLOR, 0.5, rng)
        ax.set_xticks(range(len(settings)))
        _setting_ticks(ax, settings)
        ax.set(ylim=(0, 1.02))
        ax.set_ylabel(ylab, fontsize=15)
        ax.set_title(f"{title} (Track-G)", fontsize=15)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_attribute_assignment(res, path=None, gap=0.7, box=0.45):
    """(C) p_true - p_other of the component matched to each true module; one point per module and seed."""
    settings = _settings(res["setting"])
    rng = np.random.default_rng(0)
    attrs = list(ACOL)
    width = box / len(attrs)
    fig, ax = plt.subplots(figsize=(6.5, 8.5), constrained_layout=True)
    for g, setting in enumerate(settings):
        for k, attr in enumerate(attrs):
            v = res.loc[(res.setting == setting) & (res.attribute == attr), "p_diff"].dropna().to_numpy()
            if len(v):
                _boxes(ax, v, g * gap + (k - (len(attrs) - 1) / 2) * width, ACOL[attr], width, rng)
    ax.set_xticks([g * gap for g in range(len(settings))])
    _setting_ticks(ax, settings)
    ax.set_xlim(-gap / 2 - box / 2, (len(settings) - 1) * gap + gap / 2 + box / 2)
    ax.set_ylim(-1.02, 1.02)
    ax.axhline(0, color="0.5", lw=1, ls="--")
    ax.set_ylabel("difference of relative relatedness: $p_{true} - p_{other}$", fontsize=15)
    ax.set_title("Attribute assignment", fontsize=15)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ACOL[a], alpha=0.5, edgecolor=ACOL[a], label=a)
                        for a in attrs], title="true cellular attribute", frameon=False, fontsize=13,
               loc="outside lower center", ncol=2)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def _aligned(st):
    """|corr| with column i = the component matched to module i (NaN if none), then the unmatched components;
    also the component index of every column (None for an unmatched module)."""
    R, match = st["R"], st["match"]
    rest = sorted(set(range(R.shape[1])) - set(match.values()), key=lambda j: -R[:, j].max())
    idx = [match.get(i) for i in range(len(R))] + rest
    return np.column_stack([R[:, j] if j is not None else np.full(len(R), np.nan) for j in idx]), idx


def _draw_corr(ax, M, names, xlabels, title):
    ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(len(names)):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="w", lw=1.6))   # matched component
        for j in np.flatnonzero(~np.isnan(M[i])):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11, color="k" if M[i, j] > .6 else "w")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(xlabels, fontsize=12, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_title(title, fontsize=12)


def plot_corr_heatmaps(store, seeds, path=None):
    """|corr| between the true module activities (rows) and the Track-G components (columns, in the order of the
    matched modules; matched pairs boxed). One seed: columns named by the dominating attribute; several seeds: mean.
    ``store`` is keyed by (seed, setting), as filled by the notebook's loop."""
    seeds = list(seeds)
    settings = _settings(s for _, s in store)
    fig, axes = plt.subplots(1, len(settings), figsize=(4.4 * len(settings), 5.0), constrained_layout=True)
    for ax, setting in zip(axes, settings):
        mats = [_aligned(store[s, setting])[0] for s in seeds]
        width = max(M.shape[1] for M in mats)
        M = np.nanmean([np.pad(m, ((0, 0), (0, width - m.shape[1])), constant_values=np.nan) for m in mats], axis=0)
        st = store[seeds[0], setting]
        if len(seeds) == 1:                               # dominating attribute of each component
            xlabels = [st["label"][st["comps"][j]] if j is not None else "" for j in _aligned(st)[1]]
        else:
            xlabels = [f"comp {j + 1}" for j in range(M.shape[1])]
        _draw_corr(ax, M, st["names"], xlabels, setting_label(setting).replace(", ", "\n"))
    fig.supylabel("true cellular attributes", fontsize=15)
    fig.supxlabel("Track-G components, in the order of the matched true cellular attributes", fontsize=15)
    fig.suptitle(f"seed {seeds[0]}" if len(seeds) == 1 else f"mean over {len(seeds)} seeds", fontsize=15)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig
