"""evaluation_simulation.py
Track-G on the simulated data of the paper (Results: simulations; Supplementary: details of simulation).

Design: 400 cells on a balanced binary tree (lineage kernel) with 8 random cell types (cell-type kernel); two modules per
attribute, each with 40 specific genes, 40 genes shared by all modules, 600 genes in total. The log expression is
M = H B^T + P1 P2^T (unseen attributes, rank 10) + E (pure noise); counts ~ Poisson(e^M); data = log(1 + counts).
The unseen-attribute term and the pure noise are each set to a low or a high level (signal-to-noise power ratio 1 or
0.25): four settings, 10 replicates (seeds); the modules of a seed are the same in all four settings.
Track-G gets the true number of modules. Every true module is matched to one component by the Hungarian algorithm on
|Pearson(h_j, XW component)|, then scored by
    component activity recovery : |Pearson| of the matched pair
    module gene set recovery    : Jaccard between the 40 specific genes and the top 40 genes of the component
                                  (shared genes removed from the ranking)
    attribute assignment        : p_true - p_other, the relative relatedness of the matched component to the module's
                                  attribute minus that to the other attribute (> 0 = correct)
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from Track_G import build_kernel_matrix, filter_and_standardize_expression, fit_witten_nonnegative_cca
from simulate_multikernel import SimulationConfig, make_demo_kernels, simulate_paired

# ---- settings of the paper (noise_levels_sim, two_kernel data set) ----
N_CELLS, KERNEL_SEED = 400, 2026
SEEDS = list(range(10))
KERNEL_NAMES = ["phylo", "cell_type"]                     # attributes with modules: lineage tree, cell type
SIM_CONFIG = SimulationConfig(n_genes=600, exclusive_genes=40, shared_genes=40, scale_B=True, rho=0.0,
                              components_per_topic=(2, 2, 0))   # modules per attribute (lineage, cell type, interaction)
LEVELS = {"low": 1.0, "high": 0.25}                       # noise level -> SNR = signal power / power of that noise part
SETTINGS = {f"struct_{s}__iid_{u}": dict(structured_snr=LEVELS[s], unstructured_snr=LEVELS[u])
            for s in LEVELS for u in LEVELS}              # unseen-attribute level x pure-noise level
SETTING_LABEL = {k: "unseen attr. contribution {}, pure noise {}".format(*(p.split("_")[-1] for p in k.split("__")))
                 for k in SETTINGS}
MIN_EXPR_FRAC = 0.01                                      # 1 % expression filter, as for the real data
CCA_KWARGS = dict(c_alpha=None, c_w2=None, n_restarts=10, random_seed=42, max_cca_iter=3000, cca_tolerance=1e-6,
                  require_all_components=False, verbose=False)

LABEL = {"phylo": "lineage", "cell_type": "cell type"}    # names in tables and figures
ACOL = {"lineage": "#17a2b8", "cell type": "#e8b923"}     # attribute colours (paper Fig. Simulation C)
TRACKG_COLOR = "#2b6ca3"
plt.rcParams["pdf.fonttype"] = 42                         # editable text in the PDF figures


# ----------------------------------------------------------------------------- data and fits
def make_kernels():
    """Lineage (balanced binary tree) and cell-type kernels of the 400 simulated cells (about a minute)."""
    with threadpool_limits(limits=1):
        return make_demo_kernels(N_CELLS, KERNEL_SEED)    # (kernels, cell_ids, metadata)


def simulate(kernels, seed, setting):
    """One data set: the simulation dict and the log expression log(1 + counts) (cells x genes)."""
    sim = simulate_paired(kernels, replace(SIM_CONFIG, seed=seed, **SETTINGS[setting]))
    return sim, np.log1p(sim["Y"]).astype(np.float32).astype(float)    # float32 as stored in the paper's runs


def fit_trackg(X, K, r, cell_ids, gene_ids):
    """Track-G with r components: cell scores XW, gene loadings W2 and A-block norms of the extracted components."""
    with contextlib.redirect_stdout(io.StringIO()):
        expr = filter_and_standardize_expression(pd.DataFrame(X, index=cell_ids, columns=gene_ids),
                                                 min_expr_frac=MIN_EXPR_FRAC)
    with threadpool_limits(limits=1):
        fit = fit_witten_nonnegative_cca(K=K, X2_np=expr["X2_np"], KERNEL_NAMES=KERNEL_NAMES, J=r, **CCA_KWARGS)
    ext = np.flatnonzero(fit["extracted_record"])
    comps = [f"C{j + 1}" for j in ext]
    return dict(XW=pd.DataFrame(fit["V_score_matrix"][:, ext], index=cell_ids, columns=comps),
                W2=pd.DataFrame(fit["W2_matrix"][:, ext], index=pd.Index(expr["valid_genes"]).astype(str), columns=comps),
                A_norm=pd.DataFrame(np.asarray(fit["all_scores"], float)[:, ext], index=KERNEL_NAMES, columns=comps))


def hungarian(H, cells):
    """|corr| of every true module (columns of H) with every component, and the one-to-one matching {module: comp}."""
    P, Q = H - H.mean(0), cells - cells.mean(0)
    R = np.abs((P / np.maximum(np.linalg.norm(P, axis=0), 1e-12)).T
               @ (Q / np.maximum(np.linalg.norm(Q, axis=0), 1e-12)))
    rows, cols = linear_sum_assignment(-R)
    return R, dict(zip(rows, cols))


def module_names(topics):
    """lineage 1, lineage 2, cell type 1, cell type 2."""
    return [f"{LABEL[t]} {int(np.sum(topics[:i + 1] == t))}" for i, t in enumerate(topics)]


def match_to_truth(fit, sim, gene_ids):
    """One row per true module (recovery metrics of its matched component) + the |corr| matrix for the heat maps."""
    H, B, topics = sim["H"], sim["B"], np.asarray(sim["topics"]).astype(str)
    r, nz, kept = H.shape[1], sim["B"] > 0, set(fit["W2"].index)
    specific = [set(gene_ids[nz[:, i] & (nz.sum(1) == 1)]) & kept for i in range(r)]   # genes of module i only
    shared = set(gene_ids[nz.sum(1) == r]) & kept                                       # genes of every module
    comps = list(fit["XW"].columns)
    R, match = hungarian(H, fit["XW"].to_numpy(float)) if comps else (np.zeros((r, 0)), {})
    p = fit["A_norm"] / fit["A_norm"].sum(0)              # relative relatedness p_kj of every component
    names, rows = module_names(topics), []
    for i in range(r):
        row = dict(module=names[i], attribute=LABEL[topics[i]], component=None, abs_r_cell=0.0, jaccard=0.0,
                   p_diff=np.nan)
        if i in match:
            c = comps[match[i]]
            ranked = fit["W2"][c].sort_values(ascending=False, kind="stable").index
            top = [g for g in ranked if g not in shared][:len(specific[i])]
            overlap = len(set(top) & specific[i])
            row.update(component=c, abs_r_cell=R[i, match[i]], jaccard=overlap / len(set(top) | specific[i]),
                       p_diff=p.loc[topics[i], c] - p[c].drop(topics[i]).max())
        rows.append(row)
    label = {c: LABEL[k] for c, k in fit["A_norm"].idxmax().items()}         # dominating attribute of each component
    return rows, dict(R=R, match=match, comps=comps, label=label, names=names)


def run_all(cache_dir, seeds=SEEDS):
    """Simulate every seed and setting, fit Track-G (cached in ``cache_dir``) and match it to the truth.
    Returns (results: one row per seed, setting and module; heat-map store keyed by (seed, setting))."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kernels, cell_ids, _ = make_kernels()
    K, _ = build_kernel_matrix(kernels, KERNEL_NAMES)     # K = [K_lineage^1/2 | K_cell type^1/2]
    gene_ids = np.array([f"gene_{g:04d}" for g in range(SIM_CONFIG.n_genes)])
    rows, store = [], {}
    for seed in seeds:
        for setting in SETTINGS:
            sim, X = simulate(kernels, seed, setting)
            path = cache_dir / f"seed{seed:02d}_{setting}_J{sim['true_rank']}.pkl"   # delete cache_dir to refit
            if path.exists():
                fit = pd.read_pickle(path)
            else:
                fit = fit_trackg(X, K, sim["true_rank"], cell_ids, gene_ids)
                pd.to_pickle(fit, path)
            mod_rows, store[seed, setting] = match_to_truth(fit, sim, gene_ids)
            rows += [dict(seed=seed, setting=setting, **row) for row in mod_rows]
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    res["setting"] = pd.Categorical(res["setting"], list(SETTINGS))
    return res, store


# ----------------------------------------------------------------------------- figures
def _setting_ticks(ax, size=14):
    """Tick labels "<unseen level>\\n<pure-noise level>" with the row key left of the axis (paper Fig. Simulation)."""
    ax.set_xticklabels(["{}\n{}".format(*(p.split("_")[-1] for p in k.split("__"))) for k in SETTINGS],
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
    per_seed = res.groupby(["setting", "seed"], observed=True)[["abs_r_cell", "jaccard"]].mean().reset_index()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, (col, ylab, title) in zip(axes, [("abs_r_cell", "Pearson correlation", "Component activity recovery"),
                                             ("jaccard", "Jaccard similarity", "Module gene set recovery")]):
        for g, setting in enumerate(SETTINGS):
            _boxes(ax, per_seed.loc[per_seed.setting == setting, col].to_numpy(), g, TRACKG_COLOR, 0.5, rng)
        ax.set_xticks(range(len(SETTINGS)))
        _setting_ticks(ax)
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
    rng = np.random.default_rng(0)
    attrs = list(ACOL)
    width = box / len(attrs)
    fig, ax = plt.subplots(figsize=(6.5, 8.5), constrained_layout=True)
    for g, setting in enumerate(SETTINGS):
        for k, attr in enumerate(attrs):
            v = res.loc[(res.setting == setting) & (res.attribute == attr), "p_diff"].dropna().to_numpy()
            if len(v):
                _boxes(ax, v, g * gap + (k - (len(attrs) - 1) / 2) * width, ACOL[attr], width, rng)
    ax.set_xticks([g * gap for g in range(len(SETTINGS))])
    _setting_ticks(ax)
    ax.set_xlim(-gap / 2 - box / 2, (len(SETTINGS) - 1) * gap + gap / 2 + box / 2)
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


def plot_corr_heatmaps(store, seeds=SEEDS, path=None):
    """|corr| between the true module activities (rows) and the Track-G components (columns, in the order of the
    matched modules; matched pairs boxed). One seed: columns named by the dominating attribute; several seeds: mean."""
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(4.4 * len(SETTINGS), 5.0), constrained_layout=True)
    for ax, setting in zip(axes, SETTINGS):
        mats = [_aligned(store[s, setting])[0] for s in seeds]
        width = max(M.shape[1] for M in mats)
        M = np.nanmean([np.pad(m, ((0, 0), (0, width - m.shape[1])), constant_values=np.nan) for m in mats], axis=0)
        st = store[seeds[0], setting]
        if len(seeds) == 1:                               # dominating attribute of each component
            xlabels = [st["label"][st["comps"][j]] if j is not None else "" for j in _aligned(st)[1]]
        else:
            xlabels = [f"comp {j + 1}" for j in range(M.shape[1])]
        _draw_corr(ax, M, st["names"], xlabels, SETTING_LABEL[setting].replace(", ", "\n"))
    fig.supylabel("true cellular attributes", fontsize=15)
    fig.supxlabel("Track-G components, in the order of the matched true cellular attributes", fontsize=15)
    fig.suptitle(f"seed {seeds[0]}" if len(seeds) == 1 else f"mean over {len(seeds)} seeds", fontsize=15)
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig
