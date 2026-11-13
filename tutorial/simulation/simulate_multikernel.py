"""Simulated expression for multi-kernel CCA.

One call = one data set; components_per_topic chooses the programs, e.g.
    (2, 2, 0)  two kernels: programs from the phylo and cell_type kernels only
    (2, 2, 2)  with interaction: programs from the phylo, cell_type and interaction kernels

With r = the number of programs (the advisor's design, simulate_compare.ipynb):
    H[:, j] ~ N(0, K_topic)                         cell scores of program j
    B: exclusive_genes genes per program ~ Gamma(2, 1); shared_genes genes load on every
       program ~ Gamma(2, 1) / r; all other genes 0; then B /= r (scale_B)
    X = H B^T + P1 P2^T + E                         log mean; P1 (cells x 10), P2 (genes x 10)
    Y = Poisson(exp(X)), Y_noise_free = Poisson(exp(H B^T)); the data is log1p(Y)

Rows are cells, columns are genes. The demo kernels are synthetic (tree + crossed cell types).
Requires NumPy and networkx (networkx builds the demo tree).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
import json
import networkx as nx

import numpy as np

TOPICS = ("phylo", "cell_type", "interaction")
STRUCTURED_NOISE_RANK = 10


@dataclass(frozen=True)
class SimulationConfig:
    """Generation settings.

    ``components_per_topic`` = programs per kernel (phylo, cell_type, interaction).
    Each program has ``exclusive_genes`` genes of its own (specific genes); ``shared_genes`` is
    the total number of genes that load on every program. All other genes carry noise only.
    ``scale_B`` divides B by the number of programs. Noise = rank-10 structured noise
    (entries of both factors ~ N(0, structured_noise_std^2)) + iid N(0, unstructured_noise_std^2).
    ``structured_snr`` / ``unstructured_snr``: if set, that noise part is instead scaled so that
    signal power / its power = the given SNR (column-centered power per entry).
    ``rho`` sets within-topic cross-covariance Cov(h_i,h_j)=rho*K.
    """
    n_genes: int = 600
    components_per_topic: tuple[int, int, int] = (2, 2, 2)
    exclusive_genes: int = 40
    shared_genes: int = 40
    scale_B: bool = True
    structured_noise_std: float = 0.3
    unstructured_noise_std: float = 0.1
    structured_snr: float | None = None
    unstructured_snr: float | None = None
    rho: float = 0.0
    seed: int = 0


def validate_kernels(kernels, tol=1e-8):
    """Return the kernels in TOPICS order (all kernels must share the same cell order).

    If interaction is absent, it is the Hadamard (elementwise) product of phylo and cell_type.
    """
    result = OrderedDict()
    for name in TOPICS:
        K = kernels[name] if name in kernels else result["phylo"] * result["cell_type"]
        K = np.asarray(K, dtype=float)
        # A kernel is the covariance of H: it must be symmetric and PSD.
        if not np.allclose(K, K.T, atol=tol, rtol=tol):
            raise ValueError(f"{name}: kernel is not symmetric.")
        K = (K + K.T) / 2
        values, vectors = np.linalg.eigh(K)
        if values[0] < -tol:
            raise ValueError(f"{name}: kernel is not positive semidefinite.")
        if values[0] < 0:  # tiny negative eigenvalues from round-off -> 0
            K = (vectors * np.maximum(values, 0)) @ vectors.T
        result[name] = K
    return result


def make_demo_kernels(n_cells=400, seed=2026):
    """Return (kernels, cell_ids, metadata) for a reproducible SYNTHETIC example.

    phylo = depth(LCA) / height on a balanced binary tree, cells placed on random leaves
    (diagonal 1). cell_type is an indicator Gram matrix for 5--8 crossed types.
    interaction is the elementwise product. All are PSD by construction.
    """
    rng = np.random.default_rng(seed)
    # phylo kernel
    height = (np.ceil(np.log2(n_cells)).astype(int))
    tree =  nx.balanced_tree(r=2, h=height, create_using=nx.DiGraph())
    leaves = [node for node, out_degree in tree.out_degree() if out_degree == 0]
    tree_leaf_index = rng.permutation(2**height)
    cell_leaf_mapping = {cell_id : leaves[tree_leaf_index[cell_id]] for cell_id in range(n_cells)}
    roots = [n for n, d in tree.in_degree() if d == 0]
    phylo = np.zeros((n_cells, n_cells), dtype=float)
    for i in range(n_cells):
        for j in range(n_cells):
            if i <= j:
                lca = nx.lowest_common_ancestor(tree, cell_leaf_mapping[i], cell_leaf_mapping[j])
                phylo[i, j] = nx.shortest_path_length(tree, source=roots[0], target=lca)  # root -> LCA = depth of the LCA
                phylo[j, i] = phylo[i, j]
    phylo /= height  # depth(LCA) / height: diagonal 1, like the real kernels

    # cell type kernel
    n_types = min(8, max(5, n_cells // 40))
    cell_type_index = rng.permutation(np.arange(n_cells) % n_types)
    cell_type = (cell_type_index[:, None] == cell_type_index[None, :]).astype(float)
    kernels = OrderedDict(phylo=phylo, cell_type=cell_type,
                          interaction=phylo * cell_type)
    ids = np.array([f"toy_cell_{i:04d}" for i in range(n_cells)], dtype=str)
    metadata = {
        "source": "synthetic_demo_not_C_elegans", "seed": int(seed),
        "n_cells": int(n_cells), "n_cell_types": int(n_types),
        "tree_leaf_index": tree_leaf_index[:n_cells].tolist(),   # leaf position of each cell
        "cell_type_labels": [f"toy_type_{v}" for v in cell_type_index],
        "description": "Shared-ancestry binary-tree kernel with crossed synthetic cell types.",
    }
    return kernels, ids, metadata

def load_kernel_npz(path):
    """Load (kernels, cell_ids, metadata) written by ``save_kernel_npz``."""
    with np.load(path, allow_pickle=False) as data:
        kernels = validate_kernels({t: data[t] for t in TOPICS if t in data.files})
        cell_ids = data["cell_ids"].astype(str)
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {}
    return kernels, cell_ids, metadata


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_kernel_npz(kernels, path, cell_ids=None, metadata=None):
    """Save the kernels, cell IDs and JSON metadata to an NPZ (no pickle); return its Path."""
    kernels = validate_kernels(kernels)
    n = kernels["phylo"].shape[0]
    ids = np.array([f"cell_{i:04d}" for i in range(n)]) if cell_ids is None else np.asarray(cell_ids, dtype=str)
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kernels, cell_ids=ids,
                        metadata_json=np.array(json.dumps(_json_safe(metadata or {}))))
    return path


def simulate_paired(kernels, config=None):
    """Simulate one data set with the programs of ``config.components_per_topic``.

    E.g. (2, 2, 0) = two-kernel data (phylo + cell_type programs), (2, 2, 2) = with interaction.
    Returns a dict with X (log mean = signal + noise), signal (H B^T), noise (structured + iid),
    structured_noise, Y and Y_noise_free (Poisson counts of exp(X) and exp(signal)), H, B, topics,
    true_rank (= number of programs), snr (column-centered signal power / noise power, recorded
    only) and config. Ground truth is for evaluation only; never supply it to a selector.

    Within a topic: h_j = sqrt(rho)*u + sqrt(1-rho)*e_j, independent u,e_j ~ N(0, K_topic).
    """
    if config is None:
        config = SimulationConfig()
    kernel = validate_kernels(kernels)
    counts = config.components_per_topic
    r = sum(counts)                      # number of programs = true rank
    n_genes = config.n_genes
    rng = np.random.default_rng(config.seed)
    n_cells = kernel["phylo"].shape[0]
    # Gene layout: exclusive_genes per component, then shared_genes loading on every component, the rest noise.
    if config.exclusive_genes * r + config.shared_genes > n_genes:
        raise ValueError("n_genes must be >= exclusive_genes * number of components + shared_genes.")
    H = np.zeros((n_cells, r), dtype=float)
    B = np.zeros((n_genes, r), dtype=float)
    for i, num in enumerate(counts):
        if num == 0:  # no program from this kernel (e.g. interaction in the two-kernel data)
            continue
        # h ~ N(0, K) drawn as K^{1/2} z with the symmetric root. rng.multivariate_normal factors K by an SVD whose
        # basis of repeated eigenvalues changes with the BLAS thread count, so the same seed gave different H.
        w, V = np.linalg.eigh(kernel[TOPICS[i]])
        root = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T
        if config.rho == 0:
            H[:, sum(counts[:i]):sum(counts[:i])+num] = root @ rng.standard_normal((n_cells, num))
        else:
            # rho != 0: h_j = sqrt(rho)*u + sqrt(1-rho)*e_j, with one shared draw u per topic
            u = root @ rng.standard_normal(n_cells)
            e = root @ rng.standard_normal((n_cells, num))
            H[:, sum(counts[:i]):sum(counts[:i])+num] = np.sqrt(config.rho) * u[:, None] + np.sqrt(1 - config.rho) * e
    # simulate comp spec gene loadings
    for i in range(r):
        B[config.exclusive_genes*i:config.exclusive_genes*(i+1),i] = rng.gamma(shape=2.0, scale=1.0, size=config.exclusive_genes)
    # simulate shared genes: shared_genes genes in total, each with a loading on every component
    B[config.exclusive_genes*r:config.exclusive_genes*r+config.shared_genes,:] = rng.gamma(shape=2.0, scale=1.0, size=(config.shared_genes, r))/r
    # for noise genes, B are all zero, we do not need to further simulate entries in B
    if config.scale_B:
        B /= r  # advisor's normalization (B / 5 there), here by the number of programs
    signal = H @ B.T
    # Noise: rank-10 structured part  P1 @ P2^T, unrelated to any kernel, + iid part.
    structured = (rng.normal(scale=config.structured_noise_std, size=(n_cells, STRUCTURED_NOISE_RANK))
                  @ rng.normal(scale=config.structured_noise_std, size=(n_genes, STRUCTURED_NOISE_RANK)).T)
    unstructured = rng.normal(scale=config.unstructured_noise_std, size=signal.shape)
    # Optional: set a noise part by its SNR instead of its std (column-centered power per entry, as the CCA centers genes).
    power = lambda M: float(np.var(M, axis=0).mean())
    if config.structured_snr is not None:
        structured *= np.sqrt(power(signal) / (config.structured_snr * power(structured)))
    if config.unstructured_snr is not None:
        unstructured *= np.sqrt(power(signal) / (config.unstructured_snr * power(unstructured)))
    noise = structured + unstructured
    X = signal + noise
    return {
        "X": X, "signal": signal, "noise": noise, "structured_noise": structured,
        # Poisson counts with log mean X (noisy) or signal (noise-free); the data is log1p(Y).
        "Y": rng.poisson(np.exp(X)), "Y_noise_free": rng.poisson(np.exp(signal)),
        # Ground truth (evaluation only).
        "H": H, "B": B, "topics": np.repeat(np.asarray(TOPICS), counts), "true_rank": r,
        # Column-centered signal power / noise power (as the CCA centers genes), recorded only.
        "snr": power(signal) / power(noise),
        "config": _json_safe(asdict(config)),
    }


__all__ = ["TOPICS", "STRUCTURED_NOISE_RANK", "SimulationConfig", "validate_kernels",
           "make_demo_kernels", "load_kernel_npz", "save_kernel_npz", "simulate_paired"]

