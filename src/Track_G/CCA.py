"""Witten nonnegative sparse multi-kernel CCA (Track-G), kernel concatenation, gene weights and saving."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from .parallel_analysis import PAResult


def nonnegative_soft_threshold(x: np.ndarray, delta: float) -> np.ndarray:
    """Return S(x_+, delta) = (x - delta)_+."""
    return np.maximum(np.asarray(x, dtype=float) - float(delta), 0.0)


def _validated_c(c_l1: Optional[float], p: int) -> float:
    """Validate/cap a Witten L1 bound; c >= sqrt(p) is nonbinding."""
    if p < 1:
        raise ValueError("The loading dimension must be positive.")

    sqrt_p = np.sqrt(float(p))
    if c_l1 is None:
        return sqrt_p

    c_l1 = float(c_l1)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, sqrt_p)
    if not np.isfinite(c_l1):
        raise ValueError("c_l1 must be finite.")
    if c_l1 < 1.0 - tolerance:
        raise ValueError(
            "The unit-L2 Witten update requires c_l1 >= 1; "
            f"received {c_l1}."
        )

    # A larger value is mathematically equivalent to sqrt(p).
    return min(max(c_l1, 1.0), sqrt_p)


def witten_c_from_legacy_delta(delta_value: float, p: int) -> float:
    """Reproduce the old experiment mapping c=sqrt(delta*p).

    This is a compatibility helper only.  In the paper, c is the user-supplied
    L1 bound, whereas Delta is the threshold solved inside every block update.
    """
    delta_value = float(delta_value)
    if not np.isfinite(delta_value) or delta_value < 0.0:
        raise ValueError("legacy delta_value must be finite and nonnegative.")
    sqrt_p = np.sqrt(float(p))
    legacy_lower_bound = min(1.01, sqrt_p)
    legacy_c = min(
        max(np.sqrt(delta_value * float(p)), legacy_lower_bound),
        sqrt_p,
    )
    return _validated_c(legacy_c, p)


def _tied_maximum_solution(
    top_indices: np.ndarray,
    c_l1: float,
    p: int,
) -> np.ndarray:
    """Construct an exact optimizer when the positive gradient maximum is tied.

    If c is below sqrt(number of tied maxima), scalar thresholding cannot break
    the tie.  Any vector supported on the tied coordinates with L1=c and L2=1
    attains the global upper bound max(grad)*c.  This function returns one such
    deterministic vector.
    """
    top_indices = np.asarray(top_indices, dtype=int)
    m = int(top_indices.size)
    if m < 1:
        raise ValueError("At least one maximum-gradient index is required.")

    c = min(float(c_l1), np.sqrt(float(m)))
    roundoff = 64.0 * np.finfo(float).eps * max(1.0, c * c)
    q = max(1, min(m, int(np.ceil(c * c - roundoff))))

    weights = np.zeros(q, dtype=float)
    if q == 1:
        weights[0] = 1.0
    else:
        r = q - 1
        discriminant = max(0.0, r * (q - c * c))
        repeated_weight = (
            c * r + np.sqrt(discriminant)
        ) / (r * q)
        last_weight = c - r * repeated_weight
        weights[:r] = repeated_weight
        weights[r] = max(last_weight, 0.0)
        weights /= np.linalg.norm(weights)

    vector = np.zeros(p, dtype=float)
    vector[top_indices[:q]] = weights
    return vector


def witten_nonnegative_update(
    grad: np.ndarray,
    c_l1: float,
    *,
    max_bisect_iter: int = 100,
    l1_tolerance: float = 1e-10,
    support_tolerance: float = 1e-12,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    """Solve one exact nonnegative Witten block update.

    Returns ``(vector, Delta, L1, nnz)``.  If the gradient has no positive
    coordinate, zero is an exact block optimum.  ``vector`` is then ``None`` so
    the outer nonconvex solver can reject that restart instead of inventing an
    old/random/uniform unit vector.
    """
    grad = np.asarray(grad, dtype=float).reshape(-1)
    if grad.size == 0:
        raise ValueError("The Witten gradient is empty.")
    if not np.all(np.isfinite(grad)):
        raise ValueError("The Witten gradient contains NaN or Inf.")
    if not isinstance(max_bisect_iter, (int, np.integer)) or max_bisect_iter < 1:
        raise ValueError("max_bisect_iter must be positive.")
    if not np.isfinite(l1_tolerance) or l1_tolerance <= 0.0:
        raise ValueError("l1_tolerance must be finite and positive.")
    if not np.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative.")

    p = int(grad.size)
    c_l1 = _validated_c(c_l1, p)
    positive_grad = np.maximum(grad, 0.0)
    grad_max = float(np.max(positive_grad))

    if grad_max <= 0.0:
        return None, np.nan, 0.0, 0

    # Scaling leaves the optimizer unchanged and prevents overflow/underflow.
    scaled_grad = positive_grad / grad_max
    vector_no_penalty = scaled_grad / np.linalg.norm(scaled_grad)
    l1_no_penalty = float(np.sum(vector_no_penalty))
    requested_tolerance = l1_tolerance * max(1.0, c_l1)

    if l1_no_penalty <= c_l1 + requested_tolerance:
        return (
            vector_no_penalty,
            0.0,
            l1_no_penalty,
            int(np.count_nonzero(vector_no_penalty > support_tolerance)),
        )

    # Handle exact maximum ties, including c=1, without returning an infeasible
    # equal-weight vector.
    top_indices = np.flatnonzero(scaled_grad == 1.0)
    top_limit_l1 = np.sqrt(float(top_indices.size))
    if c_l1 <= top_limit_l1 + requested_tolerance:
        vector = _tied_maximum_solution(top_indices, c_l1, p)
        vector_l1 = float(np.sum(vector))

        lower_values = scaled_grad[scaled_grad < 1.0]
        if abs(vector_l1 - top_limit_l1) <= requested_tolerance:
            lower_breakpoint = (
                float(np.max(lower_values)) if lower_values.size else 0.0
            )
            threshold = 0.5 * (1.0 + lower_breakpoint) * grad_max
        else:
            # No single scalar threshold selects an asymmetric tied solution.
            threshold = np.nan

        return (
            vector,
            threshold,
            vector_l1,
            int(np.count_nonzero(vector > support_tolerance)),
        )

    # F(delta)=||S(g,delta)||_1/||S(g,delta)||_2 is monotone decreasing.
    lo, hi = 0.0, 1.0

    # Feasible left-limit solution at delta=max(g).
    best_vector = np.zeros(p, dtype=float)
    best_vector[top_indices] = 1.0 / np.sqrt(float(top_indices.size))
    best_l1 = float(np.sum(best_vector))
    best_delta = grad_max

    for _ in range(max_bisect_iter):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break

        thresholded = nonnegative_soft_threshold(scaled_grad, mid)
        thresholded_norm = np.linalg.norm(thresholded)
        if thresholded_norm == 0.0:
            hi = mid
            continue

        candidate = thresholded / thresholded_norm
        candidate_l1 = float(np.sum(candidate))

        if candidate_l1 > c_l1 + requested_tolerance:
            lo = mid
        else:
            hi = mid
            # Keep only feasible candidates, choosing the one closest to the
            # active L1 boundary from below.
            if candidate_l1 > best_l1:
                best_vector = candidate
                best_l1 = candidate_l1
                best_delta = mid * grad_max

        if abs(candidate_l1 - c_l1) <= requested_tolerance:
            if candidate_l1 <= c_l1 + requested_tolerance:
                best_vector = candidate
                best_l1 = candidate_l1
                best_delta = mid * grad_max
            break

    feasibility_tolerance = 10.0 * requested_tolerance
    if np.sum(best_vector) > c_l1 + feasibility_tolerance:
        raise RuntimeError("Witten bisection returned an L1-infeasible vector.")

    return (
        best_vector,
        best_delta,
        best_l1,
        int(np.count_nonzero(best_vector > support_tolerance)),
    )


def random_feasible_unit_vector(
    p: int,
    c_l1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a nonnegative initialization satisfying both norm bounds."""
    c_l1 = _validated_c(c_l1, p)
    sqrt_p = np.sqrt(float(p))
    support_size = (
        p
        if np.isclose(c_l1, sqrt_p, rtol=0.0, atol=1e-12)
        else max(1, min(p, int(np.floor(c_l1 * c_l1))))
    )

    support = rng.choice(p, size=support_size, replace=False)
    vector = np.zeros(p, dtype=float)
    vector[support] = rng.random(support_size) + 0.1
    vector /= np.linalg.norm(vector)

    if np.sum(vector) > c_l1 + 1e-10:
        raise RuntimeError("Generated Witten initialization is not L1-feasible.")
    return vector


def _make_w_initializations(
    Y_current: np.ndarray,
    c_w2: float,
    n_restarts: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Create guaranteed-positive one-hot seeds plus random feasible seeds."""
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")

    p_gene = int(Y_current.shape[1])
    column_maxima = np.max(Y_current, axis=0)
    positive_columns = np.flatnonzero(column_maxima > 0.0)
    if positive_columns.size == 0:
        return []

    ranked_columns = positive_columns[
        np.argsort(column_maxima[positive_columns])[::-1]
    ]
    n_one_hot = min(positive_columns.size, max(1, n_restarts // 2))

    initializations: List[np.ndarray] = []
    for column_index in ranked_columns[:n_one_hot]:
        vector = np.zeros(p_gene, dtype=float)
        vector[int(column_index)] = 1.0
        initializations.append(vector)

    while len(initializations) < n_restarts:
        initializations.append(
            random_feasible_unit_vector(p_gene, c_w2, rng)
        )
    return initializations


def _solve_single_restart(
    Y_current: np.ndarray,
    w_initial: np.ndarray,
    c_alpha: float,
    c_w2: float,
    max_iter: int,
    vector_tolerance: float,
    objective_tolerance: float,
) -> Optional[Dict[str, Any]]:
    """Run the paper's order: initialize w, then update a, then update w."""
    w = np.asarray(w_initial, dtype=float).copy()
    previous_a: Optional[np.ndarray] = None
    previous_objective: Optional[float] = None
    converged = False

    final_delta_a = np.nan
    final_delta_w2 = np.nan
    final_l1_a = np.nan
    final_l1_w2 = np.nan
    final_nnz_a = 0
    final_nnz_w2 = 0

    for inner_iter in range(max_iter):
        a, delta_a, l1_a, nnz_a = witten_nonnegative_update(
            Y_current @ w,
            c_alpha,
        )
        if a is None:
            return None

        objective_after_a = float(a @ Y_current @ w)
        w_new, delta_w2, l1_w2, nnz_w2 = witten_nonnegative_update(
            Y_current.T @ a,
            c_w2,
        )
        if w_new is None:
            return None

        objective = float(a @ Y_current @ w_new)
        scale = max(1.0, abs(objective), abs(objective_after_a))
        if objective < objective_after_a - 1e-9 * scale:
            raise RuntimeError("A Witten W update reduced the PMD objective.")

        if previous_a is None or previous_objective is None:
            vector_change = np.inf
            relative_objective_change = np.inf
        else:
            scale = max(1.0, abs(objective), abs(previous_objective))
            if objective < previous_objective - 1e-9 * scale:
                raise RuntimeError("Witten coordinate ascent lost monotonicity.")
            vector_change = max(
                np.linalg.norm(a - previous_a),
                np.linalg.norm(w_new - w),
            )
            relative_objective_change = (
                abs(objective - previous_objective) / scale
            )

        previous_a = a.copy()
        w = w_new
        previous_objective = objective
        final_delta_a = delta_a
        final_delta_w2 = delta_w2
        final_l1_a = l1_a
        final_l1_w2 = l1_w2
        final_nnz_a = nnz_a
        final_nnz_w2 = nnz_w2

        if (
            vector_change <= vector_tolerance
            and relative_objective_change <= objective_tolerance
        ):
            converged = True
            break

    if previous_a is None or previous_objective is None:
        return None
    if previous_objective <= 0.0:
        return None

    return {
        "a": previous_a,
        "w": w,
        "objective": float(previous_objective),
        "iterations": inner_iter + 1,
        "converged": converged,
        "delta_a": final_delta_a,
        "delta_w2": final_delta_w2,
        "l1_a": final_l1_a,
        "l1_w2": final_l1_w2,
        "nnz_a": final_nnz_a,
        "nnz_w2": final_nnz_w2,
    }


def _paired_pearson(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Columnwise Pearson correlations for paired score matrices."""
    if U.shape != V.shape:
        raise ValueError("Paired score matrices must have the same shape.")

    U_centered = U - U.mean(axis=0, keepdims=True)
    V_centered = V - V.mean(axis=0, keepdims=True)
    numerator = np.sum(U_centered * V_centered, axis=0)
    denominator = (
        np.linalg.norm(U_centered, axis=0)
        * np.linalg.norm(V_centered, axis=0)
    )

    correlations = np.full(U.shape[1], np.nan, dtype=float)
    valid = denominator > 0.0
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def fit_witten_nonnegative_cca(
    K: np.ndarray,
    X2_np: np.ndarray,
    KERNEL_NAMES: Sequence[str],
    *,
    J: int = 18,
    c_alpha: Optional[float] = None,
    c_w2: Optional[float] = None,
    n_restarts: int = 10,
    random_seed: int = 42,
    max_cca_iter: int = 2000,
    cca_tolerance: float = 1e-6,
    objective_tolerance: float = 1e-10,
    require_convergence: bool = True,
    require_all_components: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit sequential nonnegative sparse PMD and return legacy output names.

    ``c_alpha`` and ``c_w2`` are the paper's direct L1 bounds.  Passing ``None``
    uses the nonbinding maxima sqrt(number of features).  Multiple restarts and
    invalid-component rejection are numerical robustness additions; each block
    update itself is the paper's equation (8).
    """
    K = np.asarray(K, dtype=float)
    X2_np = np.asarray(X2_np, dtype=float)
    kernel_names = np.asarray(KERNEL_NAMES, dtype=object)

    if K.ndim != 2 or X2_np.ndim != 2:
        raise ValueError("K and X2_np must both be two-dimensional matrices.")
    if K.shape[0] != X2_np.shape[0]:
        raise ValueError("K and X2_np must have the same number of cells.")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(X2_np)):
        raise ValueError("K and X2_np must contain only finite values.")
    if not isinstance(J, (int, np.integer)) or J < 1:
        raise ValueError("J must be a positive integer.")
    if not isinstance(max_cca_iter, (int, np.integer)) or max_cca_iter < 1:
        raise ValueError("max_cca_iter must be a positive integer.")
    if not isinstance(n_restarts, (int, np.integer)) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer.")
    if not isinstance(random_seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer.")
    if not np.isfinite(cca_tolerance) or cca_tolerance <= 0.0:
        raise ValueError("cca_tolerance must be finite and positive.")
    if not np.isfinite(objective_tolerance) or objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and nonnegative.")
    if not isinstance(require_convergence, (bool, np.bool_)):
        raise ValueError("require_convergence must be Boolean.")
    if not isinstance(require_all_components, (bool, np.bool_)):
        raise ValueError("require_all_components must be Boolean.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("verbose must be Boolean.")
    # if kernel_names.size != 3:
    #     raise ValueError("This compatibility wrapper expects exactly 3 kernels.")
    n_cells = int(K.shape[0])
    p_alpha = int(K.shape[1])
    p_gene  = int(X2_np.shape[1])
    n_kernels = int(kernel_names.size)
    if p_alpha != n_kernels * n_cells:
        raise ValueError(
        "K columns must be n_kernels * n_cells-wide blocks in KERNEL_NAMES order.")

    c_alpha_static = _validated_c(c_alpha, p_alpha)
    c_w2_static = _validated_c(c_w2, p_gene)
    Y_residual = K.T @ X2_np

    A_matrix = np.zeros((p_alpha, J), dtype=float)
    W2_matrix = np.zeros((p_gene, J), dtype=float)
    objective_record = np.zeros(J, dtype=float)
    iterations_record = np.zeros(J, dtype=int)
    restart_record = np.full(J, -1, dtype=int)
    converged_record = np.zeros(J, dtype=bool)
    extracted_record = np.zeros(J, dtype=bool)
    residual_fro_before = np.full(J, np.nan, dtype=float)
    residual_fro_after = np.full(J, np.nan, dtype=float)
    nnz_a_record = np.zeros(J, dtype=int)
    nnz_w2_record = np.zeros(J, dtype=int)
    delta_a_record = np.full(J, np.nan, dtype=float)
    delta_w2_record = np.full(J, np.nan, dtype=float)
    l1_a_record = np.zeros(J, dtype=float)
    l1_w2_record = np.zeros(J, dtype=float)

    if verbose:
        print(f"Witten c_alpha={c_alpha_static:.12g}")
        print(f"Witten c_w2={c_w2_static:.12g}")
        print("Initial residual Frobenius norm:", np.linalg.norm(Y_residual, "fro"))

    failure_reason: Optional[str] = None
    for j in range(J):
        residual_fro_before[j] = np.linalg.norm(Y_residual, "fro")
        signal_tolerance = (
            100.0 * np.finfo(float).eps
            * max(1.0, residual_fro_before[j])
        )
        if float(np.max(Y_residual)) <= signal_tolerance:
            failure_reason = "no positive residual signal remains"
            break

        rng = np.random.default_rng(random_seed + j)
        w_initializations = _make_w_initializations(
            Y_residual,
            c_w2_static,
            n_restarts,
            rng,
        )

        candidates: List[Tuple[int, Dict[str, Any]]] = []
        for restart_index, w_initial in enumerate(w_initializations):
            candidate = _solve_single_restart(
                Y_residual,
                w_initial,
                c_alpha_static,
                c_w2_static,
                max_cca_iter,
                cca_tolerance,
                objective_tolerance,
            )
            if candidate is None:
                continue
            if require_convergence and not candidate["converged"]:
                continue
            candidates.append((restart_index, candidate))

        if not candidates:
            failure_reason = "all Witten restarts failed or did not converge"
            break

        best_restart_index, best = max(
            candidates,
            key=lambda item: item[1]["objective"],
        )
        scalar_c = float(best["objective"])
        if scalar_c <= signal_tolerance:
            failure_reason = "remaining PMD objective is numerically zero"
            break

        a_j = best["a"]
        w2_j = best["w"]
        A_matrix[:, j] = a_j
        W2_matrix[:, j] = w2_j
        objective_record[j] = scalar_c
        iterations_record[j] = int(best["iterations"])
        restart_record[j] = best_restart_index
        converged_record[j] = bool(best["converged"])
        extracted_record[j] = True
        nnz_a_record[j] = int(best["nnz_a"])
        nnz_w2_record[j] = int(best["nnz_w2"])
        delta_a_record[j] = float(best["delta_a"])
        delta_w2_record[j] = float(best["delta_w2"])
        l1_a_record[j] = float(best["l1_a"])
        l1_w2_record[j] = float(best["l1_w2"])

        # Witten PMD rank-one deflation of the cross-product matrix.
        Y_residual = Y_residual - scalar_c * np.outer(a_j, w2_j)
        residual_fro_after[j] = np.linalg.norm(Y_residual, "fro")
        allowed_increase = 1e-9 * max(1.0, residual_fro_before[j])
        if residual_fro_after[j] > residual_fro_before[j] + allowed_increase:
            raise RuntimeError("PMD deflation increased the residual Frobenius norm.")

        if verbose:
            print(
                f"Component {j + 1:2d}/{J} | restart={best_restart_index:2d} | "
                f"iter={iterations_record[j]:4d} | objective={scalar_c:.8f} | "
                f"Delta(A)={delta_a_record[j]:.5g} | "
                f"Delta(W2)={delta_w2_record[j]:.5g} | "
                f"NNZ(A)={nnz_a_record[j]:4d} | "
                f"NNZ(W2)={nnz_w2_record[j]:4d}"
            )

    n_extracted = int(np.count_nonzero(extracted_record))
    if n_extracted < J and require_all_components:
        raise RuntimeError(
            f"Only {n_extracted}/{J} valid Witten components were extracted: "
            f"{failure_reason}. No fallback component was fabricated."
        )
    
    A_blocks = []
    scores_list = []
    for i in range(n_kernels):
        block = A_matrix[i*n_cells:(i+1)*n_cells, :]
        A_blocks.append(block)
        scores_list.append(np.linalg.norm(block, axis=0))
    all_scores = np.vstack(scores_list)  # (n_kernels, J)

    # The out-of-range sentinel fails loudly if legacy code tries to index
    # KERNEL_NAMES without first checking extracted_record.
    not_extracted_index = int(kernel_names.size)
    winning_kernels = np.full(J, not_extracted_index, dtype=int)
    if n_extracted:
        winning_kernels[extracted_record] = np.argmax(
            all_scores[:, extracted_record],
            axis=0,
        )
    winning_kernel_names = np.full(J, "not_extracted", dtype=object)
    winning_kernel_names[extracted_record] = kernel_names[
        winning_kernels[extracted_record]
    ]
    kernel_assignments = {
        k: np.flatnonzero(winning_kernels == k).tolist()
        for k in range(kernel_names.size)
    }
    a_winning_kernels = winning_kernel_names.tolist()

    U_score_matrix = K @ A_matrix
    V_score_matrix = X2_np @ W2_matrix
    pearson_record = _paired_pearson(U_score_matrix, V_score_matrix)
    pearson_record[~extracted_record] = np.nan

    # Extra scale-aware post-hoc assignment.  Legacy scores above are unchanged.
    contribution_all_scores = []
    for kernel_index in range(kernel_names.size):
        start = kernel_index * n_cells
        end = (kernel_index + 1) * n_cells
        contribution = K[:, start:end] @ A_matrix[start:end, :]
        contribution -= contribution.mean(axis=0, keepdims=True)
        contribution_all_scores.append(np.linalg.norm(contribution, axis=0))
    contribution_all_scores = np.vstack(contribution_all_scores)
    contribution_winning_kernels = np.full(
        J,
        not_extracted_index,
        dtype=int,
    )
    if n_extracted:
        contribution_winning_kernels[extracted_record] = np.argmax(
            contribution_all_scores[:, extracted_record],
            axis=0,
        )

    scores_cols = {
        f"score_{kernel_names[i]}": scores_list[i]
        for i in range(n_kernels)
    }
    contribution_cols = {
        f"contribution_{kernel_names[i]}": contribution_all_scores[i]
        for i in range(n_kernels)}
    
    cca_diagnostics = pd.DataFrame(
        {
            "component_index": np.arange(J),
            "component": [f"C_{j}" for j in range(J)],
            "extracted": extracted_record,
            "selected_restart": restart_record,
            "iterations": iterations_record,
            "converged": converged_record,
            "objective": objective_record,
            "pearson_KA_XW": pearson_record,
            "residual_fro_before": residual_fro_before,
            "residual_fro_after": residual_fro_after,
            "nnz_A": nnz_a_record,
            "nnz_W2": nnz_w2_record,
            "l1_A": l1_a_record,
            "l1_W2": l1_w2_record,
            "Delta_A": delta_a_record,
            "Delta_W2": delta_w2_record,
            **scores_cols,
            **contribution_cols,
            "winning_kernel": winning_kernel_names,
        }
    )

    # Both KP-Tracer and C. elegans aliases are returned so existing notebooks
    # keep working without renaming any downstream matrix.
    A_block_dict = {
        f"A_{kernel_names[i]}": A_blocks[i]
        for i in range(n_kernels)
    }
    scores_dict = {
        f"scores_{kernel_names[i]}": scores_list[i]
        for i in range(n_kernels)
    }

    return {
    "A_matrix": A_matrix,
    "W2_matrix": W2_matrix,
    **A_block_dict,
    **scores_dict,
    "all_scores": all_scores,
    "winning_kernels": winning_kernels,
    "winning_kernel_names": winning_kernel_names,
    "a_winning_kernels": a_winning_kernels,
    "kernel_assignments": kernel_assignments,
    "objective_record": objective_record,
    "iterations_record": iterations_record,
    "inner_iterations_record": iterations_record,
    "restart_record": restart_record,
    "converged_record": converged_record,
    "extracted_record": extracted_record,
    "residual_fro_before": residual_fro_before,
    "residual_fro_after": residual_fro_after,
    "nnz_a_record": nnz_a_record,
    "nnz_w2_record": nnz_w2_record,
    "delta_a_record": delta_a_record,
    "delta_w2_record": delta_w2_record,
    "l1_a_record": l1_a_record,
    "l1_w2_record": l1_w2_record,
    "U_score_matrix": U_score_matrix,
    "V_score_matrix": V_score_matrix,
    "pearson_record": pearson_record,
    "contribution_all_scores": contribution_all_scores,
    "contribution_winning_kernels": contribution_winning_kernels,
    "cca_diagnostics": cca_diagnostics,
    "Y_residual": Y_residual,
    "c_alpha": c_alpha_static,
    "c_w2": c_w2_static,
    "c_alpha_static": c_alpha_static,
    "c_w2_static": c_w2_static,
    "n_extracted": n_extracted,
}


def build_kernel_matrix(kernels: Dict[str, np.ndarray], kernel_names: Sequence[str],
                        frobenius_normalize: bool = False,
                        use_sqrt: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Concatenate cell kernels column-wise in ``kernel_names`` order: K = [K_1 ... K_m] (n x mn).

    ``frobenius_normalize=True`` first divides each kernel by its Frobenius norm; the default keeps
    the kernels as they are (standard concatenation).
    ``use_sqrt=True`` (default, Track-G) replaces each kernel K_i with its symmetric square root K_i^{1/2}
    (via eigendecomposition, negative eigenvalues clipped to 0); ``use_sqrt=False`` uses K_i itself.
    Returns ``(K, kernels_for_cca)``.
    """
    names = [str(name) for name in kernel_names]

    if use_sqrt:
        def matrix_sqrt(M: np.ndarray) -> np.ndarray:
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.clip(eigvals, 0.0, None)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        kernels = {name: matrix_sqrt(kernels[name]) for name in names}

    if frobenius_normalize:
        kernels_for_cca = {name: kernels[name] / np.linalg.norm(kernels[name], "fro") for name in names}
    else:
        kernels_for_cca = {name: kernels[name] for name in names}

    K = np.concatenate([kernels_for_cca[name] for name in names], axis=1)
    return K, kernels_for_cca

def compute_gene_weights_all_components(
    K: np.ndarray,
    X2_np: np.ndarray,
    W2_matrix: np.ndarray,
    A_matrix: np.ndarray,
    valid_genes,
    w_threshold: float = 1e-4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-gene diagnostics for every CCA component.

    For each component j and each gene g with W[g, j] > w_threshold, computes:
      - W            : gene weight in W2 matrix for this component
      - rank         : rank of W within this component (1 = highest)
      - loo_obj      : leave-one-out objective (note: monotone in W when converged,
                       so adds no information beyond W; kept for completeness)
      - obj_comp     : original objective of this component (scalar, same for all
                       genes in the same component)
      - loo_drop     : obj_comp - loo_obj, the drop in objective when gene is removed
      - pearson      : Pearson correlation between gene expression and KA score
      - pearson_pval : corresponding p-value
      - component    : component index (0-indexed)

    Genes with W <= w_threshold are excluded entirely (not NaN rows).

    Parameters
    ----------
    K          : (n_cells, n_kernels * n_cells) concatenated kernel matrix
    X2_np      : (n_cells, n_genes) z-score standardized expression
    W2_matrix  : (n_genes, J) gene loading matrix from Witten CCA
    A_matrix   : (n_kernels * n_cells, J) kernel coefficient matrix
    valid_genes: gene names, length n_genes
    w_threshold: minimum W to include a gene (default 1e-4)
    verbose    : print progress per component

    Returns
    -------
    pd.DataFrame with MultiIndex (gene, component), all dimensions saved
    """
    W2_matrix = W2_matrix.copy()
    A_matrix = A_matrix.copy()
    J = W2_matrix.shape[1]
    valid_genes = list(valid_genes)

    all_records = []

    for idx_comp in range(J):
        # Reconstruct the residual for the current component
        Y_residual = K.T @ X2_np
        for i in range(idx_comp):
            obj_i = Y_residual.dot(W2_matrix[:, i]).dot(A_matrix[:, i])
            Y_residual = Y_residual - obj_i * np.outer(A_matrix[:, i], W2_matrix[:, i])

        obj_comp = float(Y_residual.dot(W2_matrix[:, idx_comp]).dot(A_matrix[:, idx_comp]))
        ka = (K @ A_matrix)[:, idx_comp]

        if verbose:
            print(f"Component {idx_comp}: objective = {obj_comp:.6f}")

        # Sort genes by W in descending order
        w_vec = W2_matrix[:, idx_comp]
        sorted_idx = np.argsort(w_vec)[::-1]

        rank = 0
        for gene_rank_idx, gene_col in enumerate(sorted_idx):
            w_val = float(w_vec[gene_col])
            if w_val <= w_threshold:
                break  # Already sorted in descending order, all subsequent values are <= threshold

            rank += 1
            gene_name = valid_genes[gene_col]

            # LOO objective
            new_w = w_vec.copy()
            new_w[gene_col] = 0.0
            new_w_norm = np.linalg.norm(new_w)
            if new_w_norm > 0:
                new_w /= new_w_norm
            loo_obj = float(Y_residual.dot(new_w).dot(A_matrix[:, idx_comp]))

            # Pearson
            r, p = scipy.stats.pearsonr(ka, X2_np[:, gene_col])

            all_records.append({
                'gene':         gene_name,
                'component':    idx_comp,
                'W':            w_val,
                'rank':         rank,
                'obj_comp':     obj_comp,
                'loo_obj':      loo_obj,
                'loo_drop':     obj_comp - loo_obj,
                'pearson':      float(r),
                'pearson_pval': float(p),
            })

    df = pd.DataFrame(all_records).set_index(['gene', 'component'])
    return df


def gene_kernel_scores(W2_matrix: np.ndarray, all_scores: np.ndarray, gene_names: Optional[Sequence[str]] = None,
                       kernel_names: Optional[Sequence[str]] = None, component_names: Optional[Sequence[str]] = None,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Relatedness of each component and each gene to each kernel (cellular attribute).

    p[k, j] = ||alpha_j^(k)||_2 / sum_l ||alpha_j^(l)||_2   (``all_scores`` = A-block norms, kernels x components)
    s[g, k] = sum_j p[k, j] * W2[g, j]                     (genes rank high for k if they load on k-related components)
    Returns ``(p, s)``: p is kernels x components (columns sum to 1), s is genes x kernels.
    """
    S = np.asarray(all_scores, dtype=float)
    col = S.sum(0, keepdims=True)
    P = np.divide(S, col, out=np.zeros_like(S), where=col > 0)          # components with A = 0 get p = 0
    kernels = [str(k) for k in kernel_names] if kernel_names is not None else [f"K{k}" for k in range(S.shape[0])]
    components = list(component_names) if component_names is not None else [f"C_{j}" for j in range(S.shape[1])]
    p = pd.DataFrame(P, index=kernels, columns=components)
    s = pd.DataFrame(np.asarray(W2_matrix, dtype=float) @ P.T, index=gene_names, columns=kernels)
    return p, s


# Saving (dataset-independent)
def save_pa_cca_outputs(path, result: Dict[str, Any], data, pa: Optional[PAResult] = None,
                        settings: Optional[dict] = None) -> Path:
    """Save one PA + CCA run as a single, self-contained pickle of labelled DataFrames.

    ``out = pd.read_pickle(path)`` is a dict with
      W2                       genes x components: gene loadings (e.g. for pathway enrichment)
      A                        (kernel, cell) x components: kernel coefficients
      KA, XW                   cells x components: canonical scores K @ A and X @ W2
      cca_diagnostics          one row per component: winning kernel, objective, Pearson(KA, XW), ...
      cells                    cell metadata, if ``data`` has ``cell_metadata``
      genes                    per gene: fraction of expressing cells and mean expression, if available
      gene_weights             per-gene per-component diagnostics (W, rank, loo_obj, loo_drop, pearson, pearson_pval); only genes with W > w_threshold included;
                               present if ``data`` has both ``K`` and ``X2_np``
      pa_summary, pa_spectrum  PA choice of J and the spectrum/threshold of every null, if ``pa`` is given
      settings                 dict of run settings (filter, Frobenius, J, PA and CCA parameters)
    Components are labelled C_0, C_1, ... as in ``cca_diagnostics["component"]``; ``out["A"].to_numpy()``
    gives back the A matrix.

    ``data`` is the dict returned by the dataset loader.  It needs ``valid_genes``, ``cell_names`` and
    ``KERNEL_NAMES`` and uses ``cell_metadata``, ``gene_frac_expr``, ``log_exp_filtered``,
    ``min_expr_frac`` and ``frobenius_normalize`` when present.  ``settings`` (e.g. the keyword
    arguments given to ``fit_witten_nonnegative_cca``) is stored under ``settings["cca"]``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    components = pd.Index(result["cca_diagnostics"]["component"], name="component")
    genes = pd.Index(np.asarray(data["valid_genes"], dtype=str), name="gene")
    cells = pd.Index(np.asarray(data["cell_names"], dtype=str), name="cell")
    kernel_names = list(data["kernels_for_cca"].keys())
    kernel_cells = pd.MultiIndex.from_product([kernel_names, cells], names=["kernel", "cell"])

    out = {
        "W2": pd.DataFrame(result["W2_matrix"], index=genes, columns=components),
        "A": pd.DataFrame(result["A_matrix"], index=kernel_cells, columns=components),
        "KA": pd.DataFrame(result["U_score_matrix"], index=cells, columns=components),
        "XW": pd.DataFrame(result["V_score_matrix"], index=cells, columns=components),
        "cca_diagnostics": result["cca_diagnostics"],
    }

    if data.get("cell_metadata") is not None:
        cell_metadata = data["cell_metadata"].copy()
        cell_metadata.index = cell_metadata.index.astype(str)
        out["cells"] = cell_metadata.reindex(cells)

    gene_columns = {}
    if data.get("gene_frac_expr") is not None:
        gene_columns["frac_expr"] = data["gene_frac_expr"].reindex(data["valid_genes"]).to_numpy()
    if data.get("log_exp_filtered") is not None:
        gene_columns["mean_log_exp"] = data["log_exp_filtered"].mean(axis=0).to_numpy()
    if gene_columns:
        out["genes"] = pd.DataFrame(gene_columns, index=genes)

    run_settings = dict(
        kernel_names=kernel_names, n_cells=len(cells), n_genes=len(genes),
        min_expr_frac=data.get("min_expr_frac"), frobenius_normalize=data.get("frobenius_normalize"),
        J=int(result["A_matrix"].shape[1]), n_extracted=int(result["n_extracted"]),
        c_alpha=float(result["c_alpha"]), c_w2=float(result["c_w2"]),
    )
    if pa is not None:
        out["pa_summary"] = pa.summary
        r = pa.result
        out["pa_spectrum"] = pd.DataFrame({
            "null": pa.null,
            "rank": np.arange(1, len(r["singular_values"]) + 1),
            "singular_value": r["singular_values"],
            "threshold": r["threshold"],
            "null_median": np.median(r["null_singular_values"], axis=0),
            "above_threshold": r["above_threshold"],
        })
        run_settings["pa"] = dict(pa.settings, J=pa.J)
    if settings:
        run_settings["cca"] = dict(settings)
    out["settings"] = run_settings

    if data.get("K") is not None and data.get("X2_np") is not None:
        out["gene_weights"] = compute_gene_weights_all_components(
            K=data["K"],
            X2_np=data["X2_np"],
            W2_matrix=result["W2_matrix"],
            A_matrix=result["A_matrix"],
            valid_genes=data["valid_genes"],
            verbose=False,
        )

    pd.to_pickle(out, path)
    return path


def summarize_cca(
    result: dict,
    data: dict,
    top_n: int | None = None,
    w_threshold: float = 1e-4,
) -> dict:
    """Collect all interpretable CCA outputs into one named dictionary.

    Parameters
    ----------
    result : dict returned by fit_witten_nonnegative_cca.
    data   : dict returned by prepare_for_cca (needs valid_genes, KERNEL_NAMES).
    top_n  : number of top genes to return per component, ranked by W descending.
             If None (default), return all genes with W > w_threshold.
    w_threshold : minimum W to include a gene in top_genes (default 1e-4).

    Returns
    -------
    dict with keys:
        latent_component_expression : DataFrame (n_cells, J), latent component expression K @ A.
        gene_component_expression   : DataFrame (n_cells, J), gene component expression X @ W2.
        kernel_attr : DataFrame (n_kernels, J), kernel attribution per component
                      (columns sum to 1).
        gene_kernel : DataFrame (n_genes, n_kernels), gene-kernel association score.
        top_genes   : dict mapping component name -> DataFrame of top genes with
                      columns [gene, W, rank], sorted by W descending.
        diagnostics : cca_diagnostics DataFrame from the fit.
    """
    kernel_names = [str(k) for k in data["KERNEL_NAMES"]]
    valid_genes = list(data["valid_genes"])
    J = result["W2_matrix"].shape[1]
    components = [f"C_{j}" for j in range(J)]

    # latent component expression and gene component expression
    latent_component_expression = pd.DataFrame(
        result["U_score_matrix"],
        index=data["cell_names"],
        columns=components,
    )
    gene_component_expression = pd.DataFrame(
        result["V_score_matrix"],
        index=data["cell_names"],
        columns=components,
    )

    # kernel attribution and gene-kernel scores
    kernel_attr, gene_kernel = gene_kernel_scores(
        result["W2_matrix"],
        result["all_scores"],
        gene_names=valid_genes,
        kernel_names=kernel_names,
        component_names=components,
    )
    gene_kernel.index = pd.Index(valid_genes, name="gene")

    # top genes per component, ranked by W descending
    W2 = result["W2_matrix"]
    top_genes = {}
    for j, comp in enumerate(components):
        w_vec = W2[:, j]
        order = np.argsort(w_vec)[::-1]
        genes_sorted = [(valid_genes[i], float(w_vec[i])) for i in order
                        if float(w_vec[i]) > w_threshold]
        if top_n is not None:
            genes_sorted = genes_sorted[:top_n]
        top_genes[comp] = pd.DataFrame(
            [(g, w, rank + 1) for rank, (g, w) in enumerate(genes_sorted)],
            columns=["gene", "W", "rank"],
        )

    return dict(
        latent_component_expression=latent_component_expression,
        gene_component_expression=gene_component_expression,
        kernel_attr=kernel_attr,
        gene_kernel=gene_kernel,
        top_genes=top_genes,
        diagnostics=result["cca_diagnostics"],
    )
