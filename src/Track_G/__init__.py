"""Track-G: gene modules related to multiple cellular attributes (lineage tree, location, cell type, ...).

Modules
-------
parallel_analysis : choose the number of components J (permutation null on K^T X)
CCA               : Witten nonnegative sparse multi-kernel CCA, kernel concatenation (K^1/2), gene scores, saving
gene_filtering    : gene filter + per-gene standardization of the expression

>>> from Track_G import (filter_and_standardize_expression, build_kernel_matrix, run_parallel_analysis,
...                      fit_witten_nonnegative_cca, gene_kernel_scores)
"""
from .parallel_analysis import (NULLS, PAResult, run_parallel_analysis, plot_parallel_analysis, pa_entrywise,
                                pa_row_permute, pa_expression, global_entrywise_pa, cross_covariance_pa,
                                expression_pa)
from .CCA import (fit_witten_nonnegative_cca, nonnegative_soft_threshold, random_feasible_unit_vector,
                  witten_c_from_legacy_delta, witten_nonnegative_update, build_kernel_matrix,
                  compute_gene_weights_all_components, gene_kernel_scores, save_pa_cca_outputs)
from .gene_filtering import filter_genes, filter_and_standardize_expression

__all__ = [
    # Parallel analysis
    "NULLS", "PAResult", "run_parallel_analysis", "plot_parallel_analysis", "pa_entrywise",
    "pa_row_permute", "pa_expression", "global_entrywise_pa", "cross_covariance_pa", "expression_pa",
    # CCA
    "fit_witten_nonnegative_cca", "nonnegative_soft_threshold", "random_feasible_unit_vector",
    "witten_c_from_legacy_delta", "witten_nonnegative_update", "build_kernel_matrix",
    "compute_gene_weights_all_components", "gene_kernel_scores", "save_pa_cca_outputs",
    # Gene filtering
    "filter_genes", "filter_and_standardize_expression",
]
