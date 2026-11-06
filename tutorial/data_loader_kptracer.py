def load_kptracer_data(min_expr_frac: float = 0.01, min_gene_std: float = 1e-6, frobenius_normalize: bool = False, use_sqrt: bool = False) -> Dict[str, Any]:
    """Load the KP-Tracer expression, kernels and cell metadata and build the CCA inputs.

    min_expr_frac : keep genes expressed (> 0) in more than this fraction of cells, e.g. 0.01 = 1 %
        (default), 0.005 = the ``*_filter_low_gene`` notebooks, 0 = no expression filter (all
        non-constant genes, as in the main PA notebooks).
    min_gene_std : minimum per-gene std (numpy ddof=0) to retain a gene. Default 1e-6 removes
        near-constant genes. Pass 0.0 to skip.
    frobenius_normalize : divide each kernel by its Frobenius norm before the concatenation
        (default False = standard concatenation).
    use_sqrt : replace each kernel with its symmetric square root before the concatenation
        (default False = standard concatenation).
    ``load_kptracer_data(0, True)`` reproduces the inputs of the main PA notebooks: the same genes and
    the same K; X2_np agrees up to float32 rounding (the notebooks standardize the float32 expression
    directly, here it is cast to float64 first).  The keys match the notebooks' data cell, so
    ``globals().update(load_kptracer_data(...))`` works.
    """
    # Load expression matrix and fill missing values with 0
    log_exp = pd.read_pickle(DATA_DIR / "3454_Lkb1_All_log_expression_all_genes.pkl").fillna(0)
    kernel_tables = {name: pd.read_pickle(DATA_DIR / KERNEL_FILES[name]) for name in KERNEL_NAMES.tolist()}
    cell_metadata = pd.read_pickle(DATA_DIR / "3454_Lkb1_All_cell_metadata_aligned.pkl")

    # Align all matrices to the phylogenetic kernel's cell order
    nodes_order = kernel_tables["Sigma_phylo"].index.astype(str).to_numpy()
    log_exp.index = log_exp.index.astype(str)
    cell_metadata.index = cell_metadata.index.astype(str)
    log_exp = log_exp.reindex(index=nodes_order.tolist())
    cell_metadata = cell_metadata.reindex(index=nodes_order.tolist())
    if log_exp.isna().any().any() or cell_metadata.isna().all(axis=1).any():
        raise ValueError("Some kernel cells are missing from the expression matrix or the metadata.")

    # Filter low-expression/near-constant genes and standardize
    expression = filter_and_standardize_expression(log_exp, min_expr_frac, min_gene_std)
    print(f"Total genes retained after filtering: {len(expression['valid_genes'])} "
          f"(std>{min_gene_std:.2g}, expressed in > {min_expr_frac:.1%} of cells)")

    # Build concatenated kernel matrix K for CCA
    kernels_raw = {name: table.loc[nodes_order, nodes_order].to_numpy(dtype=float)
                   for name, table in kernel_tables.items()}
    K, kernels_for_cca = build_kernel_matrix(kernels_raw, KERNEL_NAMES, frobenius_normalize, use_sqrt)
    n_cells, p_gene = expression["X2_np"].shape
    assert K.shape == (n_cells, len(KERNEL_NAMES) * n_cells)

    return dict(
        cell_names=nodes_order, kernels_raw=kernels_raw, kernels_for_cca=kernels_for_cca, K=K,
        cell_metadata=cell_metadata, n_cells=n_cells, p_gene=p_gene, KERNEL_NAMES=KERNEL_NAMES.copy(),
        min_expr_frac=float(min_expr_frac), 
        min_gene_std=float(min_gene_std),frobenius_normalize=bool(frobenius_normalize),
        use_sqrt=bool(use_sqrt),
        **expression,
    )
