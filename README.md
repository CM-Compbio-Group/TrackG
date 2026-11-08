# TrackG
Identify gene modules related to a flexible number and types of cellular attributes

## Repository layout
```
src/Track_G/              the Track-G package
    parallel_analysis.py  number of components J (permutation null on K^T X)
    CCA.py                nonnegative sparse multi-kernel CCA, kernel concatenation (K^1/2), gene scores, saving
    gene_filtering.py     gene filter + per-gene standardization
utils/                    wrappers of other methods (ALPINE, CoGAPS, Hotspot); not used by the tutorial
tutorial/
    data/kptracer/        raw KP-Tracer data of tumor family 3454_Lkb1_All (counts + lineage tree)
    kptracer/             data loader (expression, kernels) and evaluation / figures of the KP-Tracer analysis
    simulation/           simulated data and evaluation / figures of the simulation
    kptracer.ipynb        Track-G on KP-Tracer
    simulation.ipynb      Track-G on simulated data
```

## Usage
```python
import sys
sys.path.insert(0, "src")
from Track_G import (filter_and_standardize_expression, build_kernel_matrix, run_parallel_analysis,
                     fit_witten_nonnegative_cca, gene_kernel_scores)

expr = filter_and_standardize_expression(log_exp)              # cells x genes log expression
K, _ = build_kernel_matrix(kernels, kernel_names)               # K = [K_1^1/2 | ... | K_M^1/2]
J = run_parallel_analysis(expr["X2_np"], K, nulls=("entrywise",)).J
fit = fit_witten_nonnegative_cca(K=K, X2_np=expr["X2_np"], KERNEL_NAMES=kernel_names, J=J)
p, s = gene_kernel_scores(fit["W2_matrix"], fit["all_scores"], expr["valid_genes"], kernel_names)
```
`p` is the relative relatedness of every component to every attribute, `s` the score of every gene for every attribute.

## Tutorial
Run the notebooks from the `tutorial/` folder; outputs are written to `tutorial/results/`.
Requirements: numpy, scipy, pandas, matplotlib, seaborn, anndata, biopython, networkx, threadpoolctl.
