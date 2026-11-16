# TrackG

We developed `Track-G` to identify gene modules related to a flexible number and types of cellular attributes, such as position in the developmental lineage, cell type and spatial region. `Track-G` represents every cellular attribute by a cell-cell similarity matrix (kernel), so that categorical and continuous attributes and lineage trees are handled in the same way. Based on a multi-kernel non-negative canonical correlation analysis (CCA) framework, `Track-G` jointly infers gene modules and the relatedness of every module to every attribute, which delineates the attribute underlying each module through a soft assignment. The number of modules is chosen by parallel analysis. Applied to lineage-tracing data of mouse lung tumors (KP-Tracer), `Track-G` separated lineage-related from location-related gene modules and made fewer misassignments of genes to attributes than methods that consider one attribute at a time.

<br>

<img width="969" src="overview.png" />

<br>

## Installation

```bash
git clone https://github.com/CM-Compbio-Group/TrackG.git
cd TrackG
conda env create -f environment.yml
conda activate trackg
python -m ipykernel install --user --name trackg --display-name trackg
```

Alternatively, you can follow these steps:

```bash
git clone https://github.com/CM-Compbio-Group/TrackG.git
cd TrackG
conda create -n trackg python=3.10
conda activate trackg
pip install numpy scipy pandas matplotlib seaborn anndata biopython networkx threadpoolctl
pip install jupyter jupyterlab ipykernel
python -m ipykernel install --user --name trackg --display-name trackg
```

`Track-G` is a Python package in `src/Track_G/` and needs no further installation: add `src/` to the Python path, as the tutorial notebooks do. Tested with Python 3.10, numpy 2.2, scipy 1.15, pandas 2.3, matplotlib 3.10, anndata 0.11 and biopython 1.85.

## How to Use

* Please refer to the tutorials, run from the `tutorial/` folder (outputs are written to `tutorial/results/`):
  * `tutorial/simulation.ipynb`: `Track-G` on simulated data with a lineage tree and a categorical cell type (about 2 min).
  * `tutorial/kptracer.ipynb`: `Track-G` on the KP-Tracer tumor family 3454_Lkb1_All, with the lineage tree and the tumor location as attributes (about 10 min). The raw data are included in `tutorial/data/kptracer/`.

* Basic usage:

```python
import sys
sys.path.insert(0, "src")
from Track_G import (kernel_from_newick, kernel_from_categorical, kernel_from_continuous, prepare_for_cca,
                     run_parallel_analysis, fit_witten_nonnegative_cca, summarize_cca)

# log_exp: cells x genes DataFrame of log1p-normalized expression; kernel rows follow its cell order
kernels = {"lineage": kernel_from_newick("tree.nwk", log_exp.index),          # lineage tree
           "cell type": kernel_from_categorical(cell_metadata["cell_type"]),  # categorical attribute
           "space": kernel_from_continuous(cell_metadata[["x", "y"]])}        # continuous attribute
data = prepare_for_cca(log_exp, kernels)            # filtered + standardized genes, K = [K_1^1/2 | ... | K_M^1/2]
pa = run_parallel_analysis(data["X2_np"], data["K"])
J = pa.J                                            # number of components inferred by parallel analysis
fit = fit_witten_nonnegative_cca(K=data["K"], X2_np=data["X2_np"], J=J)
res = summarize_cca(fit, data)                      # dict of labelled DataFrames
```
`res["kernel_attr"]` is the relative relatedness $p_{kj}$ of every component to every attribute, `res["gene_kernel"]` the relatedness $s_{gk}$ of every gene to every attribute, `res["latent_component_expression"]` the latent component expression $KA$ of every cell, and `res["top_genes_by_kernel"]` the gene ranking of every attribute.

## Repository structure

```
src/Track_G/                the Track-G package
    kernel_construction.py  kernels from a Newick tree, a categorical vector or continuous values
    preprocess.py           prepare_for_cca: expression + kernels -> everything needed for PA and CCA
    parallel_analysis.py    number of components J (permutation null on K^T X)
    CCA.py                  nonnegative sparse multi-kernel CCA, kernel concatenation (K^1/2), summarize_cca, saving
    gene_filtering.py       gene filter + per-gene standardization
utils/                      wrappers of other methods (ALPINE, CoGAPS, Hotspot); not used by the tutorials
tutorial/
    data/kptracer/          raw KP-Tracer data of tumor family 3454_Lkb1_All (counts + lineage tree)
    kptracer/               negative controls and figures of the KP-Tracer analysis
    simulation/             simulated data and evaluation / figures of the simulation
    kptracer.ipynb          Track-G on KP-Tracer
    simulation.ipynb        Track-G on simulated data
```
