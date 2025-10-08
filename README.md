# OptiMOF

Metal-Organic Framework screening and stability prediction powered by machine learning.

## Overview

OptiMOF is a comprehensive machine learning pipeline for predicting the stability properties of metal-organic frameworks (MOFs) from crystallographic information files (CIFs). The project combines geometric feature extraction with multiple predictive models to assess:

- **Thermal Stability**: Predicts MOF breakdown temperature using artificial neural networks (PyTorch)
- **Solvent Removal Stability**: Binary classification of MOF stability upon solvent removal (PyTorch)
- **Water Stability**: Multi-class classification of MOF stability in aqueous environments (Random Forest/XGBoost)

The models are trained on experimental data from the CoRE MOF database and use geometric descriptors extracted via Zeo++ to make stability predictions on new MOF structures.

## Key Features

- **Automated Feature Extraction**: Extract geometric and topological features from CIF files using Zeo++
- **Multiple ML Models**: Pre-trained neural networks and ensemble models for different stability metrics
- **Scalable Processing**: MPI support for parallel feature extraction on large MOF datasets
- **Nearest Neighbor Search**: Find similar MOFs in the database for reference

## Prerequisites

### Environment Setup

Create and activate the conda environment from the provided specification:

```bash
conda env create -f environment.yml
conda activate mof
```

This will install all required dependencies including Python 3.12, PyTorch, scikit-learn, XGBoost, and other necessary packages.

### Optional: Parallel Computing

For large-scale feature extraction, **MPI (Message Passing Interface)** can be used to parallelize the processing of multiple CIF files across distributed systems or multi-core machines. Install an MPI implementation such as:

- **OpenMPI**: `conda install -c conda-forge openmpi mpi4py`
- **MPICH**: `conda install -c conda-forge mpich mpi4py`

With MPI installed, you can use `make mpi-extract-all` to extract features in parallel.

## Quick Start

```bash
# Format code (optional)
make format

# Extract features from a single CIF file
make extract-feature ARGS="--cif path/to/structure.cif"

# Tune hyperparameters (optional, uses Optuna)
make tune-all

# Train all models
make train-all

# Test model performance
make test-all
```

See `make help` for all available commands. See [tuning](doc/tuning.md) for more on hyperparameter tuning.

## Acknowledgements

CoRE 2019 MOF database created by Bobbitt et al. is used to train the model, and the development process is supported by personnel from [Snurr Research Group](https://zeolites.cqe.northwestern.edu/) @NorthwesternU. During the design & construction of this package, references were made to a deep learning pipeline developed by Nandy et al. and part of the source code was used in the feature extraction stage. The Zeo++ software package developed by Willems et al. is used to conduct geometry-based analysis on the CIF files, which is crucial to feature extraction.

## References

N. Scott Bobbitt, Shi, K., Bucior, B. J., Chen, H., Tracy-Amoroso, N., Li, Z., Sun, Y., Merlin, J. H., J. Ilja Siepmann, Siderius, D. W., & Snurr, R. Q. (2023). MOFX-DB: An Online Database of Computational Adsorption Data for Nanoporous Materials. Journal of Chemical and Engineering Data/Journal of Chemical & Engineering Data, 68(2), 483–498. https://doi.org/10.1021/acs.jced.2c00583

Nandy, A., Terrones, G., Arunachalam, N., Duan, C., Kastner, D. W., & Kulik, H. J. (2022). MOFSimplify, machine learning models with extracted stability data of three thousand metal–organic frameworks. Scientific Data, 9(1). https://doi.org/10.1038/s41597-022-01181-0

Willems, T. F., Rycroft, C. H., Kazi, M., Meza, J. C., & Haranczyk, M. (2012). Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials. Microporous and Mesoporous Materials, 149(1), 134–141. https://doi.org/10.1016/j.micromeso.2011.08.020‌
