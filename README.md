# OptiMOF

OptiMOF - Metal organic framework screening & optimization powered by deep learning.

## Overview

OptiMOF is an algorithm package for metal-organic framework (MOF) screening & optimization, powered by deep learning. This tool provides a comprehensive workflow tha includes mining and analyzing MOF literature, extracting critical features, predicting MOF stability based on crystallographic information files (CIFs).

## Supported features

1. Literature Mining: Utilizes NLP to process MOF literature and extract stability data.
2. Feature Extraction: Automated feature extraction from [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File).
3. Machine Learning Models: Several artificial neural network (ANN) models were trained on geometric representations of MOFs found, and can be used to make predictions on MOFs' thermal and solvent stability.
4. Prediction with Uncertainty: Provides stability predictions on new MOFs with quantified uncertainty.

## Usage

### Prerequisites

To use and develop with OptiMOF, ensure that your environment meets the following requirements.

- Basic usage: `python < 3.8` due to compatibility issues. See [here](environment.yml) for a list of dependencies.
- Feature extraction: `C++14` compiler and [GNU Make](https://www.gnu.org/software/make/) are required for compiling zeo++.
- Run automated scripts: [Taskfile](https://taskfile.dev/installation/) version 3 is required to run the automation scripts.

Make sure these tools and versions are properly installed and configured in your development environment to fully utilize the features and tests provided by this project.

## Acknowledgements

CoRE 2019 MOF database created by Bobbitt et al. is used to train the model, and the development process is supported by personnel from Snurr's Research Group @NorthwesternU. During the design & construction of this package, references were made to a deep learning pipeline developed by Nandy et al. and part of the source code was used in the feature extraction stage. The Zeo++ software package developed by Willems et al. is used to conduct geometry-based analysis on the CIF files, which is crucial to feature extraction.

## References

N. Scott Bobbitt, Shi, K., Bucior, B. J., Chen, H., Tracy-Amoroso, N., Li, Z., Sun, Y., Merlin, J. H., J. Ilja Siepmann, Siderius, D. W., & Snurr, R. Q. (2023). MOFX-DB: An Online Database of Computational Adsorption Data for Nanoporous Materials. Journal of Chemical and Engineering Data/Journal of Chemical & Engineering Data, 68(2), 483–498. https://doi.org/10.1021/acs.jced.2c00583

Nandy, A., Terrones, G., Arunachalam, N., Duan, C., Kastner, D. W., & Kulik, H. J. (2022). MOFSimplify, machine learning models with extracted stability data of three thousand metal–organic frameworks. Scientific Data, 9(1). https://doi.org/10.1038/s41597-022-01181-0

Willems, T. F., Rycroft, C. H., Kazi, M., Meza, J. C., & Haranczyk, M. (2012). Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials. Microporous and Mesoporous Materials, 149(1), 134–141. https://doi.org/10.1016/j.micromeso.2011.08.020‌
