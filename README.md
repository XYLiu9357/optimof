# OptiMOF

OptiMOF is an algorithm package for Metal organic framework screening & optimization, powered by deep learning. This tool provides a comprehensive workflow for mining and analyzing MOF literature to extract critical stability data.

## Overview

OptiMOF employs a natural language processing (NLP)-based procedure to mine the existing MOF literature. This method focuses on extracting data related to structurally characterized MOFs, specifically their solvent removal and thermal stabilities.

## Supported features

1. Literature Mining: Utilizes NLP to process MOF literature and extract stability data.
2. Feature Extraction: Automated feature extraction from [Crystallographic Information File](https://en.wikipedia.org/wiki/Crystallographic_Information_File).
3. Machine Learning Models: Artificial neural network models trained on extracted data using latent representations for stability prediction of new MOFs.
4. Prediction with Uncertainty: Provides stability predictions on new MOFs with quantified uncertainty.

## Usage

### Prerequisites

To use and develop with `RedBlackTree`, ensure that your environment meets the following requirements.

- Basic usage: `python < 3.8` due to library compatibility issues
- Feature extraction: a compiler that supports `C++14` and [GNU Make](https://www.gnu.org/software/make/) are required for compiling zeo++.
- Run automated scripts: [Taskfile](https://taskfile.dev/installation/) version 3 is required to run the automation scripts.

Make sure these tools and versions are properly installed and configured in your development environment to fully utilize the features and tests provided by this project.
