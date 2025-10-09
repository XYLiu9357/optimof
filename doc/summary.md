# OptiMOF: Machine Learning Pipeline for MOF Stability Prediction

## Methodology Summary

This document describes the complete data pipeline for predicting stability properties of metal-organic frameworks (MOFs) using machine learning, from feature extraction to model predictions.

---

## 1. Feature Extraction from CIF Files

### 1.1 Overview
The pipeline extracts structural and chemical features from Crystallographic Information Files (CIF) using two complementary approaches:

1. **Geometric descriptors** via Zeo++ software
2. **Revised Autocorrelation (RAC) descriptors** for chemical environments

### 1.2 Zeo++ Geometric Feature Extraction

The Zeo++ software (`src/model_features/feature_extraction.py:92-110`) is invoked to compute geometric properties using a probe radius of **1.86 Ångströms** (representing typical molecular dimensions):

**Pore Size Descriptors:**
- **Di**: Maximum included sphere diameter
- **Df**: Maximum free sphere diameter
- **Dif**: Maximum included sphere along free sphere path

**Surface Area Metrics:**
- **VSA**: Volumetric surface area (m²/cm³)
- **GSA**: Gravimetric surface area (m²/g)

**Pore Volume Metrics:**
- **VPOV/GPOV**: Volumetric/gravimetric pore volume
- **POAV/PONAV**: Pore accessible/non-accessible volume (Ų)
- **GPOAV/GPONAV**: Gravimetric versions (cm³/g)
- **POAV_vol_frac/PONAV_vol_frac**: Volume fractions

**Cell Properties:**
- **cell_v**: Unit cell volume (Ų)

All Zeo++ calculations use 10,000 Monte Carlo samples for accurate surface area and volume estimates.

### 1.3 RAC Chemical Descriptors

The RAC_finder module (`src/model_features/RAC_finder.py`) extracts chemical environment descriptors characterizing:

- **Linker chemistry**: Organic ligand properties
- **Metal node chemistry**: Local coordination environments
- **Secondary building units (SBUs)**: Structural motifs

These descriptors capture atomic properties (electronegativity, coordination number, identity) weighted by graph distance from reference atoms, generating ~165 RAC features per MOF.

### 1.4 Feature Aggregation

For MOFs with multiple linker/metal types, numeric RAC descriptors are **averaged** to create a single feature vector (`src/model_features/feature_extraction.py:250-259`). The final merged dataset contains **190 features** after:

1. Combining geometric and RAC descriptors
2. Dropping empty/constant columns (38 D_func, D_lc, D_mc variants with no variance)
3. Model-specific feature selection

---

## 2. Data Preprocessing and Normalization

### 2.1 Data Cleaning (`src/model_features/preprocess.py`)

Raw CSV files from experimental measurements are processed as follows:

**Thermal Stability Data:**
- Source: `thermal_all_data.csv` with MOF breakdown temperatures
- Drops metadata columns (filename, CoRE_name)
- Filters rows with missing temperature labels
- Output: 190 features + 1 target (thermal breakdown temperature in K)

**Solvent Removal Stability Data:**
- Source: `solvent_all_data.csv` with binary stability labels
- Original labels: {-1 (unstable), +1 (stable)}
- Normalized to: {0 (unstable), 1 (stable)} for neural network training
- Output: 190 features + 1 binary target

**Water Stability Data:**
- Source: `water_and_haz_all_data.csv` with multi-class labels
- Original labels: {1, 2, 3, 4} indicating stability levels
- Normalized to: {0, 1, 2, 3} for 0-indexed classification
- Drops 11 additional features not useful for water prediction
- Output: 179 features + 1 multi-class target

### 2.2 Train-Test Split

Data is split using `sklearn.model_selection.train_test_split` with:
- **Test size**: 20% of data
- **Random state**: 0 (for reproducibility)
- **Stratification**: None (continuous target for thermal; class imbalance considerations for classification)

Implementation in `src/model_training/base/data_processor.py:48-70`.

### 2.3 Feature Normalization

**StandardScaler** from scikit-learn applies Z-score normalization:

```
x_scaled = (x - μ) / σ
```

Where μ is the mean and σ is the standard deviation computed on **training data only** (`data_processor.py:72-79`). The same scaler is applied to test and prediction data to prevent data leakage.

Scalers are saved as pickle files:
- `model/scalers/thermal_scaler.pkl`
- `model/scalers/solvent_scaler.pkl`
- (Water models use sklearn's built-in scaling in GridSearchCV)

---

## 3. Model Architectures

### 3.1 Thermal Stability Model (Regression)

**Task**: Predict MOF breakdown temperature (continuous value)

**Architecture**: Flexible Multi-Layer Perceptron (FlexibleMLP) with:
- **Input**: 190 features
- **Hidden layers**: [128, 512, 2048, 256, 1024] neurons
- **Output**: 1 neuron (temperature prediction)
- **Activation**: ReLU
- **Regularization**: Batch normalization + 12.4% dropout
- **Loss function**: Mean Squared Error (MSE)

**Training Configuration**:
- Optimizer: Adam (lr = 0.00199)
- Batch size: 256
- Max epochs: 1000
- Early stopping: patience = 100 epochs

Architecture determined via Optuna hyperparameter tuning (860 trials) and saved in `model/structure/saved_structures/thermal_pytorch_structure.json`.

### 3.2 Solvent Removal Stability Model (Binary Classification)

**Task**: Predict if MOF maintains structural integrity upon solvent removal

**Architecture**: FlexibleMLP with:
- **Input**: 190 features
- **Hidden layers**: [128, 256, 1024, 2048, 128, 256] neurons
- **Output**: 1 neuron with sigmoid activation (probability)
- **Activation**: ReLU
- **Regularization**: Batch normalization + 2.5% dropout
- **Loss function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

**Training Configuration**:
- Optimizer: AdamW (lr = 7.64×10⁻⁵)
- Batch size: 512
- Max epochs: 1000
- Early stopping: patience = 100 epochs

Architecture from 767 Optuna trials, saved in `model/structure/saved_structures/solvent_pytorch_structure.json`.

### 3.3 Water Stability Model (Multi-class Classification)

**Task**: Classify MOF water stability into 4 categories (0-3, representing increasing stability)

**Architecture**: Random Forest Classifier
- **Input**: 179 features (11 features dropped vs. other models)
- **Output**: 4-class probability distribution
- **Hyperparameters** (from GridSearchCV):
  - n_estimators: tuned over [100, 150, 200, 250, 300]
  - max_depth: tuned over [None, 10, 20]
  - min_samples_split: tuned over [2, 5, 10]
  - min_samples_leaf: tuned over [1, 2, 4]
  - bootstrap: tuned over [True, False]

Random Forest was selected over XGBoost after comparative evaluation. Implementation in `src/model_training/water_stability_model.py`.

---

## 4. Model Training Process

### 4.1 PyTorch Models (Thermal & Solvent)

**Training Loop** (`src/model_training/base/base_pytorch_pipeline.py`):

1. **Data preparation**: Load cleaned data, split train/test, fit scaler on training data
2. **Model initialization**: Build FlexibleMLP with tuned architecture
3. **Training**:
   - Forward pass through network
   - Compute loss (MSE for regression, BCE for classification)
   - Backpropagation via Adam/AdamW optimizer
   - Update weights
4. **Validation**: Monitor validation loss each epoch
5. **Early stopping**: Stop if no improvement for 100 consecutive epochs
6. **Checkpoint**: Save best model based on validation performance

**Device**: Automatically uses CUDA GPU if available, otherwise CPU

### 4.2 Scikit-learn Models (Water)

**Training Pipeline** (`src/model_training/base/base_sklearn_pipeline.py`):

1. **Data preparation**: Split features/labels, create train/test sets
2. **Model initialization**: Create Random Forest classifier
3. **Grid search**: 5-fold cross-validation over hyperparameter grid
4. **Best model selection**: Choose parameters maximizing cross-validation accuracy
5. **Final training**: Retrain on full training set with best parameters
6. **Save**: Pickle the trained model

---

## 5. Model Evaluation and Performance

### 5.1 Thermal Stability Model (Regression)

**Test Set Performance**:
- **R² score**: 0.4099
- **RMSE**: 67.29 K
- **MAE**: 46.65 K
- **MSE**: 4528.41 K²

**Interpretation**: The model explains ~41% of variance in breakdown temperatures. The RMSE of 67.29 K indicates predictions are typically within ±67 K of actual values. Given that MOF breakdown temperatures span 150-600 K, this represents useful predictive capability for screening applications.

**Diagnostic Plots** (`performance/thermal/`):

1. **Actual vs. Predicted** (`actual_vs_predicted.png`): Shows positive correlation along the diagonal with some scatter. The model performs well in the 250-500 K range where most data concentrates, with higher uncertainty at extremes.

2. **Residuals Plot** (`residuals.png`): Residuals are randomly distributed around zero with no obvious heteroscedasticity, indicating unbiased predictions across the temperature range.

3. **Residual Distribution** (`residual_distribution.png`): Approximately normal distribution centered at zero, confirming model assumptions are reasonable.

4. **QQ Plot** (`qq_plot.png`): Quantile-quantile plot shows residuals closely follow a normal distribution except in the extreme tails, where some outliers with large prediction errors exist.

### 5.2 Solvent Removal Stability Model (Binary Classification)

**Test Set Performance**:
- **Accuracy**: 76.61%
- **Precision**: 81.45%
- **Recall**: 81.45%
- **F1 Score**: 81.45%
- **AUC-ROC**: 0.8351

**Interpretation**: The model correctly classifies ~77% of MOFs, with balanced precision and recall indicating no bias toward either class. The AUC-ROC of 0.835 demonstrates strong discriminative ability.

**Diagnostic Plots** (`performance/solvent/`):

1. **Confusion Matrix** (`confusion_matrix.png`):
   - True Negatives (unstable correctly predicted): 110
   - False Positives (unstable predicted as stable): 51
   - False Negatives (stable predicted as unstable): 51
   - True Positives (stable correctly predicted): 224

   The model shows symmetric error rates, slightly favoring correct identification of stable MOFs.

2. **ROC Curve** (`roc_curve.png`): The curve rises steeply toward the top-left corner (AUC = 0.8351), indicating the model achieves high true positive rates while maintaining low false positive rates across various threshold values.

### 5.3 Water Stability Model (Multi-class Classification)

**Test Set Performance**:
- **Overall Accuracy**: 67.12%
- **Macro-averaged F1**: 0.55
- **Weighted F1**: 0.65
- **ROC AUC**: 0.8277 (one-vs-rest)

**Per-Class Performance**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Unstable) | 0.75 | 0.35 | 0.48 | 17 |
| 1 (Low stability) | 0.66 | 0.65 | 0.65 | 68 |
| 2 (Moderate stability) | 0.68 | 0.82 | 0.74 | 112 |
| 3 (High stability) | 0.56 | 0.23 | 0.32 | 22 |

**Interpretation**: The model performs best on Class 2 (moderate stability), which has the most training examples. Performance degrades for rare classes (0 and 3) due to class imbalance. The weighted F1 of 0.65 accounts for this imbalance.

**Diagnostic Plots** (`performance/water_rf/`):

1. **Confusion Matrix** (`confusion_matrix.png`): Shows the model has strong performance on Class 2 (92 correct predictions out of 112), with most errors being adjacent-class confusions (e.g., Class 1 predicted as Class 2). This indicates the model captures the ordinal nature of stability levels.

2. **ROC Curve** (`roc_curve.png`): Multi-class ROC using one-vs-rest strategy shows:
   - Class 0: AUC = 0.85 (best discrimination despite low recall)
   - Class 1: AUC = 0.84
   - Class 2: AUC = 0.79
   - Class 3: AUC = 0.84

   All classes exceed 0.79 AUC, demonstrating the model provides useful probability estimates for ranking MOFs by water stability.

---

## 6. Prediction Workflow

### 6.1 Making Predictions on New MOFs

The prediction pipeline (`src/utils/predict.py`) processes new CIF files:

1. **Feature Extraction**: Extract geometric and RAC features from input CIF
2. **Feature Alignment**: Ensure extracted features match training feature set (190 columns)
3. **Scaling**: Apply saved StandardScaler transformations
4. **Prediction**:
   - **Thermal**: Forward pass through neural network → temperature value
   - **Solvent**: Forward pass → logits → sigmoid → probability ∈ [0,1]
   - **Water**: Random Forest prediction → 4-class probabilities

### 6.2 Model Inference Details

**Thermal Model**:
```python
temperature = thermal_model(scaled_features)  # Direct regression output
```

**Solvent Model**:
```python
logits = solvent_model(scaled_features)
probability = sigmoid(logits)  # P(stable)
prediction = 1 if probability > 0.5 else 0
```

**Water Model**:
```python
probabilities = water_rf_model.predict_proba(features)  # [P(class_0), P(class_1), P(class_2), P(class_3)]
predicted_class = argmax(probabilities)
```

### 6.3 Command-Line Interface

Users can make predictions via:
```bash
python -m src --cif path/to/structure.cif
```

This executes the complete pipeline: extraction → scaling → prediction for all three stability metrics.

---

## 7. Hyperparameter Tuning

All models underwent extensive hyperparameter optimization using Optuna (neural networks) or GridSearchCV (Random Forest).

**Thermal Model Tuning**:
- 860 trials exploring architecture depth (1-8 layers), width (32-2048 neurons), dropout (0-0.5), learning rate (10⁻⁵ to 10⁻²), and batch size
- Best validation RMSE: 13.66 K achieved at trial 212

**Solvent Model Tuning**:
- 767 trials with similar search space
- Best validation BCE loss: 0.00231 achieved at trial 692

**Water Model Tuning**:
- Grid search over 5 hyperparameters with 5-fold cross-validation
- Random Forest outperformed XGBoost in preliminary comparisons

Tuned structures are stored in `model/structure/saved_structures/` and loaded at training time.

---

## 8. Data Sources and Training Set Sizes

All models are trained on the **CoRE MOF 2019 database** compiled by Bobbitt et al., containing experimental stability measurements.

**Dataset Statistics**:
- **Thermal**: ~3,850 MOFs with measured breakdown temperatures (after filtering missing values)
- **Solvent**: ~2,180 MOFs with stability labels
- **Water**: ~1,095 MOFs with water stability classifications

The 80-20 train-test split ensures independent evaluation while maintaining sufficient training data for complex neural network architectures.

---

## 9. Software Dependencies

**Core Libraries**:
- PyTorch 2.x (neural network training)
- scikit-learn 1.x (preprocessing, Random Forest, evaluation metrics)
- XGBoost (alternative water model)
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn (visualization)
- Optuna (hyperparameter optimization)

**External Tools**:
- Zeo++ v0.3 (C++ executable for geometric analysis)
- openbabel 2.4.1 (required for RAC extraction, Python < 3.8 compatibility)

**Environment**: Python 3.12 with conda environment specified in `environment.yml`

---

## 10. Model Limitations and Future Directions

**Current Limitations**:
1. **Thermal model R²**: 0.41 indicates substantial unexplained variance, likely due to breakdown temperature dependence on factors beyond geometry (e.g., bond strengths, defects)
2. **Class imbalance**: Water stability model struggles with rare classes (0 and 3)
3. **Geometric features only**: RAC descriptors provide limited electronic structure information

**Potential Improvements**:
1. Incorporate quantum mechanical descriptors (DFT-derived charges, band gaps)
2. Use graph neural networks to preserve MOF topology
3. Ensemble methods combining multiple model types
4. Active learning to target underrepresented regions
5. Transfer learning from related materials databases

---

## 11. Reproducibility

All random processes use fixed seeds:
- Train-test split: `random_state=0`
- PyTorch: Seed managed in training configuration
- Scikit-learn: `random_state=0` in model initialization

Models are version-controlled, and trained model files are saved as pickle objects, ensuring exact reproduction of results.

---

## References

This methodology implements and extends techniques from:

1. **Bobbitt et al. (2023)**: CoRE MOF database used for training data
2. **Nandy et al. (2022)**: RAC descriptor extraction methodology adapted from MOFSimplify
3. **Willems et al. (2012)**: Zeo++ geometric analysis algorithms

Full citations available in `README.md`.

---
