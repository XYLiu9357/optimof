# Hyperparameter Tuning System

This document describes the new hyperparameter tuning system for OptiMOF.

## Overview

The training framework uses Optuna for automatic hyperparameter tuning to determine optimal neural network architectures. The tuning process:
1. Searches over architecture parameters (layer sizes, dropout, activation functions, etc.)
2. Searches over training parameters (learning rate, batch size, optimizer)
3. Saves the best configuration to `model/structure/` as a JSON file
4. The saved structure can be used to train models with the optimal hyperparameters

## Architecture Variants

The system supports three MLP (Multi-Layer Perceptron) architecture variants:

### 1. Simple
- Basic fully-connected layers
- Activation function + Dropout between layers
- Simplest and fastest to train

### 2. BatchNorm
- Adds batch normalization after each linear layer
- Improves training stability and convergence
- Slightly slower than simple architecture

### 3. Residual
- Adds skip connections between layers
- Uses projection layers when dimensions don't match
- Best for very deep networks
- Helps prevent gradient vanishing

## Search Spaces

### Architecture Parameters
- **arch_type**: `simple`, `batchnorm`, `residual`
- **n_layers**: 2-7 hidden layers
- **layer_sizes**: 64, 128, 256, 512, 1024, 2048 neurons
- **dropout**: 0.0-0.5
- **activation**: `relu`, `leaky_relu`, `elu`

### Training Parameters
- **learning_rate**: 1e-5 to 1e-2 (log scale)
- **batch_size**: 64, 128, 256, 512
- **optimizer**: `adam`, `adamw`

## Usage

### 1. Run Hyperparameter Tuning

#### Using Makefile (recommended)
```bash
# Tune thermal model with 50 trials (default)
make tune-thermal

# Tune with custom number of trials
make tune-thermal TRIALS=100

# Tune solvent model
make tune-solvent TRIALS=50

# Tune water model (sklearn)
make tune-water TRIALS=30

# Tune all models
make tune-all
```

#### Using Python directly
```bash
# Thermal stability model
python -m src.model_training.tune_thermal --n-trials 50 --study-name thermal_pytorch

# Solvent removal stability model
python -m src.model_training.tune_solvent --n-trials 50 --study-name solvent_pytorch

# Water stability model (Random Forest)
python -m src.model_training.tune_water --model-type rf --n-trials 30

# Water stability model (XGBoost)
python -m src.model_training.tune_water --model-type xgboost --n-trials 30
```

### 2. Train Model with Tuned Structure

After tuning completes, a structure file is saved to `model/structure/`. Use this to train:

#### Using Makefile
```bash
# Train with default structure file
make train-thermal

# Train with specific structure file
make train-thermal STRUCTURE=model/structure/thermal_pytorch_structure.json
make train-solvent STRUCTURE=model/structure/solvent_pytorch_structure.json
```

#### Using Python directly
```bash
# Train thermal model
python -m src.model_training.thermal_model --structure model/structure/thermal_pytorch_structure.json

# Train solvent model
python -m src.model_training.solvent_model --structure model/structure/solvent_pytorch_structure.json
```

## Structure File Format

Structure files are saved as JSON in `model/structure/`:

```json
{
  "model_type": "pytorch_regression",
  "architecture": {
    "input_size": 190,
    "hidden_layers": [512, 256, 128],
    "output_size": 1,
    "dropout_prob": 0.3,
    "arch_type": "residual",
    "activation": "leaky_relu"
  },
  "training": {
    "learning_rate": 0.0001,
    "batch_size": 256,
    "num_epochs": 1000,
    "patience": 100,
    "optimizer": "adamw"
  },
  "tuning_metadata": {
    "best_trial": 42,
    "best_value": 0.0234,
    "n_trials": 50,
    "study_name": "thermal_pytorch",
    "date": "2025-10-08T12:34:56",
    "params": { ... }
  }
}
```

## Resumable Studies

Optuna studies are stored in SQLite databases in `model/structure/optuna_studies/`. This means:
- **Resumable**: If tuning is interrupted, re-run the command to continue where you left off
- **Inspectable**: Use Optuna's dashboard or API to analyze trial history
- **Reproducible**: All trials are logged with their hyperparameters and results

## Early Stopping (Pruning)

The system uses Median Pruning to stop unpromising trials early:
- Saves computation time by not fully training bad configurations
- Starts pruning after 5 trials complete (n_startup_trials)
- Evaluates pruning after 10 warmup steps (n_warmup_steps)
- Prunes if current trial is worse than median of complete trials

## Best Practices

### 1. Start with Fewer Trials
```bash
# Quick exploration (10-20 trials)
make tune-thermal TRIALS=10

# Production tuning (50-100 trials)
make tune-thermal TRIALS=100
```

### 2. Monitor Progress
The tuning process shows a progress bar and prints results after each trial:
```
Trial 1 finished with value: 0.0542
Trial 2 finished with value: 0.0389
Trial 3 pruned.
...
```

### 3. Check Study Statistics
After tuning completes, review the statistics:
```
Study Statistics:
  Number of finished trials: 50
  Number of pruned trials: 12
  Number of complete trials: 38

Best trial:
  Value: 0.0234
  Params:
    arch_type: residual
    n_layers: 5
    layer_0_size: 512
    ...
```

### 4. Compare Architectures
Run multiple studies with different names to compare:
```bash
# Try different study approaches
python -m src.model_training.tune_thermal --study-name thermal_v1 --n-trials 50
python -m src.model_training.tune_thermal --study-name thermal_v2 --n-trials 100
```

## GPU Usage

- Tuning runs sequentially (n_jobs=1) for GPU safety
- Each trial uses the GPU if available
- Monitor GPU memory with `nvidia-smi` during tuning

## Troubleshooting

### Trial Failures
If trials fail with errors, they return `inf` (PyTorch) or `0.0` (sklearn) and continue. Check error messages for issues like:
- Out of memory (reduce batch_size range)
- Invalid architecture (check layer sizes)

### Long Training Times
- Reduce `num_epochs` during tuning (default: 50 for tuning, 1000 for final training)
- Use smaller dataset for tuning
- Increase pruning aggressiveness

### No Improvement
If all trials perform similarly:
- Expand search space ranges
- Check data quality and preprocessing
- Try different model types

## Files Created

```
model/structure/
├── optuna_studies/
│   ├── thermal_pytorch.db          # Study database
│   ├── solvent_pytorch.db
│   └── water_rf.db
├── thermal_pytorch_structure.json   # Best architecture
├── solvent_pytorch_structure.json
└── water_rf_structure.json
```

## Migration from Old System

The old system used fixed hyperparameter files:
- `model/thermal_hyperparams.json`
- `model/solvent_hyperparams.json`

These are no longer used. The new workflow is:
1. Run hyperparameter tuning → creates structure file
2. Train with structure file → creates model
3. Test model → generates performance metrics

## Example Workflow

```bash
# Complete workflow for thermal model
make tune-thermal TRIALS=50           # Tune hyperparameters
make train-thermal                    # Train with best structure
make test-thermal                     # Test the model

# Or do all models
make tune-all
make train-all
make test-all
```
