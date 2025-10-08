"""Centralized path configuration for OptiMOF project."""

from pathlib import Path

# Project root (config is in src/config, so go up two levels)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
THERMAL_DATA_DIR = DATA_DIR / "thermal"
SOLVENT_DATA_DIR = DATA_DIR / "solvent"
WATER_DATA_DIR = DATA_DIR / "water_and_haz"

# Model directories
MODEL_DIR = PROJECT_ROOT / "model"
SCALER_DIR = MODEL_DIR / "scalers"

# Performance/output directories
PERFORMANCE_DIR = PROJECT_ROOT / "performance"
THERMAL_PERFORMANCE_DIR = PERFORMANCE_DIR / "thermal"
SOLVENT_PERFORMANCE_DIR = PERFORMANCE_DIR / "solvent"
WATER_RF_PERFORMANCE_DIR = PERFORMANCE_DIR / "water_rf"
WATER_BOOST_PERFORMANCE_DIR = PERFORMANCE_DIR / "water_boost"
