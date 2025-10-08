"""Training constants for OptiMOF project."""

# Train/test split parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 0

# Label column names
THERMAL_LABEL = "thermal"
SOLVENT_LABEL = "solvent"
WATER_LABEL = "water"

# Water stability label normalization
# Original labels are {1, 2, 3, 4}, normalize to {0, 1, 2, 3}
WATER_LABEL_OFFSET = 1
