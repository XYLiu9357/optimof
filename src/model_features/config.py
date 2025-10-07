"""config.py
Centralized configuration for feature extraction and data processing.
Single source of truth for all column names, drop lists, and feature definitions.
"""

from typing import List

# ============================================================================
# LABEL COLUMNS
# ============================================================================

LABEL_COLUMNS = ["thermal", "solvent", "water"]
IDENTIFIER_COLUMN = "name"
ALL_LABEL_COLUMNS = [IDENTIFIER_COLUMN] + LABEL_COLUMNS

# ============================================================================
# CSV-SPECIFIC METADATA COLUMNS TO DROP
# ============================================================================

THERMAL_DROP_COLS = [
    "filename",
    "0",
    "CoRE_name",
    "name",
]

SOLVENT_DROP_COLS = [
    "Unnamed: 0",
    "doi",
    "filename",
    "0",
    "CoRE_name",
    "name",
]

WATER_DROP_COLS = [
    "acid_label",
    "base_label",
    "boiling_label",
    "data_set",
]

# ============================================================================
# COLUMN RENAMING MAPPINGS
# ============================================================================

THERMAL_RENAME_MAP = {
    "refcode": "name",
    "T": "thermal",
}

SOLVENT_RENAME_MAP = {
    "refcode": "name",
    "flag": "solvent",
}

WATER_RENAME_MAP = {
    "MOF_name": "name",
    "water_label": "water",
}

# ============================================================================
# EMPTY/CONSTANT FEATURE COLUMNS (to drop from merged data)
# ============================================================================
# These columns are all NaN or have no variance across the dataset

MERGED_DROP_COLS = [
    # D_func features
    "D_func-I-0-all",
    "D_func-I-1-all",
    "D_func-I-2-all",
    "D_func-I-3-all",
    "D_func-S-0-all",
    "D_func-T-0-all",
    "D_func-Z-0-all",
    "D_func-alpha-0-all",
    "D_func-chi-0-all",
    # D_lc features
    "D_lc-T-0-all",
    "D_lc-Z-0-all",
    "D_lc-alpha-0-all",
    "D_lc-chi-0-all",
    # D_mc features
    "D_mc-I-0-all",
    "D_mc-I-1-all",
    "D_mc-I-2-all",
    "D_mc-I-3-all",
    "D_mc-S-0-all",
    "D_mc-T-0-all",
    "D_mc-chi-0-all",
]

# ============================================================================
# MODEL-SPECIFIC COLUMN DROPS
# ============================================================================

# Additional columns to drop for water stability model only
# (These features don't improve water model performance)
WATER_MODEL_DROP_COLS = [
    "D_lc-I-0-all",
    "D_lc-I-1-all",
    "D_lc-I-2-all",
    "D_lc-I-3-all",
    "D_lc-S-0-all",
    "D_mc-Z-0-all",
    "POAV",
    "PONAV",
    "cell_v",
    "lc-I-0-all",
    "mc-I-0-all",
]

# ============================================================================
# GEOMETRIC FEATURES (from Zeo++)
# ============================================================================

GEOMETRIC_FEATURES = [
    # Pore diameters
    "Di",  # Maximum included sphere diameter
    "Df",  # Maximum free sphere diameter
    "Dif",  # Maximum included sphere along free sphere path
    # Surface areas
    "VSA",  # Volumetric surface area
    "GSA",  # Gravimetric surface area
    # Pore volumes
    "VPOV",  # Volumetric pore volume
    "GPOV",  # Gravimetric pore volume
    "POAV",  # Pore-accessible volume
    "PONAV",  # Pore non-accessible volume
    "GPOAV",  # Gravimetric pore-accessible volume
    "GPONAV",  # Gravimetric pore non-accessible volume
    # Volume fractions
    "POAV_vol_frac",  # Pore-accessible volume fraction
    "PONAV_vol_frac",  # Pore non-accessible volume fraction
    # Cell properties
    "cell_v",  # Unit cell volume
]

# ============================================================================
# ZEO++ PARAMETERS
# ============================================================================

ZEO_PROBE_RADIUS = 1.86  # Angstroms (typical for molecular simulations)
ZEO_NUM_SAMPLES = 10000  # Number of Monte Carlo samples for surface area/volume

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_feature_columns(all_columns: List[str]) -> List[str]:
    """Extract feature column names by removing labels and identifier.

    Args:
        all_columns: All column names from a DataFrame

    Returns:
        List of feature column names only
    """
    return [col for col in all_columns if col not in ALL_LABEL_COLUMNS]


def get_expected_feature_count() -> int:
    """Get the expected number of features after extraction.

    Returns:
        Expected number of feature columns (~180+)
    """
    # This is approximate - actual count depends on RAC extraction
    # Geometric features + RAC features (lc, mc, func, sbu, linker)
    return len(GEOMETRIC_FEATURES) + 165  # Approximate RAC feature count


def validate_dataframe_columns(df, required_columns: List[str]) -> bool:
    """Check if DataFrame has all required columns.

    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present

    Returns:
        True if all required columns present, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return False
    return True
