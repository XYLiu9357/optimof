"""predict.py
Runs all models over a collection of CIF files.
1. Call extract_features to extract feature dataframe
2. Preprocess feature dataframe: formatting & feature engineering
3. Feed data through each model
4. Output predicted labels in dataframe
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.model_features.feature_extraction import extract_features


def extract_from_file(filename: str) -> pd.DataFrame:
    pass


def df_preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    pass


def pred_thermal(input_df: pd.DataFrame) -> np.ndarray[np.float32]:
    pass


def pred_solvent(input_df: pd.DataFrame) -> np.ndarray[bool]:
    pass


def pred_water(input_df: pd.DataFrame) -> np.ndarray[bool]:
    pass


def predict_from_file(filename: str) -> pd.DataFrame:
    pass
