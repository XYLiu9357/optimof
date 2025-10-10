"""mof_map.py
Implements a wrapper for scikit-learn KDTree for querying MOFs
that are closest to satisfying some criteria.

This container is not optimized for large-scale insertion and
removal of content. Its sole purpose is to support efficient
nearest neighbor query for processed MOFs.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


class MOFMap:
    def __init__(
        self,
        mof_df=None,
        dist_metric="euclidean",
        project_path=".",
        weights=None,
    ):
        """Initialize MOFMap.

        Args:
            mof_df: DataFrame with columns ['name', 'thermal', 'solvent', 'water']
            dist_metric: Distance metric for KDTree ('euclidean', 'manhattan', etc.)
            project_path: Project root path
            weights: Optional tuple of (thermal_weight, solvent_weight, water_weight)
                    for weighted distance calculation. If provided, features will be
                    multiplied by sqrt(weights) to achieve weighted Euclidean distance.
        """
        project_path = Path(project_path)
        self.import_file_path = project_path / "data" / "mof_map" / "mof-tree.pkl"
        self.export_filepath = project_path / "data" / "mof_map" / "mof-tree.pkl"
        self.dist_metric = dist_metric
        self.weights = weights

        if mof_df is None:
            self.keys = None
            self.values = None
            self.kdtree = None
            return

        # Checks data integrity
        feats = [
            "thermal",
            "solvent",
            "water",
        ]
        assert isinstance(mof_df, pd.DataFrame)
        assert "name" in mof_df.columns, f"MOFMap: missing labels"
        assert all(col in mof_df.columns for col in feats), f"MOFMap: missing features"
        assert not mof_df.isna().any().any(), "MOFMap: NaN values detected"

        self.keys = mof_df.loc[:, feats].copy()
        self.values = mof_df.loc[:, "name"]

        # Apply weights if provided (multiply by sqrt(weight) for weighted Euclidean)
        if weights is not None:
            assert len(weights) == 3, "weights must be a tuple of 3 values"
            self.keys['thermal'] *= np.sqrt(weights[0])
            self.keys['solvent'] *= np.sqrt(weights[1])
            self.keys['water'] *= np.sqrt(weights[2])

        self.kdtree = KDTree(self.keys, metric=self.dist_metric)

    def insert(self, keys, y):
        if self.keys is None:
            self.keys = keys
            self.values = y
        else:
            self.keys = np.vstack((self.keys, keys))
            self.values = np.hstack((self.values, y))

        self.kdtree = KDTree(self.keys, metric=self.dist_metric)

    def remove(self, value):
        indices_to_remove = np.where(self.values == value)[0]
        if len(indices_to_remove) > 0:
            self.keys = np.delete(self.keys, indices_to_remove, axis=0)
            self.values = np.delete(self.values, indices_to_remove, axis=0)
            self.kdtree = KDTree(self.keys, metric=self.dist_metric)
        else:
            print(f"Warning: removal query for {value} has no matching target")

    def nearest_neighbor_query(self, query: np.ndarray[np.float32]):
        dist, ind = self.kdtree.query(query, k=1)
        exact_match = np.isclose(dist, 0)
        if exact_match:
            print("Exact match found for the given query")
        return self.values[ind.flatten()].values, exact_match

    def import_from_file(self, file_path=None):
        if file_path is None:
            file_path = self.import_file_path
        self.keys, self.values, self.kdtree = joblib.load(file_path)

    def export_to_file(self, file_path=None):
        if file_path is None:
            file_path = self.export_filepath
        joblib.dump((self.keys, self.values, self.kdtree), file_path)
