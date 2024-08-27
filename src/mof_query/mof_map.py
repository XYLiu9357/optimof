"""container.py
Implements a wrapper for scikit-learn KDTree for querying MOFs 
that are closest to satisfying some criteria. 

This container is not optimized for large-scale insertion and 
removal of content. Its sole purpose is to support efficient 
nearest neighbor query for processed MOFs.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree
import joblib


class MOFMap:

    def __init__(
        self,
        mof_df=None,
        dist_metric=euclidean_distances,
        project_path=".",
    ):
        self.import_file_path = os.path.join(project_path, "data", "mof-tree.pkl")
        self.export_filepath = os.path.join(project_path, "data", "mof-tree.pkl")
        self.dist_metric = dist_metric

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
        assert not any(mof_df.isna()), "NaN values detected"

        self.keys = mof_df.loc[:, feats]
        self.values = mof_df.loc[:, "name"]
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

    def nearest_neighbor_query(self, query):
        dist, ind = self.kdtree.query(query, k=1)
        exact_match = np.isclose(dist, 0)
        if exact_match:
            print("Exact match found for the given query")
        return self.values[ind.flatten()], exact_match

    def import_from_file(self, file_path=None):
        if file_path is None:
            file_path = self.import_file_path
        self.keys, self.values, self.kdtree = joblib.load(file_path)

    def export_to_file(self, file_path=None):
        if file_path is None:
            file_path = self.export_filepath
        joblib.dump((self.keys, self.values, self.kdtree), file_path)
