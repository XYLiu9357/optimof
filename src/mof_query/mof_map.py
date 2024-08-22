"""container.py
Implements a wrapper for scikit-learn KDTree for querying MOFs 
that are closest to satisfying some criteria. 

This container is not optimized for large-scale insertion and 
removal of content. Its sole purpose is to support efficient 
nearest neighbor query for processed MOFs.
"""

import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KDTree
import joblib


class MOFMap:

    def __init__(self, dist_metric=euclidean_distances, project_path="."):
        self.keys = None
        self.values = None
        self.kdtree = None

        self.import_file_path = os.path.join(project_path, "data", "mof-tree.pkl")
        self.export_filepath = os.path.join(project_path, "data", "mof-tree.pkl")
        self.dist_metric = dist_metric

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

            # Rebuild the KDTree
            self.kdtree = KDTree(self.keys, metric=self.dist_metric)
        else:
            print(f"Warning: removal query for {value} has no matching target")

    def nearest_neighbor_query(self, query):
        dist, ind = self.kdtree.query(query, k=1)
        return self.values[ind.flatten()]

    def import_from_file(self, file_path):
        self.keys, self.values, self.kdtree = joblib.load(file_path)

    def export(self, file_path):
        joblib.dump((self.keys, self.values, self.kdtree), file_path)
