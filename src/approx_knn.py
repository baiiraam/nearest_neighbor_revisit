"""
Approximate Nearest Neighbors using sklearn's BallTree.
"""

import numpy as np
from sklearn.neighbors import BallTree
from typing import Optional


class ApproxKNN:
    """
    Approximate k-Nearest Neighbors classifier using BallTree.
    Provides the same API as KNN but with faster predictions on large datasets.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use.
    metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'minkowski'.
    q : float, default=2.0
        Parameter for Minkowski distance (q=1 -> Manhattan, q=2 -> Euclidean).
    task : str, default='classification'
        Task type: 'classification' or 'regression'.
    leaf_size : int, default=40
        Leaf size for BallTree (affects speed/accuracy trade-off).
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        leaf_size: int = 40,
    ) -> None:
        self.k = k
        self.metric = metric.lower()
        self.q = q
        self.task = task.lower()
        self.leaf_size = leaf_size

        # Validate parameters
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.metric not in ["euclidean", "manhattan", "minkowski"]:
            raise ValueError(f"Unknown metric: {self.metric}")
        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Unknown task: {self.task}")
        if self.metric == "minkowski" and self.q <= 0:
            raise ValueError("q must be > 0 for Minkowski distance")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.tree: Optional[BallTree] = None
        self.n_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ApproxKNN":
        """
        Build BallTree index on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ApproxKNN
            Returns the instance itself.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got {len(X)} and {len(y)}"
            )

        self.X_train = X.copy()
        self.y_train = y.copy()

        # Map metric names for BallTree
        if self.metric == "euclidean":
            sklearn_metric = "euclidean"
            metric_kwargs = {}
        elif self.metric == "manhattan":
            sklearn_metric = "manhattan"
            metric_kwargs = {}
        elif self.metric == "minkowski":
            sklearn_metric = "minkowski"
            metric_kwargs = {"p": self.q}
        else:
            sklearn_metric = "euclidean"
            metric_kwargs = {}

        # Build BallTree index
        self.tree = BallTree(
            X, metric=sklearn_metric, leaf_size=self.leaf_size, **metric_kwargs
        )

        if self.task == "classification":
            self.n_classes = len(np.unique(y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using approximate nearest neighbors.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        y_pred : np.ndarray of shape (n_queries,)
            Predicted target values.
        """
        X = np.asarray(X)

        if self.tree is None or self.y_train is None:
            raise RuntimeError("Must call fit() before predict()")

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.X_train.shape[1]}"
            )

        # Query BallTree for k nearest neighbors
        distances, indices = self.tree.query(X, k=min(self.k, len(self.y_train)))

        if self.task == "classification":
            return self._predict_classification(indices)
        else:
            return self._predict_regression(indices)

    def _predict_classification(self, indices: np.ndarray) -> np.ndarray:
        """Predict class labels using majority vote."""
        n_queries = indices.shape[0]
        # k_actual = indices.shape[1]  # Remove - not used
        predictions = np.zeros(n_queries, dtype=self.y_train.dtype)

        for i in range(n_queries):
            neighbor_labels = self.y_train[indices[i]]
            # Majority vote (most frequent label)
            # Use bincount for efficiency
            counts = np.bincount(neighbor_labels.astype(int))
            predictions[i] = np.argmax(counts)

        return predictions

    def _predict_regression(self, indices: np.ndarray) -> np.ndarray:
        """Predict values using average."""
        n_queries = indices.shape[0]
        predictions = np.zeros(n_queries, dtype=float)

        for i in range(n_queries):
            neighbor_values = self.y_train[indices[i]]
            predictions[i] = np.mean(neighbor_values)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-class probabilities (classification only).

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        proba : np.ndarray of shape (n_queries, n_classes)
            Probability estimates for each class.
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks"
            )

        if self.n_classes is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        X = np.asarray(X)

        if self.tree is None or self.y_train is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        distances, indices = self.tree.query(X, k=min(self.k, len(self.y_train)))

        n_queries = indices.shape[0]
        proba = np.zeros((n_queries, self.n_classes))

        for i in range(n_queries):
            neighbor_labels = self.y_train[indices[i]]
            for j in range(self.n_classes):
                proba[i, j] = np.mean(neighbor_labels == j)

        return proba
