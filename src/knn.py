"""
k-Nearest Neighbors implementation from scratch.
"""

import numpy as np
from typing import Optional


class KNN:
    """
    k-Nearest Neighbors classifier/regressor.

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
    weights : str, default='uniform'
        Weighting scheme: 'uniform' or 'distance' (bonus).
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        weights: str = "uniform",
    ) -> None:
        self.k = k
        self.metric = metric.lower()
        self.q = q
        self.task = task.lower()
        self.weights = weights.lower()

        # Validate parameters
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.metric not in ["euclidean", "manhattan", "minkowski"]:
            raise ValueError(f"Unknown metric: {self.metric}")
        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Unknown task: {self.task}")
        if self.weights not in ["uniform", "distance"]:
            raise ValueError(f"Unknown weights: {self.weights}")
        if self.metric == "minkowski" and self.q <= 0:
            raise ValueError("q must be > 0 for Minkowski distance")

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.n_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Memorize the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNN
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

        if self.task == "classification":
            self.n_classes = len(np.unique(y))

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute distances from query points to all training points.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        distances : np.ndarray of shape (n_queries, n_train)
            Distance matrix.
        """
        # Using broadcasting to compute pairwise distances
        # |x - y|^2 = |x|^2 + |y|^2 - 2xy
        # Shape: (n_queries, n_train)

        if self.metric == "euclidean":
            # Euclidean distance: sqrt(sum((x - y)^2))
            # Using squared distance trick for efficiency
            X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n_queries, 1)
            train_sq = np.sum(self.X_train**2, axis=1)  # (n_train,)
            cross = np.dot(X, self.X_train.T)  # (n_queries, n_train)
            distances_sq = X_sq + train_sq - 2 * cross
            # Ensure non-negative due to numerical issues
            distances_sq = np.maximum(distances_sq, 0)
            distances = np.sqrt(distances_sq)

        elif self.metric == "manhattan":
            # Manhattan distance: sum(|x - y|)
            # Using broadcasting: (n_queries, 1, n_features) - (1, n_train, n_features)
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            distances = np.sum(diff, axis=2)

        elif self.metric == "minkowski":
            # Minkowski distance: (sum(|x - y|^q))^(1/q)
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            distances = np.sum(diff**self.q, axis=2) ** (1 / self.q)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return distances

    def _get_neighbor_indices(self, distances: np.ndarray) -> np.ndarray:
        """
        Get indices of k nearest neighbors for each query point.

        Parameters
        ----------
        distances : np.ndarray of shape (n_queries, n_train)
            Distance matrix.

        Returns
        -------
        neighbor_indices : np.ndarray of shape (n_queries, k)
            Indices of k nearest neighbors.
        """
        # Use argpartition for O(n) selection of k smallest
        # This is more efficient than full argsort for large n
        k = min(self.k, distances.shape[1])
        neighbor_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]

        # Sort by distance to ensure consistent order (useful for tie-breaking)
        rows = np.arange(distances.shape[0])[:, np.newaxis]
        neighbor_distances = distances[rows, neighbor_indices]
        sorted_idx = np.argsort(neighbor_distances, axis=1)
        neighbor_indices = neighbor_indices[rows, sorted_idx]

        return neighbor_indices

    def _get_weights(
        self, distances: np.ndarray, neighbor_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute weights for neighbors based on distance.

        Parameters
        ----------
        distances : np.ndarray of shape (n_queries, n_train)
            Distance matrix.
        neighbor_indices : np.ndarray of shape (n_queries, k)
            Indices of neighbors.

        Returns
        -------
        weights : np.ndarray of shape (n_queries, k)
            Weights for each neighbor.
        """
        if self.weights == "uniform":
            return np.ones_like(neighbor_indices, dtype=float)

        # Distance-based weights: 1/(d + epsilon)
        rows = np.arange(distances.shape[0])[:, np.newaxis]
        neighbor_distances = distances[rows, neighbor_indices]
        epsilon = 1e-8
        weights = 1.0 / (neighbor_distances + epsilon)

        # Normalize weights per query point
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels (classification) or values (regression).

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

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Must call fit() before predict()")

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.X_train.shape[1]}"
            )

        distances = self._compute_distances(X)
        neighbor_indices = self._get_neighbor_indices(distances)
        weights = self._get_weights(distances, neighbor_indices)

        if self.task == "classification":
            return self._predict_classification(neighbor_indices, weights)
        else:
            return self._predict_regression(neighbor_indices, weights)

    def _predict_classification(
        self, neighbor_indices: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels using weighted majority vote.
        """
        n_queries = neighbor_indices.shape[0]
        predictions = np.zeros(n_queries, dtype=self.y_train.dtype)

        for i in range(n_queries):
            neighbor_labels = self.y_train[neighbor_indices[i]]
            neighbor_weights = weights[i]

            # Weighted voting
            vote_counts = {}
            for label, weight in zip(neighbor_labels, neighbor_weights):
                vote_counts[label] = vote_counts.get(label, 0) + weight

            # Handle ties by choosing the label with smallest value (consistent with sklearn)
            # Actually sklearn chooses the label that appears first, but for binary we're fine
            predictions[i] = max(vote_counts, key=lambda x: (vote_counts[x], -x))  # type: ignore

        return predictions

    def _predict_regression(
        self, neighbor_indices: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Predict values using weighted average.
        """
        n_queries = neighbor_indices.shape[0]
        predictions = np.zeros(n_queries, dtype=float)

        for i in range(n_queries):
            neighbor_values = self.y_train[neighbor_indices[i]]
            neighbor_weights = weights[i]
            predictions[i] = np.sum(neighbor_values * neighbor_weights)

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

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        distances = self._compute_distances(X)
        neighbor_indices = self._get_neighbor_indices(distances)

        n_queries = neighbor_indices.shape[0]
        proba = np.zeros((n_queries, self.n_classes))

        for i in range(n_queries):
            neighbor_labels = self.y_train[neighbor_indices[i]]
            for j in range(self.n_classes):
                proba[i, j] = np.mean(neighbor_labels == j)

        return proba
