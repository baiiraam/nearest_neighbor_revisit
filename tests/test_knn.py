"""
Tests for KNN implementation.
"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from src.knn import KNN


def test_knn_sanity_check():
    """Test that our KNN matches sklearn's implementation."""
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    split = n_samples // 2
    X_train, X_test = X[:split], X[split:]
    y_train, _y_test = y[:split], y[split:]

    # Our KNN
    knn_custom = KNN(k=5, metric="euclidean", task="classification")
    knn_custom.fit(X_train, y_train)
    y_pred_custom = knn_custom.predict(X_test)
    y_proba_custom = knn_custom.predict_proba(X_test)

    # sklearn KNN
    knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    y_proba_sklearn = knn_sklearn.predict_proba(X_test)

    # Assert predictions match
    np.testing.assert_array_equal(y_pred_custom, y_pred_sklearn)

    # Assert probabilities match within tolerance
    np.testing.assert_allclose(y_proba_custom, y_proba_sklearn, rtol=1e-9, atol=1e-9)

    print("All sanity checks passed!")


def test_distance_computation():
    """Test distance computation."""
    np.random.seed(42)

    X_train = np.random.randn(10, 3)
    X_test = np.random.randn(5, 3)

    knn = KNN(k=3, metric="euclidean")
    knn.fit(X_train, np.zeros(10))

    distances = knn._compute_distances(X_test)

    # Check shape
    assert distances.shape == (5, 10)

    # Check non-negative
    assert np.all(distances >= 0)

    # Check Euclidean property (distance to self should be approximately 0)
    knn_self = KNN(k=3, metric="euclidean")
    knn_self.fit(X_train, np.zeros(10))
    self_distances = knn_self._compute_distances(X_train)
    # Use a larger tolerance for floating point errors (1e-7 instead of 1e-10)
    np.testing.assert_array_almost_equal(np.diag(self_distances), 0, decimal=7)


def test_manhattan_distance():
    """Test Manhattan distance."""
    np.random.seed(42)

    X_train = np.random.randn(10, 3)
    X_test = np.random.randn(5, 3)

    knn = KNN(k=3, metric="manhattan")
    knn.fit(X_train, np.zeros(10))
    distances = knn._compute_distances(X_test)

    # Check shape
    assert distances.shape == (5, 10)

    # Check non-negative
    assert np.all(distances >= 0)


def test_predict_proba():
    """Test predict_proba outputs valid probabilities."""
    np.random.seed(42)

    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    knn = KNN(k=5, task="classification")
    knn.fit(X, y)

    proba = knn.predict_proba(X)

    # Check shape
    assert proba.shape == (50, 2)

    # Check probabilities sum to 1
    np.testing.assert_array_almost_equal(np.sum(proba, axis=1), 1)

    # Check probabilities are in [0, 1]
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_error_handling():
    """Test error handling for invalid inputs."""
    knn = KNN(k=5)

    # Test predict before fit
    X = np.random.randn(10, 3)
    with pytest.raises(RuntimeError):
        knn.predict(X)

    # Test fit with mismatched dimensions
    X_bad = np.random.randn(10, 3)
    y_bad = np.random.randn(9)
    with pytest.raises(ValueError):
        knn.fit(X_bad, y_bad)
