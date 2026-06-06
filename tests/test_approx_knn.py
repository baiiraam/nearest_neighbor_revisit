"""
Tests for ApproxKNN implementation.
"""

import numpy as np
import pytest
from src.knn import KNN
from src.approx_knn import ApproxKNN


def test_approx_knn_vs_exact():
    """Test that ApproxKNN approximates exact KNN reasonably well."""
    np.random.seed(42)

    # Generate small dataset for testing
    n_samples = 500
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    split = n_samples // 2
    X_train, X_test = X[:split], X[split:]
    y_train, _y_test = y[:split], y[split:]

    # Exact KNN
    knn_exact = KNN(k=5, metric="euclidean", task="classification")
    knn_exact.fit(X_train, y_train)
    y_pred_exact = knn_exact.predict(X_test)

    # Approx KNN
    knn_approx = ApproxKNN(k=5, metric="euclidean", task="classification", leaf_size=40)
    knn_approx.fit(X_train, y_train)
    y_pred_approx = knn_approx.predict(X_test)

    # Should be close (not necessarily identical)
    accuracy_diff = np.mean(y_pred_exact != y_pred_approx)
    print(f"Prediction difference: {accuracy_diff * 100:.2f}%")
    assert accuracy_diff < 0.15  # Less than 15% difference


def test_approx_knn_predict_proba():
    """Test predict_proba outputs valid probabilities."""
    np.random.seed(42)

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)

    knn = ApproxKNN(k=5, task="classification")
    knn.fit(X, y)

    proba = knn.predict_proba(X)

    assert proba.shape == (100, 2)
    np.testing.assert_array_almost_equal(np.sum(proba, axis=1), 1)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_approx_knn_regression():
    """Test regression mode."""
    np.random.seed(42)

    X = np.random.randn(100, 4)
    y = np.random.randn(100)  # Continuous values

    knn = ApproxKNN(k=5, task="regression")
    knn.fit(X, y)
    y_pred = knn.predict(X)

    assert y_pred.shape == (100,)
    assert isinstance(y_pred[0], float)


def test_approx_knn_error_handling():
    """Test error handling for invalid inputs."""
    knn = ApproxKNN(k=5)

    # Test predict before fit
    X = np.random.randn(10, 3)
    with pytest.raises(RuntimeError):
        knn.predict(X)

    # Test fit with mismatched dimensions
    X_bad = np.random.randn(10, 3)
    y_bad = np.random.randn(9)
    with pytest.raises(ValueError):
        knn.fit(X_bad, y_bad)
