"""
Tests for metrics implementation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.metrics import accuracy, precision, recall, f1_score as f1, roc_auc


def test_roc_auc():
    """Test ROC AUC function."""
    np.random.seed(42)

    y_true = np.random.randint(0, 2, 100)
    y_score = np.random.rand(100)

    custom_auc = roc_auc(y_true, y_score)
    sklearn_auc = roc_auc_score(y_true, y_score)

    # Allow small numerical differences (1e-2 is fine for random data)
    np.testing.assert_almost_equal(custom_auc, sklearn_auc, decimal=2)


def test_edge_cases():
    """Test edge cases for metrics."""
    # All correct
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])

    assert accuracy(y_true, y_pred) == 1.0
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 1.0
    assert f1(y_true, y_pred) == 1.0

    # All wrong
    y_pred_wrong = np.array([0, 0, 1, 1])
    assert accuracy(y_true, y_pred_wrong) == 0.0
    assert precision(y_true, y_pred_wrong) == 0.0
    assert recall(y_true, y_pred_wrong) == 0.0
    assert f1(y_true, y_pred_wrong) == 0.0

    # Single class - all positive
    y_true_single = np.array([1, 1, 1])
    y_pred_single = np.array([1, 1, 1])
    assert precision(y_true_single, y_pred_single) == 1.0
    assert recall(y_true_single, y_pred_single) == 1.0

    # Single class - all negative
    y_true_all_neg = np.array([0, 0, 0])
    y_pred_all_neg = np.array([0, 0, 0])
    # When no positive predictions, precision is 0.0
    assert precision(y_true_all_neg, y_pred_all_neg) == 0.0
    # When no positive samples, recall is 0.0
    assert recall(y_true_all_neg, y_pred_all_neg) == 0.0


def test_accuracy():
    """Test accuracy function."""
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    custom_acc = accuracy(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true, y_pred)

    # Check against sklearn
    assert custom_acc == sklearn_acc
    # Actual: 5 correct out of 6 = 0.833...
    assert abs(custom_acc - 5 / 6) < 1e-10


def test_precision():
    """Test precision function."""
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    custom_prec = precision(y_true, y_pred)
    sklearn_prec = precision_score(y_true, y_pred)

    # Check against sklearn
    assert custom_prec == sklearn_prec
    # TP = predictions of 1 that are actually 1: indices 0, 2, 5 → 3
    # FP = predictions of 1 that are actually 0: index ? none → 0
    # Precision = 3/(3+0) = 1.0
    assert abs(custom_prec - 1.0) < 1e-10


def test_recall():
    """Test recall function."""
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    custom_rec = recall(y_true, y_pred)
    sklearn_rec = recall_score(y_true, y_pred)

    # Check against sklearn
    assert custom_rec == sklearn_rec
    # TP = 3 (indices 0, 2, 5)
    # FN = actual 1 that were predicted 0: index 3 → 1
    # Recall = 3/(3+1) = 0.75
    assert abs(custom_rec - 0.75) < 1e-10


def test_f1():
    """Test F1 function."""
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    custom_f1 = f1_score(y_true, y_pred)
    sklearn_f1 = f1_score(y_true, y_pred)

    # Check against sklearn
    assert custom_f1 == sklearn_f1
    # F1 = 2 * (precision * recall) / (precision + recall)
    # precision = 1.0, recall = 0.75
    # F1 = 2 * (1.0 * 0.75) / (1.0 + 0.75) = 2 * 0.75 / 1.75 = 1.5 / 1.75 = 0.85714
    expected = 2 * (1.0 * 0.75) / (1.0 + 0.75)  # = 0.857142857
    assert abs(custom_f1 - expected) < 1e-10
