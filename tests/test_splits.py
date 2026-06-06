"""
Tests for splitting utilities.
"""

import numpy as np
from src.splits import stratified_split, stratified_kfold, Fold


def test_stratified_split_shapes():
    """Test that stratified_split returns correct shapes."""
    np.random.seed(42)

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42
    )

    # Account for rounding differences due to stratification
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == 100
    assert X_train.shape[0] >= 55  # Approximately 60
    assert X_val.shape[0] >= 15  # Approximately 20
    assert X_test.shape[0] >= 15  # Approximately 20
    assert X_train.shape[1] == 5
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_stratified_kfold_returns_correct_number_of_folds():
    """Test that stratified_kfold returns K folds."""
    np.random.seed(42)

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    folds = list(stratified_kfold(X, y, K=5, seed=42))

    assert len(folds) == 5

    for fold in folds:
        assert isinstance(fold, Fold)
        assert hasattr(fold, "train_idx")
        assert hasattr(fold, "val_idx")


def test_stratified_kfold_preserves_class_proportions():
    """Test that each fold preserves class proportions."""
    np.random.seed(42)

    X = np.random.randn(200, 5)
    # Create imbalanced data
    y = np.array([0] * 150 + [1] * 50)
    np.mean(y) * 100

    fold_class_props = []

    for fold in stratified_kfold(X, y, K=5, seed=42):
        y_val = y[fold.val_idx]
        pos_pct = np.mean(y_val) * 100
        fold_class_props.append(pos_pct)

    # All folds should have similar class proportions
    assert np.std(fold_class_props) < 2


def test_stratified_kfold_all_samples_covered():
    """Test that all samples appear in exactly one validation fold."""
    np.random.seed(42)

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    all_val_indices = []

    for fold in stratified_kfold(X, y, K=5, seed=42):
        all_val_indices.extend(fold.val_idx)

    # Check that we have exactly 100 unique indices
    assert len(set(all_val_indices)) == 100


def test_fold_class():
    """Test Fold class initialization."""
    train_idx = np.array([1, 2, 3])
    val_idx = np.array([4, 5, 6])

    fold = Fold(train_idx, val_idx)

    assert np.array_equal(fold.train_idx, train_idx)
    assert np.array_equal(fold.val_idx, val_idx)


def test_stratified_split_with_seed_reproducibility():
    """Test that seed ensures reproducible splits."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1 = stratified_split(
        X, y, seed=123
    )

    X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2 = stratified_split(
        X, y, seed=123
    )

    np.testing.assert_array_equal(X_train_1, X_train_2)
    np.testing.assert_array_equal(y_train_1, y_train_2)
