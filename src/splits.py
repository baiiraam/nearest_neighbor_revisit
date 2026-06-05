"""
Train/validation/test splitting and cross-validation utilities.
"""

import numpy as np
from typing import Tuple, Iterator


class Fold:
    """Container for cross-validation fold indices."""

    def __init__(self, train_idx: np.ndarray, val_idx: np.ndarray):
        self.train_idx = train_idx
        self.val_idx = val_idx


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified train/validation/test split preserving class proportions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target labels.
    train_frac : float, default=0.6
        Fraction for training set.
    val_frac : float, default=0.2
        Fraction for validation set.
    test_frac : float, default=0.2
        Fraction for test set.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple of np.ndarray
        Split data.
    """
    np.random.seed(seed)

    X = np.asarray(X)
    y = np.asarray(y)

    n = len(y)
    int(n * train_frac)
    int(n * val_frac)

    # Get unique classes
    classes = np.unique(y)

    train_indices = []
    val_indices = []
    test_indices = []

    # Stratify by class
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_cls = len(cls_indices)
        n_train_cls = int(n_cls * train_frac)
        n_val_cls = int(n_cls * val_frac)

        train_indices.extend(cls_indices[:n_train_cls])
        val_indices.extend(cls_indices[n_train_cls : n_train_cls + n_val_cls])
        test_indices.extend(cls_indices[n_train_cls + n_val_cls :])

    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Create arrays
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_kfold(
    X: np.ndarray, y: np.ndarray, K: int = 5, seed: int = 42
) -> Iterator[Fold]:
    """
    Generate stratified K-fold cross-validation splits.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target labels.
    K : int, default=5
        Number of folds.
    seed : int, default=42
        Random seed for reproducibility.

    Yields
    ------
    fold : Fold
        Object containing train and validation indices for each fold.
    """
    np.random.seed(seed)

    X = np.asarray(X)
    y = np.asarray(y)

    len(y)
    classes = np.unique(y)

    # For each class, create K folds
    class_indices = {cls: list(np.where(y == cls)[0]) for cls in classes}

    # Shuffle indices within each class
    for cls in classes:
        np.random.shuffle(class_indices[cls])

    # Create K folds by distributing samples from each class
    fold_sizes = []
    for cls in classes:
        n_cls = len(class_indices[cls])
        # Distribute as evenly as possible
        base_size = n_cls // K
        remainder = n_cls % K
        sizes = [base_size + 1] * remainder + [base_size] * (K - remainder)
        fold_sizes.append(sizes)

    # Build folds
    folds = [[] for _ in range(K)]
    for cls_idx, cls in enumerate(classes):
        indices = class_indices[cls]
        start = 0
        for fold_idx in range(K):
            size = fold_sizes[cls_idx][fold_idx]
            folds[fold_idx].extend(indices[start : start + size])
            start += size

    # For each fold, use it as validation, others as training
    for i in range(K):
        val_idx = np.array(folds[i])
        train_idx = np.array([idx for j in range(K) if j != i for idx in folds[j]])
        yield Fold(train_idx, val_idx)
