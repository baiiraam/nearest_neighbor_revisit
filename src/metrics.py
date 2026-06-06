"""
Classification metrics implemented from scratch.
"""

import numpy as np
from typing import Tuple


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    acc : float
        Accuracy score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute precision score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    positive_label : int, default=1
        The label of the positive class.

    Returns
    -------
    prec : float
        Precision score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Count true positives and false positives
    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_pred[i] == positive_label:
            if y_true[i] == positive_label:
                tp += 1
            else:
                fp += 1

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute recall score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    positive_label : int, default=1
        The label of the positive class.

    Returns
    -------
    rec : float
        Recall score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Count true positives and false negatives
    tp = 0
    fn = 0

    for i in range(len(y_true)):
        if y_true[i] == positive_label:
            if y_pred[i] == positive_label:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute F1 score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    positive_label : int, default=1
        The label of the positive class.

    Returns
    -------
    f1 : float
        F1 score.
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0 or 1).
    y_score : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    auc : float
        ROC AUC score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_score)}")

    # Get unique thresholds
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]  # descending

    # Add thresholds beyond extremes
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred_bin = (y_score >= threshold).astype(int)

        # Calculate TP, FP, FN, TN
        tp = np.sum((y_true == 1) & (y_pred_bin == 1))
        fp = np.sum((y_true == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true == 1) & (y_pred_bin == 0))
        tn = np.sum((y_true == 0) & (y_pred_bin == 0))

        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Convert to arrays and sort by FPR
    tpr_list = np.array(tpr_list)
    fpr_list = np.array(fpr_list)

    sort_idx = np.argsort(fpr_list)
    fpr_list = fpr_list[sort_idx]
    tpr_list = tpr_list[sort_idx]

    # Remove duplicate FPR points (keep highest TPR)
    unique_fpr, unique_indices = np.unique(fpr_list, return_index=True)
    unique_tpr = tpr_list[unique_indices]

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(unique_fpr)):
        auc += (
            (unique_fpr[i] - unique_fpr[i - 1])
            * (unique_tpr[i] + unique_tpr[i - 1])
            / 2.0
        )

    return auc


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    positive_label : int, default=1
        The label of the positive class.

    Returns
    -------
    tp, fp, fn, tn : tuple of ints
        True positives, false positives, false negatives, true negatives.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))

    return tp, fp, fn, tn
