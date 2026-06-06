"""
Run all experiments for the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional

# Import custom modules
from knn import KNN
from metrics import accuracy, precision, recall, f1_score, roc_auc
from splits import stratified_split, stratified_kfold


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data."""
    X = np.load("X_full.npy")
    y = np.load("y_full.npy")
    return X, y


def run_hyperparameter_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: List[int],
) -> Dict:
    """
    Run hyperparameter sweep over k values.
    """
    results = {
        "k": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for k in k_values:
        knn = KNN(k=k, metric="euclidean", task="classification")
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_val)
        y_proba = knn.predict_proba(X_val)[:, 1]  # Probability of class 1

        results["k"].append(k)
        results["accuracy"].append(accuracy(y_val, y_pred))
        results["precision"].append(precision(y_val, y_pred))
        results["recall"].append(recall(y_val, y_pred))
        results["f1"].append(f1_score(y_val, y_pred))
        results["auc"].append(roc_auc(y_val, y_proba))

    return results


def plot_hyperparameter_results(
    results: Dict, save_path: str = "figures/hyperparameter_sweep.png"
):
    """Plot hyperparameter sweep results."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].plot(
            results["k"],
            results[metric],
            "o-",
            color="#0B3D91",
            linewidth=2,
            markersize=8,
        )
        axes[i].set_xlabel("k", fontsize=11)
        axes[i].set_ylabel(title, fontsize=11)
        axes[i].set_title(f"{title} vs k", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xscale("log")
        axes[i].set_xticks(results["k"])
        axes[i].set_xticklabels(results["k"], rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def cv_experiment(
    X: np.ndarray, y: np.ndarray, k_values: List[int], K: int = 5
) -> Dict:
    """
    Run cross-validation experiment.
    """
    results = {"k": [], "f1_mean": [], "f1_std": []}

    for k in k_values:
        f1_scores = []

        for fold in stratified_kfold(X, y, K=K):
            X_train_fold = X[fold.train_idx]
            y_train_fold = y[fold.train_idx]
            X_val_fold = X[fold.val_idx]
            y_val_fold = y[fold.val_idx]

            knn = KNN(k=k, metric="euclidean", task="classification")
            knn.fit(X_train_fold, y_train_fold)

            y_pred = knn.predict(X_val_fold)
            f1_scores.append(f1_score(y_val_fold, y_pred))

        results["k"].append(k)
        results["f1_mean"].append(np.mean(f1_scores))
        results["f1_std"].append(np.std(f1_scores))

    return results


def plot_cv_results(results: Dict, save_path: str = "figures/cv_f1_vs_k.png"):
    """Plot cross-validation results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results["k"],
        results["f1_mean"],
        "o-",
        color="#0B3D91",
        linewidth=2,
        markersize=8,
        label="Mean F1",
    )
    ax.fill_between(
        results["k"],
        np.array(results["f1_mean"]) - np.array(results["f1_std"]),
        np.array(results["f1_mean"]) + np.array(results["f1_std"]),
        alpha=0.3,
        color="#0B3D91",
        label="±1 std",
    )

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Cross-Validation: F1 Score vs k", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(results["k"])
    ax.set_xticklabels(results["k"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def computational_benchmark(
    X: np.ndarray, y: np.ndarray, train_sizes: Optional[List[int]] = None, test_size: int = 100
) -> Dict:
    """
    Benchmark prediction time as function of training set size.
    """
    # Set default train sizes based on dataset size
    if train_sizes is None:
        max_train = len(X) - test_size
        if max_train >= 10000:
            train_sizes = [100, 500, 1000, 5000, 10000]
        else:
            # Use smaller sizes for smaller datasets
            step = max_train // 5
            train_sizes = [step, step * 2, step * 3, step * 4, max_train]
            train_sizes = [max(100, s) for s in train_sizes if s >= 100]

    results = {"N": [], "time": []}

    # Take a fixed test set
    np.random.seed(42)

    # Ensure test_size doesn't exceed dataset size
    if test_size > len(X):
        test_size = max(1, len(X) // 10)

    test_indices = np.random.choice(len(X), test_size, replace=False)
    X_test = X[test_indices]

    # Get available indices for training (excluding test set)
    available_indices = [i for i in range(len(X)) if i not in test_indices]
    max_available = len(available_indices)

    print(
        f"Computational benchmark: {max_available} training samples available, test size={test_size}"
    )

    for N in train_sizes:
        # Skip if N is larger than available
        if N > max_available:
            print(f"Skipping N={N} (only {max_available} samples available)")
            continue

        # Sample training data
        train_indices = np.random.choice(available_indices, N, replace=False)
        X_train = X[train_indices]
        y_train = y[train_indices]

        knn = KNN(k=5, metric="euclidean", task="classification")
        knn.fit(X_train, y_train)

        # Time prediction
        start = time.time()
        _ = knn.predict(X_test)
        elapsed = time.time() - start

        results["N"].append(N)
        results["time"].append(elapsed)
        print(f"  N={N}: {elapsed * 1000:.2f} ms")

    return results


def plot_benchmark(
    results: Dict, X: np.ndarray, save_path: str = "figures/benchmark.png"
):
    """Plot computational benchmark results."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        results["N"], results["time"], "o-", color="#0B3D91", linewidth=2, markersize=8
    )

    # Fit line for scaling exponent
    log_N = np.log(results["N"])
    log_time = np.log(results["time"])
    slope, intercept = np.polyfit(log_N, log_time, 1)

    ax.loglog(
        results["N"],
        np.exp(intercept) * np.array(results["N"]) ** slope,
        "--",
        color="red",
        label=f"Theoretical O(N^{slope:.2f})",
    )

    ax.set_xlabel("Training Set Size (N)", fontsize=12)
    ax.set_ylabel("Prediction Time (seconds)", fontsize=12)
    ax.set_title("KNN Prediction Time Scaling", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Empirical scaling exponent: {slope:.2f}")
    print("Theoretical exponent for O(Np): 1.00")
    print(f"Feature dimension p = {X.shape[1]}")


def majority_baseline(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Majority class baseline predictor.
    """
    majority_class = np.bincount(y_train.astype(int)).argmax()
    return np.full(len(y_val), majority_class)


def main():
    """Run all experiments."""
    print("Loading data...")
    X, y = load_data()
    print(f"Feature dimension p = {X.shape[1]}")  # Print here instead

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, seed=42)

    print(
        f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    # Hyperparameter sweep
    print("\nRunning hyperparameter sweep...")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    sweep_results = run_hyperparameter_sweep(X_train, y_train, X_val, y_val, k_values)
    plot_hyperparameter_results(sweep_results)

    best_k_idx = np.argmax(sweep_results["f1"])
    best_k = sweep_results["k"][best_k_idx]
    print(f"Best k by validation F1: {best_k}")

    # Baseline comparison
    print("\nComparing with baselines...")
    knn = KNN(k=best_k, metric="euclidean", task="classification")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    y_proba = knn.predict_proba(X_val)[:, 1]

    majority_pred = majority_baseline(y_train, y_val)

    print(f"\nYour KNN (k={best_k}):")
    print(f"  Accuracy: {accuracy(y_val, y_pred):.4f}")
    print(f"  Precision: {precision(y_val, y_pred):.4f}")
    print(f"  Recall: {recall(y_val, y_pred):.4f}")
    print(f"  F1: {f1_score(y_val, y_pred):.4f}")
    print(f"  AUC: {roc_auc(y_val, y_proba):.4f}")

    print("\nMajority Baseline:")
    print(f"  Accuracy: {accuracy(y_val, majority_pred):.4f}")
    print(f"  Precision: {precision(y_val, majority_pred):.4f}")
    print(f"  Recall: {recall(y_val, majority_pred):.4f}")
    print(f"  F1: {f1_score(y_val, majority_pred):.4f}")

    # Cross-validation
    print("\nRunning cross-validation...")
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])
    cv_results = cv_experiment(X_cv, y_cv, k_values)
    plot_cv_results(cv_results)

    cv_best_k = k_values[np.argmax(cv_results["f1_mean"])]
    print(f"Best k by CV: {cv_best_k}")

    # Final test evaluation
    print("\nFinal test evaluation...")
    knn_final = KNN(k=best_k, metric="euclidean", task="classification")
    knn_final.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_test_pred = knn_final.predict(X_test)
    y_test_proba = knn_final.predict_proba(X_test)[:, 1]

    print(f"\nTest Set Results (k={best_k}):")
    print(f"  Accuracy: {accuracy(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_test_pred):.4f}")
    print(f"  Recall: {recall(y_test, y_test_pred):.4f}")
    print(f"  F1: {f1_score(y_test, y_test_pred):.4f}")
    print(f"  AUC: {roc_auc(y_test, y_test_proba):.4f}")

    # Computational benchmark
    print("\nRunning computational benchmark...")
    benchmark_results = computational_benchmark(X_cv, y_cv, test_size=100)
    plot_benchmark(benchmark_results, X_cv)  # Pass X_cv as argument


if __name__ == "__main__":
    main()
