"""
Logistic Regression — Enhanced Base Module
==========================================
Translated and enhanced from the original Chinese implementation by Jack Cui.

Enhancements over the original:
  - Full English translation of all comments and output
  - Feature standardization (zero mean, unit variance)
  - L2 regularization in batch gradient ascent
  - Comprehensive evaluation: accuracy, precision, recall, F1,
    AUC-ROC, log-loss, Matthews Correlation Coefficient (MCC)
  - Convergence monitoring via log-loss curve
  - Decision boundary visualization with evaluation overlay
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef, roc_curve
)


# ---------------------------------------------------------------------------
# Gradient ascent demo: find maximum of f(x) = -x^2 + 4x
# ---------------------------------------------------------------------------

def gradient_ascent_demo():
    """Illustrate gradient ascent by finding the maximum of f(x) = -x^2 + 4x.

    The derivative f'(x) = -2x + 4 is zero at x=2, confirming the maximum.
    """
    def f_prime(x):
        return -2 * x + 4

    x_old = -1.0          # initial guess (must differ from x_new to enter loop)
    x_new = 0.0           # starting point
    learning_rate = 0.01  # controls update magnitude
    precision = 1e-8      # convergence threshold

    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new = x_old + learning_rate * f_prime(x_old)

    print(f"[Gradient Ascent Demo] Maximum of f(x) = -x^2 + 4x found at x = {x_new:.8f}")


# ---------------------------------------------------------------------------
# Data loading and standardization
# ---------------------------------------------------------------------------

def load_dataset(filepath="testSet.txt"):
    """Load the 2D binary classification dataset.

    Each row contains: x1  x2  label (space-separated).
    A bias column of 1.0 is prepended to every sample.

    Returns
    -------
    data_matrix : list of [1.0, x1, x2]
    labels      : list of int (0 or 1)
    """
    data_matrix = []
    labels = []
    with open(filepath) as fh:
        for line in fh.readlines():
            parts = line.strip().split()
            data_matrix.append([1.0, float(parts[0]), float(parts[1])])
            labels.append(int(parts[2]))
    return data_matrix, labels


def standardize_features(data_matrix):
    """Standardize feature columns to zero mean and unit variance.

    The bias column (index 0, all ones) is left untouched.

    Parameters
    ----------
    data_matrix : np.ndarray, shape (m, n)

    Returns
    -------
    standardized : np.ndarray, shape (m, n)
    means        : np.ndarray, shape (n-1,)  — feature means
    stds         : np.ndarray, shape (n-1,)  — feature standard deviations
    """
    arr = np.array(data_matrix, dtype=float)
    means = arr[:, 1:].mean(axis=0)
    stds = arr[:, 1:].std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features
    arr[:, 1:] = (arr[:, 1:] - means) / stds
    return arr, means, stds


# ---------------------------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


# ---------------------------------------------------------------------------
# Batch gradient ascent (with optional L2 regularization)
# ---------------------------------------------------------------------------

def gradient_ascent(data_matrix, labels, learning_rate=0.001,
                    max_iterations=500, l2_lambda=0.0):
    """Batch gradient ascent for logistic regression.

    Maximises the log-likelihood:
        L(w) = sum_i [ y_i * log(h_i) + (1-y_i) * log(1-h_i) ] - (l2_lambda/2) * ||w||^2

    Parameters
    ----------
    data_matrix   : array-like, shape (m, n)  — includes bias column
    labels        : array-like, shape (m,)
    learning_rate : float
    max_iterations: int
    l2_lambda     : float — L2 regularisation strength (0 = disabled)

    Returns
    -------
    weights      : np.ndarray, shape (n,)
    loss_history : list of float — log-loss at each iteration
    """
    X = np.mat(data_matrix, dtype=float)     # (m, n)
    y = np.mat(labels, dtype=float).T        # (m, 1)
    m, n = X.shape

    weights = np.ones((n, 1))
    loss_history = []

    for _ in range(max_iterations):
        h = sigmoid(X * weights)             # predicted probabilities (m, 1)
        error = y - h                        # residual
        # L2 penalty gradient (skip bias term at index 0)
        penalty = np.zeros_like(weights)
        penalty[1:] = l2_lambda * weights[1:]
        weights = weights + learning_rate * (X.T * error - penalty)

        # Record log-loss for convergence monitoring
        h_arr = np.asarray(h).ravel()
        y_arr = np.asarray(y).ravel()
        current_loss = log_loss(y_arr, h_arr)
        loss_history.append(current_loss)

    return np.asarray(weights).ravel(), loss_history


# ---------------------------------------------------------------------------
# Stochastic gradient ascent (improved, decreasing learning rate)
# ---------------------------------------------------------------------------

def stochastic_gradient_ascent(data_matrix, labels, num_iterations=150):
    """Improved stochastic gradient ascent with decaying learning rate.

    Learning rate decays as: alpha = 4 / (1 + j + i) + 0.01
    which prevents oscillation near convergence.

    Parameters
    ----------
    data_matrix   : np.ndarray, shape (m, n)
    labels        : list or np.ndarray, shape (m,)
    num_iterations: int

    Returns
    -------
    weights      : np.ndarray, shape (n,)
    loss_history : list of float — log-loss after each full pass
    """
    import random
    m, n = data_matrix.shape
    weights = np.ones(n)
    loss_history = []

    for j in range(num_iterations):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4.0 / (1.0 + j + i) + 0.01   # decaying learning rate
            rand_idx = int(random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_matrix[rand_idx] * weights))
            error = labels[rand_idx] - h
            weights += alpha * error * data_matrix[rand_idx]
            del data_index[rand_idx]

        # Record log-loss after each full pass
        probs = sigmoid(data_matrix @ weights)
        loss_history.append(log_loss(labels, probs))

    return weights, loss_history


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_proba(data_matrix, weights):
    """Return predicted probabilities P(y=1 | x) for each sample."""
    return sigmoid(np.array(data_matrix) @ weights)


def predict(data_matrix, weights, threshold=0.5):
    """Return binary predictions using the given probability threshold."""
    return (predict_proba(data_matrix, weights) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_prob, model_name="Model"):
    """Print a comprehensive suite of classification metrics.

    Metrics reported
    ----------------
    - Confusion matrix
    - Accuracy, Precision, Recall, F1-score (per class + macro/weighted)
    - AUC-ROC
    - Log-loss
    - Matthews Correlation Coefficient (MCC)
    """
    print(f"\n{'='*60}")
    print(f"  Evaluation — {model_name}")
    print(f"{'='*60}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0       {tn:6d}        {fp:6d}")
    print(f"  Actual 1       {fn:6d}        {tp:6d}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    print(f"\nCore Metrics:")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}  (TP / (TP+FP))")
    print(f"  Recall      : {recall:.4f}  (sensitivity, TP / (TP+FN))")
    print(f"  Specificity : {specificity:.4f}  (TN / (TN+FP))")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"\nAdvanced Metrics:")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Log-Loss    : {ll:.4f}")
    print(f"  MCC         : {mcc:.4f}  (ranges -1 to +1, 0 = random)")

    print(f"\nFull Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["Class 0", "Class 1"]))

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "auc_roc": auc,
        "log_loss": ll, "mcc": mcc,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_decision_boundary(data_matrix, labels, weights, title="Decision Boundary"):
    """Plot the data points and the learned decision boundary."""
    arr = np.array(data_matrix)
    labels = np.array(labels)

    pos = arr[labels == 1]
    neg = arr[labels == 0]

    fig, ax = plt.subplots()
    ax.scatter(pos[:, 1], pos[:, 2], s=20, c="red", marker="s", alpha=0.6, label="Class 1")
    ax.scatter(neg[:, 1], neg[:, 2], s=20, c="green", alpha=0.6, label="Class 0")

    # Decision boundary: w0 + w1*x1 + w2*x2 = 0  =>  x2 = -(w0 + w1*x1) / w2
    x1_range = np.arange(arr[:, 1].min() - 0.5, arr[:, 1].max() + 0.5, 0.1)
    x2_boundary = -(weights[0] + weights[1] * x1_range) / weights[2]
    ax.plot(x1_range, x2_boundary, color="blue", linewidth=1.5, label="Decision boundary")

    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name="Model"):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_convergence(loss_history, model_name="Model"):
    """Plot the log-loss convergence curve."""
    plt.figure()
    plt.plot(loss_history, color="steelblue", lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Loss")
    plt.title(f"Convergence — {model_name}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main: run batch gradient ascent on the 2D synthetic dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Gradient ascent demo ---
    gradient_ascent_demo()

    # --- Load and standardize data ---
    raw_data, labels = load_dataset("testSet.txt")
    X_std, feat_means, feat_stds = standardize_features(raw_data)

    # --- Train: batch gradient ascent with L2 regularization ---
    weights, loss_hist = gradient_ascent(
        X_std, labels,
        learning_rate=0.005,
        max_iterations=1000,
        l2_lambda=0.01,
    )
    print(f"\n[Batch Gradient Ascent] Final weights: {weights}")

    # --- Evaluate ---
    y_prob = predict_proba(X_std, weights)
    y_pred = predict(X_std, weights)
    evaluate(labels, y_pred, y_prob, model_name="Logistic Regression (Batch GD)")

    # --- Visualizations ---
    plot_convergence(loss_hist, model_name="Batch Gradient Ascent")
    plot_decision_boundary(X_std, labels, weights,
                           title="Decision Boundary (Batch GD, Standardized)")
    plot_roc_curve(labels, y_prob, model_name="Batch Gradient Ascent")
