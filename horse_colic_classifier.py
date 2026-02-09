"""
Horse Colic Binary Classifier — Enhanced
=========================================
Translated and enhanced from the original Chinese implementation by Jack Cui.

Dataset: UCI Horse Colic (21 clinical veterinary features, binary survival label)
  - Features include: surgery type, age, rectal temperature, pulse,
    respiratory rate, gut sounds, pain level, and other clinical indicators.
  - Label: 0 = did not survive, 1 = survived

Enhancements over the original:
  - Full English translation of all comments and output
  - Feature standardization (zero mean, unit variance)
  - Interaction terms between the most correlated feature pairs
  - Comprehensive evaluation: confusion matrix, accuracy, precision,
    recall, specificity, F1, AUC-ROC, log-loss, MCC
  - Stratified k-fold cross-validation
  - Comparison: custom stochastic gradient ascent vs sklearn LogisticRegression
  - Correlation matrix analysis to guide interaction term selection
  - ROC curve and convergence plots
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef, roc_curve
)


# ---------------------------------------------------------------------------
# Utility: sigmoid
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load tab-separated horse colic data.

    Returns
    -------
    X : np.ndarray, shape (m, 21)  — feature matrix
    y : np.ndarray, shape (m,)     — binary labels (0/1)
    """
    X, y = [], []
    with open(filepath) as fh:
        for line in fh.readlines():
            parts = line.strip().split("\t")
            X.append([float(v) for v in parts[:-1]])
            y.append(float(parts[-1]))
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Correlation matrix analysis
# ---------------------------------------------------------------------------

def correlation_analysis(X, feature_names=None, top_n=5):
    """Compute and display the correlation matrix.

    Identifies the top-N most correlated feature pairs to guide
    interaction term selection.

    Returns
    -------
    top_pairs : list of (i, j, correlation) sorted by |correlation|
    """
    corr = np.corrcoef(X.T)
    n = corr.shape[0]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, corr[i, j]))

    # Sort by absolute correlation, descending
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    top_pairs = pairs[:top_n]

    names = feature_names or [f"F{i}" for i in range(n)]
    print("\nTop correlated feature pairs (candidates for interaction terms):")
    print(f"  {'Feature A':<12} {'Feature B':<12} {'Correlation':>12}")
    for i, j, r in top_pairs:
        print(f"  {names[i]:<12} {names[j]:<12} {r:>12.4f}")

    return top_pairs


def add_interaction_terms(X, pairs):
    """Append pairwise interaction columns (x_i * x_j) to X.

    Parameters
    ----------
    X     : np.ndarray, shape (m, n)
    pairs : list of (i, j, _)  — feature index pairs

    Returns
    -------
    X_aug : np.ndarray, shape (m, n + len(pairs))
    """
    interactions = [X[:, i] * X[:, j] for i, j, *_ in pairs]
    return np.hstack([X] + [col.reshape(-1, 1) for col in interactions])


# ---------------------------------------------------------------------------
# Feature standardization
# ---------------------------------------------------------------------------

def standardize(X_train, X_test):
    """Fit scaler on train, apply to both train and test."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Custom: stochastic gradient ascent with decaying learning rate
# ---------------------------------------------------------------------------

def stochastic_gradient_ascent(X, y, num_iterations=500):
    """Improved stochastic gradient ascent for logistic regression.

    Learning rate decays as alpha = 4 / (1 + j + i) + 0.01
    to prevent oscillation near convergence.

    Returns
    -------
    weights      : np.ndarray, shape (n,)
    loss_history : list of float — log-loss after each full pass
    """
    m, n = X.shape
    weights = np.ones(n)
    loss_history = []

    for j in range(num_iterations):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4.0 / (1.0 + j + i) + 0.01
            rand_idx = int(random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(X[rand_idx] * weights))
            error = y[rand_idx] - h
            weights += alpha * error * X[rand_idx]
            del data_index[rand_idx]

        # Log-loss after full pass
        probs = sigmoid(X @ weights)
        loss_history.append(log_loss(y, probs))

    return weights, loss_history


def classify(x, weights):
    """Classify a single sample."""
    return 1.0 if sigmoid(np.sum(x * weights)) > 0.5 else 0.0


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_prob, model_name="Model"):
    """Print a comprehensive suite of classification metrics."""
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
    print(f"  Accuracy    : {accuracy:.4f}  ({(1-accuracy)*100:.2f}% error rate)")
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
                                 target_names=["Did Not Survive", "Survived"]))

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "auc_roc": auc,
        "log_loss": ll, "mcc": mcc,
    }


# ---------------------------------------------------------------------------
# Stratified k-fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate(X, y, n_splits=5, model_name="Logistic Regression"):
    """Run stratified k-fold CV and report mean ± std for each metric."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Standardize within fold to prevent data leakage
        X_tr_s, X_val_s, _ = standardize(X_tr, X_val)

        clf = LogisticRegression(solver="lbfgs", max_iter=2000,
                                  C=1.0, random_state=42)
        clf.fit(X_tr_s, y_tr)

        y_pred = clf.predict(X_val_s)
        y_prob = clf.predict_proba(X_val_s)[:, 1]

        metrics_list.append({
            "accuracy": (y_pred == y_val).mean(),
            "auc_roc": roc_auc_score(y_val, y_prob),
            "log_loss": log_loss(y_val, y_prob),
            "f1": (2 * (y_pred == 1) & (y_val == 1)).sum() /
                  max(((y_pred == 1).sum() + (y_val == 1).sum()), 1),
            "mcc": matthews_corrcoef(y_val, y_pred),
        })

    print(f"\n{'='*60}")
    print(f"  {n_splits}-Fold Cross-Validation — {model_name}")
    print(f"{'='*60}")
    keys = list(metrics_list[0].keys())
    for key in keys:
        vals = [m[key] for m in metrics_list]
        print(f"  {key:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return metrics_list


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_convergence(loss_history, model_name="Stochastic Gradient Ascent"):
    """Plot log-loss convergence per iteration."""
    plt.figure()
    plt.plot(loss_history, color="steelblue", lw=1.5)
    plt.xlabel("Iteration (full pass)")
    plt.ylabel("Log-Loss")
    plt.title(f"Convergence — {model_name}")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true, results):
    """Overlay ROC curves for multiple models.

    Parameters
    ----------
    results : list of (model_name, y_prob)
    """
    plt.figure()
    for name, y_prob in results:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(X, feature_names=None, title="Feature Correlation Matrix"):
    """Display the correlation heatmap."""
    corr = np.corrcoef(X.T)
    n = corr.shape[0]
    names = feature_names or [f"F{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # UCI Horse Colic feature names (translated from original dataset documentation)
    feature_names = [
        "surgery",            # 1=yes, 2=no
        "age",                # 1=adult, 2=young
        "rectal_temp",        # degrees Celsius
        "pulse",              # beats per minute
        "respiratory_rate",   # breaths per minute
        "temp_extremities",   # 1=normal, 2=warm, 3=cool, 4=cold
        "peripheral_pulse",   # 1=normal, 2=increased, 3=reduced, 4=absent
        "mucous_membranes",   # 1-6 scale
        "capillary_refill",   # 1=<3sec, 2=>=3sec
        "pain",               # 1=none, 2=mild, 3=moderate, 4=severe, 5=extreme
        "peristalsis",        # 1=hypermotile, 2=normal, 3=hypomotile, 4=absent
        "abdominal_distension",  # 1=none, 2=slight, 3=moderate, 4=severe
        "nasogastric_tube",   # 1=none, 2=slight, 3=significant
        "nasogastric_reflux", # 1=none, 2=>1L, 3=<1L
        "nasogastric_reflux_ph",
        "rectal_exam_feces",  # 1=normal, 2=decreased, 3=absent, 4=increased
        "abdomen",            # 1=normal, 2=other, 3=firm, 4=distended_small, 5=distended_large
        "packed_cell_volume",
        "total_protein",
        "abdomcentesis_appearance",  # 1=clear, 2=cloudy, 3=serosanguinous
        "abdomcentesis_total_protein",
    ]

    # --- Load data ---
    print("Loading horse colic dataset...")
    X_train, y_train = load_data("horseColicTraining.txt")
    X_test, y_test = load_data("horseColicTest.txt")
    print(f"  Training samples : {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Survival rate    : {y_train.mean()*100:.1f}% train / {y_test.mean()*100:.1f}% test")

    # --- Correlation analysis to select interaction terms ---
    plot_correlation_matrix(X_train, feature_names, "Feature Correlation Matrix (Training)")
    top_pairs = correlation_analysis(X_train, feature_names, top_n=5)

    # --- Add interaction terms (top 3 most correlated pairs) ---
    interaction_pairs = [(i, j) for i, j, _ in top_pairs[:3]]
    X_train_aug = add_interaction_terms(X_train, interaction_pairs)
    X_test_aug = add_interaction_terms(X_test, interaction_pairs)
    print(f"\nAdded {len(interaction_pairs)} interaction terms: "
          f"{[(feature_names[i], feature_names[j]) for i, j in interaction_pairs]}")
    print(f"Augmented feature count: {X_train_aug.shape[1]}")

    # --- Standardize ---
    X_train_s, X_test_s, scaler = standardize(X_train_aug, X_test_aug)

    # ===================================================================
    # Model A: Custom stochastic gradient ascent
    # ===================================================================
    print("\n--- Training: Custom Stochastic Gradient Ascent ---")
    weights, loss_hist = stochastic_gradient_ascent(X_train_s, y_train, num_iterations=500)
    plot_convergence(loss_hist, "Custom Stochastic Gradient Ascent")

    y_pred_custom = np.array([classify(X_test_s[i], weights)
                               for i in range(X_test_s.shape[0])])
    y_prob_custom = sigmoid(X_test_s @ weights)
    metrics_custom = evaluate(y_test, y_pred_custom, y_prob_custom,
                               model_name="Custom Stochastic Gradient Ascent")

    # ===================================================================
    # Model B: Sklearn LogisticRegression (L2, lbfgs solver)
    # ===================================================================
    print("\n--- Training: Sklearn LogisticRegression (L2 regularization) ---")
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred_sklearn = clf.predict(X_test_s)
    y_prob_sklearn = clf.predict_proba(X_test_s)[:, 1]
    metrics_sklearn = evaluate(y_test, y_pred_sklearn, y_prob_sklearn,
                                model_name="Sklearn LogisticRegression (L2)")

    # ===================================================================
    # ROC curve comparison
    # ===================================================================
    plot_roc_curves(y_test, [
        ("Custom SGD", y_prob_custom),
        ("Sklearn LR", y_prob_sklearn),
    ])

    # ===================================================================
    # Stratified k-fold cross-validation (on full dataset, augmented)
    # ===================================================================
    X_full = np.vstack([X_train_aug, X_test_aug])
    y_full = np.concatenate([y_train, y_test])
    cross_validate(X_full, y_full, n_splits=5,
                   model_name="LogisticRegression (L2, lbfgs)")

    # ===================================================================
    # Summary comparison
    # ===================================================================
    print(f"\n{'='*60}")
    print("  Summary Comparison")
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'Custom SGD':>14} {'Sklearn LR':>14}")
    print(f"  {'-'*44}")
    for key in ["accuracy", "precision", "recall", "f1", "auc_roc",
                "log_loss", "mcc"]:
        print(f"  {key:<14} {metrics_custom[key]:>14.4f} {metrics_sklearn[key]:>14.4f}")
