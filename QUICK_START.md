# Quick Start Guide: 5-Minute Implementation

## Most Important Takeaways

### 1. Ridge Regression is Your Best Starting Point
- Handles multicollinearity
- Easy to implement
- Produces stable results
- Widely applicable

### 2. ALWAYS Standardize Before Regularization
```python
# This is NON-NEGOTIABLE
yMean = np.mean(yMat, axis=0)
yMat = yMat - yMean
xMeans = np.mean(xMat, axis=0)
xVar = np.var(xMat, axis=0)
xMat = (xMat - xMeans) / xVar
```

### 3. Test Multiple Lambda Values
```python
# Don't guess - test exponentially-spaced values
for i in range(30):
    lam = np.exp(i - 10)  # e^-10 to e^19
    ws = ridge_regression(xMat, yMat, lam)
```

---

## Copy-Paste Ready Code

### Complete Ridge Regression Pipeline
```python
import numpy as np
import matplotlib.pyplot as plt

def ridge_regression_pipeline(X_train, y_train, X_test, y_test):
    """
    Complete ridge regression with visualization

    Usage:
        best_lambda, test_error = ridge_regression_pipeline(X_train, y_train, X_test, y_test)
    """
    # Convert to matrices
    xMat_train = np.mat(X_train)
    yMat_train = np.mat(y_train).T
    xMat_test = np.mat(X_test)
    yMat_test = np.mat(y_test).T

    # CRITICAL: Standardize using training set statistics
    yMean = np.mean(yMat_train, axis=0)
    yMat_train_std = yMat_train - yMean
    yMat_test_std = yMat_test - yMean

    xMeans = np.mean(xMat_train, axis=0)
    xVar = np.var(xMat_train, axis=0)
    xMat_train_std = (xMat_train - xMeans) / xVar
    xMat_test_std = (xMat_test - xMeans) / xVar

    # Test multiple lambda values
    num_lambdas = 30
    n_features = xMat_train_std.shape[1]
    wMat = np.zeros((num_lambdas, n_features))
    train_errors = []
    test_errors = []
    lambda_values = []

    for i in range(num_lambdas):
        lam = np.exp(i - 10)
        lambda_values.append(lam)

        # Ridge regression formula: w = (X^T X + λI)^(-1) X^T y
        xTx = xMat_train_std.T * xMat_train_std
        denom = xTx + np.eye(n_features) * lam

        if np.linalg.det(denom) != 0.0:
            ws = denom.I * (xMat_train_std.T * yMat_train_std)
            wMat[i, :] = ws.T

            # Calculate errors
            yHat_train = xMat_train_std * ws
            yHat_test = xMat_test_std * ws
            train_errors.append(((yMat_train_std - yHat_train)**2).sum())
            test_errors.append(((yMat_test_std - yHat_test)**2).sum())

    # Find optimal lambda (minimum test error)
    best_idx = np.argmin(test_errors)
    best_lambda = lambda_values[best_idx]

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coefficient paths
    ax1.plot(np.log(lambda_values), wMat)
    ax1.axvline(np.log(best_lambda), color='red', linestyle='--', label=f'Best λ={best_lambda:.4f}')
    ax1.set_xlabel('log(λ)', fontsize=12)
    ax1.set_ylabel('Coefficient Values', fontsize=12)
    ax1.set_title('Ridge Regression: Coefficient Shrinkage', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Train vs Test Error
    ax2.plot(np.log(lambda_values), train_errors, label='Training Error', marker='o', markersize=3)
    ax2.plot(np.log(lambda_values), test_errors, label='Test Error', marker='s', markersize=3)
    ax2.axvline(np.log(best_lambda), color='red', linestyle='--', label=f'Best λ={best_lambda:.4f}')
    ax2.set_xlabel('log(λ)', fontsize=12)
    ax2.set_ylabel('Sum of Squared Errors', fontsize=12)
    ax2.set_title('Model Selection: Train vs Test Error', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ridge_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print results
    print("="*60)
    print("RIDGE REGRESSION RESULTS")
    print("="*60)
    print(f"Optimal λ: {best_lambda:.6f}")
    print(f"Training Error: {train_errors[best_idx]:.2f}")
    print(f"Test Error: {test_errors[best_idx]:.2f}")
    print(f"Optimal Coefficients: {wMat[best_idx, :]}")
    print("="*60)

    return best_lambda, test_errors[best_idx]
```

### How to Use
```python
# Example with your data
import numpy as np

# Load your data (replace with your actual data loading)
# X should be (n_samples, n_features)
# y should be (n_samples,)

# Option 1: Random split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# Option 2: Manual split
# X_train = X[:100]
# y_train = y[:100]
# X_test = X[100:]
# y_test = y[100:]

# Run pipeline
best_lambda, test_error = ridge_regression_pipeline(X_train, y_train, X_test, y_test)
```

---

## Alternative: LWLR for Non-Linear Data

### Copy-Paste Ready Code
```python
def lwlr_pipeline(X_train, y_train, X_test, y_test):
    """
    Locally Weighted Linear Regression with bandwidth selection

    Usage:
        best_k, predictions = lwlr_pipeline(X_train, y_train, X_test, y_test)
    """
    xMat_train = np.mat(X_train)
    yMat_train = np.mat(y_train).T
    xMat_test = np.mat(X_test)

    # Test multiple bandwidth values
    k_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    errors = []

    print("="*60)
    print("LWLR BANDWIDTH SELECTION")
    print("="*60)

    for k in k_values:
        # Predict for each test point
        m_test = xMat_test.shape[0]
        yHat = np.zeros(m_test)

        for i in range(m_test):
            testPoint = xMat_test[i, :]

            # Create weight matrix with Gaussian kernel
            m_train = xMat_train.shape[0]
            weights = np.mat(np.eye(m_train))

            for j in range(m_train):
                diffMat = testPoint - xMat_train[j, :]
                weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))

            # Weighted regression
            xTx = xMat_train.T * (weights * xMat_train)
            if np.linalg.det(xTx) != 0.0:
                ws = xTx.I * (xMat_train.T * (weights * yMat_train))
                yHat[i] = testPoint * ws

        # Calculate error
        error = ((y_test - yHat)**2).sum()
        errors.append(error)
        print(f"k = {k:6.2f}  |  Test RSS = {error:12.2f}")

    # Find optimal k
    best_idx = np.argmin(errors)
    best_k = k_values[best_idx]

    print("="*60)
    print(f"Optimal bandwidth: k = {best_k}")
    print(f"Best test error: {errors[best_idx]:.2f}")
    print("="*60)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, marker='o', linewidth=2, markersize=8)
    plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('Bandwidth (k)', fontsize=12)
    plt.ylabel('Test Error (RSS)', fontsize=12)
    plt.title('LWLR: Bandwidth Selection', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lwlr_bandwidth_selection.png', dpi=300)
    plt.show()

    return best_k, errors[best_idx]
```

---

## Quick Comparison: Which Method to Use?

### Decision Tree

```
Start Here
    |
    v
Is relationship LINEAR?
    |
    +--YES--> Are features CORRELATED?
    |             |
    |             +--YES--> USE RIDGE REGRESSION
    |             |
    |             +--NO---> USE STANDARD REGRESSION
    |
    +--NO---> Do you have ENOUGH DATA (100+ samples)?
                  |
                  +--YES--> USE LWLR
                  |
                  +--NO---> Try polynomial features + Ridge
```

---

## Performance Metrics (Copy-Paste Ready)

```python
def evaluate_regression(y_true, y_pred):
    """Calculate all regression metrics"""
    # Residual Sum of Squares
    rss = np.sum((y_true - y_pred)**2)

    # Mean Squared Error
    mse = np.mean((y_true - y_pred)**2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    print("="*50)
    print("REGRESSION METRICS")
    print("="*50)
    print(f"RSS:  {rss:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    print("="*50)

    return {'RSS': rss, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}
```

---

## Common Issues and Quick Fixes

### Issue 1: "Matrix is singular, cannot compute inverse"
```python
# Fix: Add small regularization
lam = 0.001  # Minimum regularization
ws = (xTx + lam * np.eye(n)).I * (xMat.T * yMat)
```

### Issue 2: Coefficients are huge (±1000+)
```python
# Fix: You forgot to standardize!
xMat = (xMat - xMat.mean(axis=0)) / xMat.std(axis=0)
yMat = yMat - yMat.mean()
```

### Issue 3: Test error >> Train error
```python
# Fix: Increase regularization
lam = 10.0  # Instead of lam = 0.01
# OR increase bandwidth for LWLR
k = 5.0  # Instead of k = 0.1
```

### Issue 4: All methods perform similarly
```python
# This is actually good! Means:
# - Relationship is linear
# - No multicollinearity
# - No overfitting
# Use simplest model (standard regression)
```

---

## 30-Second Checklist

Before running any model:
- [ ] Data loaded correctly
- [ ] Train/test split performed (70/30 or 80/20)
- [ ] Features standardized (for regularization)
- [ ] Multiple hyperparameters tested (not just one value)
- [ ] Metrics calculated on TEST set (not training)

---

## Files You Need

### Essential
1. **implementation_examples.py** - Full working code
2. **quick_reference_guide.md** - Formulas and strategies

### Detailed Reference
3. **jack_cherish_ml_analysis.md** - Complete analysis
4. **SUMMARY_AND_NEXT_STEPS.md** - Action plan

---

## Getting Help

### Check These First
1. Is data standardized? (Most common issue)
2. Is train/test split done? (Second most common)
3. Are you testing multiple hyperparameters? (Third most common)
4. Are you evaluating on test set? (Fourth most common)

### Error Messages
- "Matrix is singular" → Add regularization (λ > 0)
- Huge coefficients → Standardize data
- Test error >> train error → Increase regularization
- Very slow → Use smaller dataset or increase k (LWLR)

---

## One-Line Summary

**Ridge regression with λ from 0.01 to 100, standardized data, 70/30 train/test split, evaluate on test RSS.**

That's it. Everything else is details.

---

## Immediate Next Action

```bash
cd /Users/apple/Documents/Task4/model_b
python implementation_examples.py
# Then modify for your data
```

**You're ready to start!**
