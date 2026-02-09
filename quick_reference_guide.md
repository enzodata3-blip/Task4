# Quick Reference Guide: Key Methodologies from Jack-Cherish Repository

## Critical Preprocessing Steps

### Always Standardize Before Regularization
```python
# REQUIRED before ridge regression, stagewise regression, and similar methods
yMean = np.mean(yMat, axis=0)
yMat = yMat - yMean  # Center y

xMeans = np.mean(xMat, axis=0)
xVar = np.var(xMat, axis=0)
xMat = (xMat - xMeans) / xVar  # Z-score standardization
```

**Why:** Regularization penalizes large coefficients. Without standardization, features with larger scales get penalized more, biasing results.

---

## Regularization Techniques Comparison

| Method | Use Case | Strength | Weakness |
|--------|----------|----------|----------|
| **Ridge Regression** | Multicollinearity, many features | Stable, shrinks all coefficients | Doesn't select features (all non-zero) |
| **Stagewise Regression** | Feature selection needed | Produces sparse models | Slower than Lasso |
| **LWLR** | Non-linear relationships | No parametric assumptions | Computationally expensive |

---

## Ridge Regression: Lambda Selection Strategy

### Test Exponentially Spaced Values
```python
# Test λ from very small to very large
for i in range(30):
    lam = np.exp(i - 10)  # e^-10 to e^19
    ws = ridge_regression(xMat, yMat, lam)
```

### Interpretation Guide
- **λ → 0**: Approaches standard linear regression (no regularization)
- **λ = small (0.01-0.1)**: Light regularization, coefficients near OLS
- **λ = medium (1-10)**: Moderate shrinkage, often optimal
- **λ = large (100+)**: Heavy shrinkage, coefficients approach zero

### Visual Selection
Plot coefficient paths and look for:
1. **Stability region**: Where coefficients stop changing rapidly
2. **Divergence point**: Where coefficients start diverging (too little regularization)
3. **Convergence point**: Where coefficients approach zero (too much regularization)

---

## LWLR: Bandwidth Selection

### Bandwidth Parameter k Effects

| k Value | Behavior | Train Error | Test Error | Use When |
|---------|----------|-------------|------------|----------|
| Very small (0.01-0.1) | Highly local, follows noise | Very low | High | Never (overfits) |
| Small (0.5-2.0) | Local fitting | Low | Medium | Non-linear, enough data |
| Medium (2-10) | Semi-local | Medium | Low | Good default range |
| Large (10+) | Nearly global | High | Low | Linear relationship suspected |

### Selection Process
```python
# Test multiple k values on validation set
k_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for k in k_values:
    yHat = lwlr_test(xTest, xTrain, yTrain, k)
    error = rss_error(yTest, yHat)
    print(f"k={k}: RSS={error}")

# Choose k with lowest test error
```

**Key Insight:** If k=0.1 has much higher test error than k=1.0, you're overfitting. If all k values have similar error, relationship is likely linear (use standard regression).

---

## Gradient Descent: Learning Rate Strategies

### Batch Gradient Ascent (Logistic Regression)
```python
alpha = 0.001  # Fixed learning rate
maxCycles = 500

for k in range(maxCycles):
    h = sigmoid(dataMatrix * weights)
    error = labelMat - h
    weights = weights + alpha * dataMatrix.T * error
```

**Pros:** Stable, guaranteed convergence (if α small enough)
**Cons:** Slow on large datasets, can get stuck in local minima

### Stochastic Gradient Ascent
```python
# Adaptive learning rate
alpha = 4/(1.0 + j + i) + 0.01

# Random sample selection
randIndex = int(np.random.uniform(0, len(dataIndex)))
```

**Adaptive Rate Benefits:**
- `4/(1.0 + j + i)`: Decreases over time (helps convergence)
- `+ 0.01`: Never reaches zero (continues making progress)
- Result: Fast early progress, fine-tuning later

**Pros:** Much faster than batch, escapes local minima
**Cons:** More variance in convergence path

---

## Train/Test Splitting Best Practices

### Method 1: Pre-split Files (Recommended for Consistent Comparison)
```python
trainData = load_data('horseColicTraining.txt')
testData = load_data('horseColicTest.txt')
```
**Use when:** Multiple experiments need identical splits

### Method 2: Manual Index Split
```python
trainData = data[0:100]
testData = data[100:200]
```
**Use when:** Quick experiments, small datasets

### Method 3: Random Split with Seed
```python
np.random.seed(42)  # Reproducibility
indices = np.random.permutation(len(data))
splitPoint = int(len(data) * 0.7)
```
**Use when:** No pre-existing split, want randomization

### Typical Split Ratios
- **70/30**: Standard for medium datasets (100-1000 samples)
- **80/20**: Common for larger datasets (1000+ samples)
- **60/20/20**: Train/validation/test when tuning hyperparameters

---

## Performance Metrics Quick Reference

### Regression Metrics

```python
# Residual Sum of Squares (RSS)
rss = ((yActual - yPredicted)**2).sum()
# Use for: Model comparison (lower is better)
# Range: [0, ∞)

# Mean Squared Error (MSE)
mse = np.mean((yActual - yPredicted)**2)
# Use for: Averaging across samples
# Range: [0, ∞)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
# Use for: Same units as target variable
# Range: [0, ∞)

# R-squared
ss_res = np.sum((yActual - yPredicted)**2)
ss_tot = np.sum((yActual - np.mean(yActual))**2)
r2 = 1 - (ss_res / ss_tot)
# Use for: Proportion of variance explained
# Range: (-∞, 1], perfect fit = 1.0, baseline = 0
```

### Classification Metrics

```python
# Error Rate
error_rate = (misclassifications / total_samples) * 100
# Range: [0, 100], lower is better

# Accuracy
accuracy = 100 - error_rate
# Range: [0, 100], higher is better
```

---

## Detecting Overfitting

### Warning Signs

1. **Large Train/Test Gap**
```python
train_error = 10.5
test_error = 45.2  # Much higher → OVERFITTING
```

2. **Parameter Sensitivity**
```python
# LWLR with k=0.1: test_error = 500
# LWLR with k=1.0: test_error = 100  # k=0.1 overfits
```

3. **Coefficient Instability**
```python
# Ridge with λ=0.001: coefficients = [100, -95, 200, ...]  # Very large
# Ridge with λ=1.0: coefficients = [2.5, -1.8, 3.2, ...]  # Reasonable
```

### Solutions

| Problem | Solution | Implementation |
|---------|----------|----------------|
| High variance | Increase regularization | Increase λ (ridge), increase k (LWLR) |
| Model too complex | Feature selection | Stagewise regression, remove features |
| Too little data | Cross-validation | k-fold CV instead of single split |
| Multicollinearity | Ridge regression | Add L2 penalty |

---

## Feature Engineering Insights

### Information Gain (Decision Trees)
- Selects features that best separate classes
- Based on entropy reduction
- Handles feature interactions automatically

### Stagewise Regression Interpretation
```python
# After 200 iterations:
coefficients = [2.5, -1.8, 0.05, 3.2, 0.0, 0.0, ...]
#               ^^^   ^^^^  ^^^^  ^^^  ^^^  ^^^
#               Important   Small  Not selected (zero)
```

**Feature Importance:** Order of entry + final coefficient magnitude

### Naive Bayes Feature Independence
- Assumes P(x₁, x₂ | y) = P(x₁ | y) × P(x₂ | y)
- Works well even when assumption violated (surprisingly robust)
- Fast computation, interpretable probabilities

---

## Model Comparison Workflow

### Step 1: Baseline
```python
# Standard linear regression (no regularization)
ws = (X.T @ X).I @ X.T @ y
```

### Step 2: Regularized Models
```python
# Test multiple regularization strengths
for lam in [0.01, 0.1, 1.0, 10.0]:
    ws_ridge = ridge_regression(X, y, lam)
```

### Step 3: Non-parametric Alternative
```python
# LWLR with optimal bandwidth
ws_lwlr = lwlr_test(xTest, xTrain, yTrain, k=1.0)
```

### Step 4: Compare Performance
```python
print(f"Baseline:      RSS = {rss_baseline}")
print(f"Ridge (λ=1):   RSS = {rss_ridge}")
print(f"LWLR (k=1):    RSS = {rss_lwlr}")
```

### Decision Rules
- If ridge ≈ baseline: No multicollinearity, regularization not needed
- If ridge < baseline: Multicollinearity present, use ridge
- If LWLR << ridge: Non-linear relationship, use LWLR
- If ridge ≈ LWLR: Linear relationship, prefer ridge (faster)

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting Standardization
```python
# WRONG
ws = ridge_regression(X, y, lam=1.0)

# CORRECT
X_std = (X - X.mean()) / X.std()
y_std = y - y.mean()
ws = ridge_regression(X_std, y_std, lam=1.0)
```

### Pitfall 2: Using Training Error for Model Selection
```python
# WRONG: Choose model with lowest training error
best_model = min(models, key=lambda m: m.train_error)

# CORRECT: Choose model with lowest validation/test error
best_model = min(models, key=lambda m: m.test_error)
```

### Pitfall 3: Testing Too Few λ or k Values
```python
# WRONG: Only test 3 values
lambda_values = [0.1, 1.0, 10.0]

# CORRECT: Test wide range (exponentially spaced)
lambda_values = [np.exp(i-10) for i in range(30)]
```

### Pitfall 4: No Random Sampling in SGD
```python
# WRONG: Sequential sampling
for i in range(m):
    update_weights(X[i], y[i])

# CORRECT: Random sampling
for i in range(m):
    randIndex = np.random.randint(0, len(dataIndex))
    update_weights(X[randIndex], y[randIndex])
```

---

## Recommended Implementation Order

### Phase 1: Foundation (Start Here)
1. Data loading and standardization functions
2. Standard linear regression (baseline)
3. Train/test splitting
4. RSS/MSE metrics

### Phase 2: Regularization
5. Ridge regression with single λ
6. Ridge regression with multiple λ (regularization path)
7. Coefficient visualization

### Phase 3: Advanced Regression
8. Locally weighted linear regression
9. Bandwidth selection
10. LWLR vs. standard regression comparison

### Phase 4: Classification (If Needed)
11. Logistic regression with gradient ascent
12. Stochastic gradient descent
13. Error rate / accuracy metrics

### Phase 5: Optimization (If Time Permits)
14. Forward stagewise regression
15. Feature selection analysis
16. Cross-validation

---

## Key Formulas Reference

### Ridge Regression
```
w = (X^T X + λI)^(-1) X^T y
```
- λ = 0: Standard OLS
- λ → ∞: w → 0

### Locally Weighted Regression
```
w(x) = (X^T W(x) X)^(-1) X^T W(x) y
W_ii(x) = exp(-||x - x_i||^2 / (2k^2))
```
- k small: local fit
- k large: global fit

### Logistic Regression (Gradient Ascent)
```
w := w + α X^T (y - σ(Xw))
σ(z) = 1 / (1 + e^(-z))
```
- α: learning rate
- σ: sigmoid function

### Adaptive Learning Rate (SGD)
```
α = 4/(1 + j + i) + 0.01
```
- j: epoch number
- i: sample number within epoch

---

## Code Quality Checklist

Before implementing, ensure:

- [ ] Data standardization before regularization
- [ ] Separate train/test sets
- [ ] Test multiple hyperparameter values
- [ ] Calculate both train and test error
- [ ] Visualize coefficient paths or decision boundaries
- [ ] Compare to baseline (standard regression)
- [ ] Check for singular matrices (determinant = 0)
- [ ] Use proper random seed for reproducibility
- [ ] Document hyperparameter choices
- [ ] Include unit tests for core functions

---

## Performance Optimization Tips

### For Large Datasets
1. Use stochastic gradient descent instead of batch
2. Consider sampling for LWLR (compute only for subset)
3. Vectorize operations (use NumPy matrix operations)
4. Pre-compute kernel matrices if using multiple times

### For High-Dimensional Data
1. Use ridge regression to handle multicollinearity
2. Apply stagewise regression for feature selection
3. Consider PCA for dimensionality reduction first
4. Monitor for singular matrix errors

### For Non-Linear Relationships
1. Try LWLR with various bandwidths
2. Consider polynomial features
3. Use regression trees (CART)
4. Test kernel-based methods (SVM)

---

## Final Recommendations

### For Your Implementation

**Priority 1: Core Infrastructure**
- Data loading and standardization
- Train/test splitting with proper validation
- RSS/MSE/R² metrics
- Coefficient visualization

**Priority 2: Regularization**
- Ridge regression with regularization path
- Automated λ selection via cross-validation
- Comparison to baseline

**Priority 3: Advanced Features**
- LWLR for non-linear relationships
- Stagewise regression for feature selection
- Model comparison framework

### Best Practices from Repository

1. **Always standardize** before regularization
2. **Test multiple hyperparameters** (don't assume optimal value)
3. **Visualize results** (coefficient paths, decision boundaries)
4. **Compare to baseline** (standard regression / random classifier)
5. **Use proper validation** (separate test set, cross-validation)
6. **Document experiments** (which parameters, why chosen)

### When to Use Each Method

**Use Ridge Regression when:**
- Features are correlated (multicollinearity)
- Need stable, interpretable coefficients
- Have many features relative to samples

**Use LWLR when:**
- Relationship clearly non-linear
- Have enough data (100+ samples)
- Computation time not critical
- Interpretability less important

**Use Stagewise Regression when:**
- Need feature selection (sparse model)
- Have many irrelevant features
- Interpretability important (which features matter)

**Use Standard Regression when:**
- Relationship is linear
- No multicollinearity
- Simple baseline needed
