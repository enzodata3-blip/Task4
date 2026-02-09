# -*- coding: utf-8 -*-
"""
Implementation Examples from Jack-Cherish Machine Learning Repository
Extracted and adapted for use in statistical analysis projects
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. DATA PREPROCESSING AND STANDARDIZATION
# =============================================================================

def load_data_tabdelimited(fileName):
    """
    Load tab-delimited data file

    Parameters:
        fileName - path to data file

    Returns:
        xArr - feature matrix (list of lists)
        yArr - target vector (list)
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def standardize_data(xMat, yMat):
    """
    Standardize data: zero mean, unit variance
    CRITICAL: Must be applied before ridge regression and other regularized methods

    Parameters:
        xMat - feature matrix (numpy matrix)
        yMat - target vector (numpy matrix)

    Returns:
        inxMat - standardized feature matrix
        inyMat - standardized target vector
    """
    inxMat = xMat.copy()
    inyMat = yMat.copy()

    # Standardize y (target)
    yMean = np.mean(yMat, 0)
    inyMat = yMat - yMean

    # Standardize X (features)
    inMeans = np.mean(inxMat, 0)  # Mean of each feature
    inVar = np.var(inxMat, 0)     # Variance of each feature
    inxMat = (inxMat - inMeans) / inVar  # Z-score normalization

    return inxMat, inyMat


def normalize_minmax(dataSet):
    """
    Min-max normalization: scales features to [0, 1]
    Used in kNN and other distance-based algorithms

    Parameters:
        dataSet - numpy array of features

    Returns:
        normDataSet - normalized data
        ranges - range of each feature (for inverse transform)
        minVals - minimum values of each feature
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet, ranges, minVals


# =============================================================================
# 2. RIDGE REGRESSION (L2 REGULARIZATION)
# =============================================================================

def ridge_regression(xMat, yMat, lam=0.2):
    """
    Ridge regression with L2 regularization

    Formula: w = (X^T X + λI)^-1 X^T y

    Parameters:
        xMat - feature matrix (numpy matrix)
        yMat - target vector (numpy matrix)
        lam - regularization parameter λ (shrinkage coefficient)

    Returns:
        ws - regression coefficients
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam  # Add λI to diagonal

    if np.linalg.det(denom) == 0.0:
        print("Matrix is singular, cannot compute inverse")
        return None

    ws = denom.I * (xMat.T * yMat)
    return ws


def ridge_test_multiple_lambda(xArr, yArr, num_lambda=30):
    """
    Test ridge regression with multiple λ values
    Creates regularization path

    Parameters:
        xArr - feature data (list)
        yArr - target data (list)
        num_lambda - number of λ values to test

    Returns:
        wMat - coefficient matrix (each row = coefficients for one λ)
        lambda_values - corresponding λ values tested
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # MUST standardize data before ridge regression
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat - xMeans) / xVar

    # Test multiple λ values (exponentially spaced)
    wMat = np.zeros((num_lambda, np.shape(xMat)[1]))
    lambda_values = []

    for i in range(num_lambda):
        lam = np.exp(i - 10)  # λ ranges from e^-10 to e^19
        lambda_values.append(lam)
        ws = ridge_regression(xMat, yMat, lam)
        if ws is not None:
            wMat[i, :] = ws.T

    return wMat, lambda_values


def plot_ridge_coefficients(wMat, lambda_values):
    """
    Visualize how coefficients change with λ
    Shows regularization effect
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(np.log(lambda_values), wMat)
    ax.set_xlabel('log(λ)', fontsize=12)
    ax.set_ylabel('Regression Coefficients', fontsize=12)
    ax.set_title('Ridge Regression: Coefficient Shrinkage vs. Regularization', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. LOCALLY WEIGHTED LINEAR REGRESSION (LWLR)
# =============================================================================

def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    Locally weighted linear regression
    Non-parametric regression using Gaussian kernel

    Parameters:
        testPoint - point to predict (1D array)
        xArr - training features
        yArr - training targets
        k - kernel bandwidth (controls bias-variance tradeoff)
            - Smaller k: more local fitting (higher variance, lower bias)
            - Larger k: more global fitting (lower variance, higher bias)

    Returns:
        prediction - predicted value for testPoint
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]

    # Create weight matrix (diagonal matrix with Gaussian weights)
    weights = np.mat(np.eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # Gaussian kernel: w = exp(-distance^2 / (2*k^2))
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))

    xTx = xMat.T * (weights * xMat)

    if np.linalg.det(xTx) == 0.0:
        print("Matrix is singular, cannot compute inverse")
        return None

    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlr_test(testArr, xArr, yArr, k=1.0):
    """
    Test LWLR on multiple points

    Parameters:
        testArr - test data points
        xArr - training features
        yArr - training targets
        k - kernel bandwidth

    Returns:
        yHat - predictions for all test points
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    return yHat


def compare_lwlr_bandwidth(xTrain, yTrain, xTest, yTest):
    """
    Compare LWLR performance with different bandwidth values
    Demonstrates bias-variance tradeoff
    """
    k_values = [0.01, 0.1, 1.0, 10.0]

    print("=" * 60)
    print("LWLR Bandwidth Comparison")
    print("=" * 60)

    for k in k_values:
        yHat = lwlr_test(xTest, xTrain, yTrain, k)
        rss = rss_error(yTest, yHat)
        print(f"k = {k:6.2f}  |  RSS Error = {rss:10.2f}")

    print("=" * 60)
    print("Note: Very small k may overfit, very large k may underfit")


# =============================================================================
# 4. FORWARD STAGEWISE REGRESSION (FEATURE SELECTION)
# =============================================================================

def stagewise_regression(xArr, yArr, eps=0.01, numIt=100):
    """
    Forward stagewise linear regression
    Greedy feature selection algorithm similar to Lasso

    Parameters:
        xArr - feature data
        yArr - target data
        eps - step size for each iteration
        numIt - number of iterations

    Returns:
        returnMat - coefficient matrix (each row = coefficients after iteration i)
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # Standardize data
    xMat, yMat = standardize_data(xMat, yMat)

    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))  # Initialize coefficients to zero
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        lowestError = float('inf')

        # Try adjusting each coefficient
        for j in range(n):
            # Try both increasing and decreasing
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign  # Small step in one direction

                yTest = xMat * wsTest
                rssE = rss_error(yMat.A, yTest.A)

                # Keep the change that reduces error most
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest

        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat


def plot_stagewise_coefficients(returnMat):
    """
    Visualize coefficient evolution in stagewise regression
    Shows feature selection process
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Regression Coefficients', fontsize=12)
    ax.set_title('Forward Stagewise Regression: Feature Selection Path', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. GRADIENT DESCENT FOR LOGISTIC REGRESSION
# =============================================================================

def sigmoid(inX):
    """Sigmoid activation function"""
    return 1.0 / (1 + np.exp(-inX))


def gradient_ascent(dataMatIn, classLabels, alpha=0.001, maxCycles=500):
    """
    Batch gradient ascent for logistic regression

    Parameters:
        dataMatIn - feature matrix (with intercept column)
        classLabels - binary labels (0 or 1)
        alpha - learning rate
        maxCycles - maximum iterations

    Returns:
        weights - logistic regression coefficients
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)

    weights = np.ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # Predictions
        error = labelMat - h               # Gradient direction
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights.getA()


def stochastic_gradient_ascent(dataMatrix, classLabels, numIter=150):
    """
    Improved stochastic gradient ascent with adaptive learning rate
    Faster convergence than batch gradient descent

    Parameters:
        dataMatrix - feature matrix
        classLabels - binary labels
        numIter - number of passes through dataset

    Returns:
        weights - logistic regression coefficients
    """
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)

    for j in range(numIter):
        dataIndex = list(range(m))

        for i in range(m):
            # Adaptive learning rate: decreases over time but never reaches 0
            alpha = 4 / (1.0 + j + i) + 0.01

            # Random sample selection (reduces cycles)
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights


# =============================================================================
# 6. PERFORMANCE METRICS
# =============================================================================

def rss_error(yArr, yHatArr):
    """
    Residual Sum of Squares (RSS)
    Primary metric for regression

    Parameters:
        yArr - actual values
        yHatArr - predicted values

    Returns:
        rss - sum of squared errors
    """
    return ((yArr - yHatArr)**2).sum()


def mse_error(yArr, yHatArr):
    """
    Mean Squared Error (MSE)
    """
    return np.mean((yArr - yHatArr)**2)


def rmse_error(yArr, yHatArr):
    """
    Root Mean Squared Error (RMSE)
    Same units as target variable
    """
    return np.sqrt(mse_error(yArr, yHatArr))


def r_squared(yArr, yHatArr):
    """
    R-squared (coefficient of determination)
    Proportion of variance explained by model

    Returns:
        r2 - value between 0 and 1 (higher is better)
    """
    ss_res = np.sum((yArr - yHatArr)**2)
    ss_tot = np.sum((yArr - np.mean(yArr))**2)
    return 1 - (ss_res / ss_tot)


def classification_error_rate(predictions, labels):
    """
    Error rate for classification

    Parameters:
        predictions - predicted class labels
        labels - true class labels

    Returns:
        error_rate - percentage of misclassifications
    """
    errorCount = 0
    for i in range(len(labels)):
        if predictions[i] != labels[i]:
            errorCount += 1
    return (float(errorCount) / len(labels)) * 100


def classification_accuracy(predictions, labels):
    """
    Accuracy for classification
    """
    return 100 - classification_error_rate(predictions, labels)


# =============================================================================
# 7. TRAIN/TEST SPLITTING
# =============================================================================

def train_test_split_manual(dataArr, labelArr, test_ratio=0.3):
    """
    Manual train/test split (holdout method)

    Parameters:
        dataArr - feature data
        labelArr - labels
        test_ratio - proportion of data for testing (0-1)

    Returns:
        trainData, testData, trainLabels, testLabels
    """
    m = len(dataArr)
    splitPoint = int(m * (1 - test_ratio))

    trainData = dataArr[:splitPoint]
    testData = dataArr[splitPoint:]
    trainLabels = labelArr[:splitPoint]
    testLabels = labelArr[splitPoint:]

    return trainData, testData, trainLabels, testLabels


def train_test_split_random(dataArr, labelArr, test_ratio=0.3, seed=None):
    """
    Random train/test split

    Parameters:
        dataArr - feature data
        labelArr - labels
        test_ratio - proportion of data for testing
        seed - random seed for reproducibility

    Returns:
        trainData, testData, trainLabels, testLabels
    """
    if seed is not None:
        np.random.seed(seed)

    m = len(dataArr)
    indices = np.random.permutation(m)
    splitPoint = int(m * (1 - test_ratio))

    trainIndices = indices[:splitPoint]
    testIndices = indices[splitPoint:]

    trainData = [dataArr[i] for i in trainIndices]
    testData = [dataArr[i] for i in testIndices]
    trainLabels = [labelArr[i] for i in trainIndices]
    testLabels = [labelArr[i] for i in testIndices]

    return trainData, testData, trainLabels, testLabels


# =============================================================================
# 8. MODEL VALIDATION AND COMPARISON
# =============================================================================

def compare_regression_models(xTrain, yTrain, xTest, yTest):
    """
    Compare standard regression vs. regularized methods

    Prints comparison table of different approaches
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON: REGRESSION METHODS")
    print("="*70)
    print(f"{'Method':<30} | {'Train RSS':>12} | {'Test RSS':>12}")
    print("-"*70)

    # Standard Linear Regression
    xMatTrain = np.mat(xTrain)
    yMatTrain = np.mat(yTrain).T
    xMatTest = np.mat(xTest)
    yMatTest = np.mat(yTest).T

    xTx = xMatTrain.T * xMatTrain
    if np.linalg.det(xTx) != 0.0:
        ws = xTx.I * (xMatTrain.T * yMatTrain)
        yHatTrain = xMatTrain * ws
        yHatTest = xMatTest * ws
        trainRSS = rss_error(yMatTrain.A, yHatTrain.A)
        testRSS = rss_error(yMatTest.A, yHatTest.A)
        print(f"{'Standard Linear Regression':<30} | {trainRSS:>12.2f} | {testRSS:>12.2f}")

    # Ridge Regression (multiple λ values)
    for lam in [0.01, 0.1, 1.0, 10.0]:
        ws = ridge_regression(xMatTrain, yMatTrain, lam)
        if ws is not None:
            yHatTrain = xMatTrain * ws
            yHatTest = xMatTest * ws
            trainRSS = rss_error(yMatTrain.A, yHatTrain.A)
            testRSS = rss_error(yMatTest.A, yHatTest.A)
            print(f"{'Ridge (λ=' + f'{lam:.2f}' + ')':<30} | {trainRSS:>12.2f} | {testRSS:>12.2f}")

    # LWLR (multiple bandwidth values)
    for k in [0.1, 1.0, 10.0]:
        yHatTrain = lwlr_test(xTrain, xTrain, yTrain, k)
        yHatTest = lwlr_test(xTest, xTrain, yTrain, k)
        trainRSS = rss_error(yTrain, yHatTrain)
        testRSS = rss_error(yTest, yHatTest)
        print(f"{'LWLR (k=' + f'{k:.1f}' + ')':<30} | {trainRSS:>12.2f} | {testRSS:>12.2f}")

    print("="*70)
    print("Note: Large difference between train/test RSS indicates overfitting")
    print("="*70 + "\n")


def cross_validate_regression(xArr, yArr, k_folds=5):
    """
    k-fold cross-validation for regression

    Parameters:
        xArr - feature data
        yArr - target data
        k_folds - number of folds

    Returns:
        mean_error - average error across folds
        std_error - standard deviation of errors
    """
    m = len(xArr)
    fold_size = m // k_folds
    errors = []

    for i in range(k_folds):
        # Split data into train and validation
        testStart = i * fold_size
        testEnd = (i + 1) * fold_size if i < k_folds - 1 else m

        testIndices = list(range(testStart, testEnd))
        trainIndices = list(range(0, testStart)) + list(range(testEnd, m))

        xTrain = [xArr[j] for j in trainIndices]
        yTrain = [yArr[j] for j in trainIndices]
        xTest = [xArr[j] for j in testIndices]
        yTest = [yArr[j] for j in testIndices]

        # Train and evaluate
        xMatTrain = np.mat(xTrain)
        yMatTrain = np.mat(yTrain).T
        xMatTest = np.mat(xTest)

        xTx = xMatTrain.T * xMatTrain
        if np.linalg.det(xTx) != 0.0:
            ws = xTx.I * (xMatTrain.T * yMatTrain)
            yHatTest = xMatTest * ws
            error = rss_error(yTest, yHatTest.A)
            errors.append(error)

    return np.mean(errors), np.std(errors)


# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================

def example_ridge_regression():
    """
    Example: Ridge regression with regularization path
    """
    print("\n" + "="*60)
    print("EXAMPLE: Ridge Regression with Regularization Path")
    print("="*60)

    # Generate synthetic data
    np.random.seed(42)
    m = 100  # samples
    n = 5    # features

    X = np.random.randn(m, n)
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ true_weights + np.random.randn(m) * 0.5

    # Test multiple lambda values
    wMat, lambda_values = ridge_test_multiple_lambda(X.tolist(), y.tolist(), num_lambda=30)

    print(f"\nTested {len(lambda_values)} lambda values")
    print(f"Lambda range: {min(lambda_values):.6f} to {max(lambda_values):.2f}")
    print(f"\nTrue weights:     {true_weights}")
    print(f"Ridge (λ=0.001):  {wMat[1, :]}")  # Low regularization
    print(f"Ridge (λ=1.0):    {wMat[11, :]}")  # Medium regularization
    print(f"Ridge (λ=100):    {wMat[21, :]}")  # High regularization

    # Plot coefficient paths
    plot_ridge_coefficients(wMat, lambda_values)


def example_lwlr_comparison():
    """
    Example: LWLR bandwidth selection
    """
    print("\n" + "="*60)
    print("EXAMPLE: Locally Weighted Linear Regression")
    print("="*60)

    # Generate synthetic non-linear data
    np.random.seed(42)
    m = 100
    x = np.linspace(0, 10, m)
    y = np.sin(x) + np.random.randn(m) * 0.2

    X = np.column_stack([np.ones(m), x])  # Add intercept

    # Split train/test
    xTrain, xTest, yTrain, yTest = train_test_split_manual(
        X.tolist(), y.tolist(), test_ratio=0.3
    )

    # Compare different bandwidth values
    compare_lwlr_bandwidth(xTrain, yTrain, xTest, yTest)


def example_feature_selection():
    """
    Example: Forward stagewise regression for feature selection
    """
    print("\n" + "="*60)
    print("EXAMPLE: Forward Stagewise Regression")
    print("="*60)

    # Generate synthetic data with some irrelevant features
    np.random.seed(42)
    m = 100
    n = 10

    X = np.random.randn(m, n)
    # Only first 3 features are relevant
    true_weights = np.array([2.0, -1.5, 3.0, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_weights + np.random.randn(m) * 0.5

    # Run stagewise regression
    returnMat = stagewise_regression(X.tolist(), y.tolist(), eps=0.01, numIt=200)

    print(f"\nTrue weights (only first 3 non-zero):")
    print(true_weights)
    print(f"\nStagewise regression (final iteration):")
    print(returnMat[-1, :])

    # Plot coefficient evolution
    plot_stagewise_coefficients(returnMat)


if __name__ == '__main__':
    """
    Run examples to demonstrate techniques
    """
    print("\n" + "="*70)
    print("JACK-CHERISH ML REPOSITORY: IMPLEMENTATION EXAMPLES")
    print("="*70)

    # Uncomment to run specific examples:
    # example_ridge_regression()
    # example_lwlr_comparison()
    # example_feature_selection()

    print("\nTo run examples, uncomment the function calls in __main__")
    print("="*70 + "\n")
