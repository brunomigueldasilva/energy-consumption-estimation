#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - MODEL TRAINING
==============================================================================

Purpose: Train multiple regression models for monthly bill prediction

This script:
1. Loads preprocessed train/test data (X_train, X_test, y_train, y_test)
2. Trains 9 different regression models with various approaches:
   - Baseline: Simple linear with 1 feature
   - Linear: Multiple linear regression (all features)
   - Ridge & Lasso: Regularized linear models with hyperparameter tuning
   - Polynomial: Non-linear feature expansion (degree 2)
   - SVR: Support Vector Regression (linear and RBF kernels)
   - Random Forest: Ensemble tree-based method
3. Saves trained models and predictions
4. Records training times and model characteristics
5. Does NOT calculate metrics (deferred to evaluation script)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import time
import json
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Model training configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    PROCESSED_DATA_DIR = OUTPUT_DIR / 'data_processed'
    MODELS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Input files
    X_TRAIN_FILE = PROCESSED_DATA_DIR / 'X_train.npy'
    X_TEST_FILE = PROCESSED_DATA_DIR / 'X_test.npy'
    Y_TRAIN_FILE = PROCESSED_DATA_DIR / 'y_train.npy'
    Y_TEST_FILE = PROCESSED_DATA_DIR / 'y_test.npy'
    TRANSFORMER_FILE = PROCESSED_DATA_DIR / 'col_transformer.pkl'
    FEATURE_NAMES_FILE = PROCESSED_DATA_DIR / 'feature_names_after_transform.json'

    # Output files
    TRAINING_TIMES_FILE = MODELS_DIR / 'treino_tempos.json'

    # Model parameters
    RANDOM_STATE = 42
    RIDGE_LASSO_ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100]
    GRID_CV_FOLDS = 5
    RF_N_ESTIMATORS = 300
    RF_MIN_SAMPLES_LEAF = 2


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def save_model(model: Any, filename: str) -> None:
    """
    Save model to pickle file.

    Args:
        model: Trained model object
        filename: Name of the pickle file (without path)
    """
    filepath = Config.MODELS_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    # Get file size
    file_size = filepath.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024**2:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024**2):.2f} MB"

    print(f"  ✓ Model saved: {filename} ({size_str})")


def save_predictions(y_pred: np.ndarray, filename: str) -> None:
    """
    Save predictions to numpy file.

    Args:
        y_pred: Predictions array
        filename: Name of the .npy file (without path)
    """
    filepath = Config.PREDICTIONS_DIR / filename
    np.save(filepath, y_pred)
    print(f"  ✓ Predictions saved: {filename} (shape: {y_pred.shape})")


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load preprocessed training and test data.

    Returns:
        Tuple containing:
        - X_train: Training features (transformed)
        - X_test: Test features (transformed)
        - y_train: Training target
        - y_test: Test target
        - feature_names: List of feature names after transformation
    """
    print_section("1. LOADING PREPROCESSED DATA")

    # Check if files exist
    required_files = [
        Config.X_TRAIN_FILE,
        Config.X_TEST_FILE,
        Config.Y_TRAIN_FILE,
        Config.Y_TEST_FILE,
        Config.FEATURE_NAMES_FILE
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")

    # Load numpy arrays
    print(f"Loading data from: {Config.PROCESSED_DATA_DIR}")
    X_train = np.load(Config.X_TRAIN_FILE)
    X_test = np.load(Config.X_TEST_FILE)
    y_train = np.load(Config.Y_TRAIN_FILE)
    y_test = np.load(Config.Y_TEST_FILE)

    print("✓ Loaded arrays:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    # Load feature names
    with open(Config.FEATURE_NAMES_FILE, 'r') as f:
        feature_info = json.load(f)

    feature_names = feature_info['feature_names']
    print(f"\n✓ Loaded {len(feature_names)} feature names")
    print(f"  First 5: {feature_names[:5]}")
    print(f"  Last 5: {feature_names[-5:]}")

    # Data validation
    print("\n✓ Data validation:")
    print(
        f"  X_train - NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(
        f"  X_test - NaN: {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")
    print(
        f"  y_train - NaN: {np.isnan(y_train).sum()}, Inf: {np.isinf(y_train).sum()}")
    print(
        f"  y_test - NaN: {np.isnan(y_test).sum()}, Inf: {np.isinf(y_test).sum()}")

    return X_train, X_test, y_train, y_test, feature_names


# ==============================================================================
# SECTION 4: BASELINE MODEL (1 FEATURE)
# ==============================================================================


def train_baseline_model(X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         feature_names: list) -> Dict[str, Any]:
    """
    Train baseline linear regression with only 1 feature:
    consumo_acumulado_ate_corte_total_kwh

    This provides a simple baseline to compare against more complex models.

    Args:
        X_train: Training features (all)
        X_test: Test features (all)
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names

    Returns:
        Dictionary with model info and training time
    """
    print_section("2. TRAINING BASELINE MODEL (1 FEATURE)")

    print("📊 BASELINE STRATEGY:")
    print("  - Use ONLY 'consumo_acumulado_ate_corte_total_kwh' feature")
    print("  - Simple linear relationship: y = a*x + b")
    print("  - Provides minimum performance threshold")
    print("  - Should be easily beaten by more complex models")

    # Find the index of the baseline feature
    baseline_feature = 'consumo_acumulado_ate_corte_total_kwh'

    try:
        feature_idx = feature_names.index(baseline_feature)
        print(
            f"\n✓ Found baseline feature: '{baseline_feature}' at index {feature_idx}")
    except ValueError:
        print(
            f"\n✗ ERROR: Feature '{baseline_feature}' not found in feature names!")
        print(f"  Available features: {feature_names}")
        raise

    # Extract single feature
    X_train_baseline = X_train[:, feature_idx].reshape(-1, 1)
    X_test_baseline = X_test[:, feature_idx].reshape(-1, 1)

    print(f"  X_train_baseline shape: {X_train_baseline.shape}")
    print(f"  X_test_baseline shape: {X_test_baseline.shape}")

    # Train model
    print("\n▶ Training LinearRegression (1 feature)...")
    start_time = time.time()

    model = LinearRegression()
    model.fit(X_train_baseline, y_train)

    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"  Coefficient: {model.coef_[0]:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = model.predict(X_train_baseline)
    y_pred_test = model.predict(X_test_baseline)

    # Save model and predictions
    save_model(model, 'model_linear_baseline.pkl')
    save_predictions(y_pred_train, 'y_pred_linear_baseline_train.npy')
    save_predictions(y_pred_test, 'y_pred_linear_baseline_test.npy')

    return {
        'model_name': 'Linear Baseline (1 feature)',
        'training_time': training_time,
        'n_features': 1,
        'n_coefficients': 1,
        'notes': 'Simplest possible model - uses only total accumulated consumption'}


# ==============================================================================
# SECTION 5: LINEAR REGRESSION (ALL FEATURES)
# ==============================================================================


def train_linear_model(X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray) -> Dict[str,
                                                   Any]:
    """
    Train multiple linear regression with all features.

    Uses all transformed features to fit a linear model.
    Assumes linear relationship between features and target.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("3. TRAINING LINEAR REGRESSION (ALL FEATURES)")

    print("📊 LINEAR REGRESSION STRATEGY:")
    print("  - Uses ALL transformed features")
    print("  - Assumes linear relationships")
    print("  - Fast training, interpretable coefficients")
    print("  - Good when: relationships are truly linear")
    print("  - Limitation: Cannot capture non-linear patterns")

    print(f"\n▶ Training LinearRegression with {X_train.shape[1]} features...")
    start_time = time.time()

    model = LinearRegression()
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"  Number of coefficients: {len(model.coef_)}")
    print(f"  Intercept: {model.intercept_:.6f}")

    # Show top 5 coefficients by magnitude
    coef_abs = np.abs(model.coef_)
    top_5_idx = np.argsort(coef_abs)[-5:][::-1]
    print("\n  Top 5 coefficients by magnitude:")
    for i, idx in enumerate(top_5_idx, 1):
        print(f"    {i}. Feature {idx}: {model.coef_[idx]:.6f}")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Save model and predictions
    save_model(model, 'model_linear_multiple.pkl')
    save_predictions(y_pred_train, 'y_pred_linear_multiple_train.npy')
    save_predictions(y_pred_test, 'y_pred_linear_multiple_test.npy')

    return {
        'model_name': 'Linear Regression (multiple)',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_coefficients': len(model.coef_),
        'notes': 'Standard linear regression with all features'
    }


# ==============================================================================
# SECTION 6: RIDGE REGRESSION (L2 REGULARIZATION)
# ==============================================================================


def train_ridge_model(X_train: np.ndarray,
                      X_test: np.ndarray,
                      y_train: np.ndarray,
                      y_test: np.ndarray) -> Dict[str,
                                                  Any]:
    """
    Train Ridge regression with L2 regularization and hyperparameter tuning.

    Ridge adds penalty term: Loss = MSE + alpha * sum(coef^2)
    - Shrinks coefficients toward zero (but never exactly zero)
    - Prevents overfitting when features are correlated
    - Keeps all features in the model

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("4. TRAINING RIDGE REGRESSION (L2 REGULARIZATION)")

    print("📊 RIDGE REGRESSION STRATEGY:")
    print("  - L2 regularization: penalizes large coefficients")
    print("  - Shrinks coefficients toward zero (but not to zero)")
    print("  - KEEPS all features (no feature elimination)")
    print("  - Good when: Many correlated features exist")
    print("  - Good when: Want to prevent overfitting while keeping all predictors")
    print("  - Hyperparameter: alpha (controls regularization strength)")

    print("\n▶ Hyperparameter tuning with GridSearchCV...")
    print(f"  Alpha values to test: {Config.RIDGE_LASSO_ALPHAS}")
    print(f"  Cross-validation folds: {Config.GRID_CV_FOLDS}")
    print("  Scoring: neg_mean_squared_error")

    start_time = time.time()

    # Grid search for best alpha
    ridge = Ridge(random_state=Config.RANDOM_STATE)
    param_grid = {'alpha': Config.RIDGE_LASSO_ALPHAS}

    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=Config.GRID_CV_FOLDS,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"\n  Best alpha: {best_alpha}")
    print(f"  Best CV score (neg_MSE): {grid_search.best_score_:.4f}")
    print(f"  Number of coefficients: {len(best_model.coef_)}")
    print(f"  Intercept: {best_model.intercept_:.6f}")

    # Show coefficient statistics
    coef_abs = np.abs(best_model.coef_)
    print("\n  Coefficient statistics:")
    print(f"    Mean |coef|: {coef_abs.mean():.6f}")
    print(f"    Max |coef|: {coef_abs.max():.6f}")
    print(f"    Min |coef|: {coef_abs.min():.6f}")

    # Make predictions
    print("\n▶ Making predictions with best model...")
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Save model and predictions
    save_model(best_model, 'model_ridge.pkl')
    save_predictions(y_pred_train, 'y_pred_ridge_train.npy')
    save_predictions(y_pred_test, 'y_pred_ridge_test.npy')

    # Save grid search results
    grid_results = {
        'best_alpha': float(best_alpha),
        'best_score': float(grid_search.best_score_),
        'all_scores': {float(alpha): float(score)
                       for alpha, score in zip(
            Config.RIDGE_LASSO_ALPHAS,
            grid_search.cv_results_['mean_test_score']
        )}
    }
    with open(Config.MODELS_DIR / 'ridge_grid_results.json', 'w') as f:
        json.dump(grid_results, f, indent=2)
    print("  ✓ Grid search results saved: ridge_grid_results.json")

    return {
        'model_name': 'Ridge Regression',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_coefficients': len(best_model.coef_),
        'best_alpha': best_alpha,
        'notes': 'L2 regularization - prevents overfitting with correlated features'
    }


# ==============================================================================
# SECTION 7: LASSO REGRESSION (L1 REGULARIZATION)
# ==============================================================================


def train_lasso_model(X_train: np.ndarray,
                      X_test: np.ndarray,
                      y_train: np.ndarray,
                      y_test: np.ndarray) -> Dict[str,
                                                  Any]:
    """
    Train Lasso regression with L1 regularization and hyperparameter tuning.

    Lasso adds penalty term: Loss = MSE + alpha * sum(|coef|)
    - Forces some coefficients to EXACTLY zero
    - Performs automatic feature selection
    - Creates sparse models (fewer features)

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("5. TRAINING LASSO REGRESSION (L1 REGULARIZATION)")

    print("📊 LASSO REGRESSION STRATEGY:")
    print("  - L1 regularization: penalizes absolute value of coefficients")
    print("  - Forces coefficients to EXACTLY zero (feature elimination)")
    print("  - AUTOMATIC FEATURE SELECTION")
    print("  - Creates sparse models (uses fewer features)")
    print("  - Good when: Many features are irrelevant or redundant")
    print("  - Good when: Want interpretable model with fewer predictors")
    print("  - Hyperparameter: alpha (controls sparsity)")

    print("\n▶ Hyperparameter tuning with GridSearchCV...")
    print(f"  Alpha values to test: {Config.RIDGE_LASSO_ALPHAS}")
    print(f"  Cross-validation folds: {Config.GRID_CV_FOLDS}")
    print("  Scoring: neg_mean_squared_error")

    start_time = time.time()

    # Grid search for best alpha
    lasso = Lasso(random_state=Config.RANDOM_STATE, max_iter=10000)
    param_grid = {'alpha': Config.RIDGE_LASSO_ALPHAS}

    grid_search = GridSearchCV(
        lasso,
        param_grid,
        cv=Config.GRID_CV_FOLDS,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    # Count non-zero coefficients (selected features)
    n_nonzero = np.sum(best_model.coef_ != 0)
    n_zero = np.sum(best_model.coef_ == 0)

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"\n  Best alpha: {best_alpha}")
    print(f"  Best CV score (neg_MSE): {grid_search.best_score_:.4f}")
    print(f"  Intercept: {best_model.intercept_:.6f}")

    print("\n  🔍 FEATURE SELECTION RESULTS:")
    print(f"    Total features: {len(best_model.coef_)}")
    print(f"    Features SELECTED (non-zero): {n_nonzero}")
    print(f"    Features ELIMINATED (zero): {n_zero}")
    print(f"    Sparsity: {n_zero / len(best_model.coef_) * 100:.1f}%")

    # Show non-zero coefficients
    if n_nonzero > 0 and n_nonzero <= 10:
        print("\n  Non-zero coefficients:")
        for i, coef in enumerate(best_model.coef_):
            if coef != 0:
                print(f"    Feature {i}: {coef:.6f}")
    elif n_nonzero > 10:
        nonzero_idx = np.nonzero(best_model.coef_ != 0)[0]
        coef_nonzero = best_model.coef_[nonzero_idx]
        top_5_idx = nonzero_idx[np.argsort(np.abs(coef_nonzero))[-5:][::-1]]
        print("\n  Top 5 non-zero coefficients:")
        for i, idx in enumerate(top_5_idx, 1):
            print(f"    {i}. Feature {idx}: {best_model.coef_[idx]:.6f}")

    # Make predictions
    print("\n▶ Making predictions with best model...")
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Save model and predictions
    save_model(best_model, 'model_lasso.pkl')
    save_predictions(y_pred_train, 'y_pred_lasso_train.npy')
    save_predictions(y_pred_test, 'y_pred_lasso_test.npy')

    # Save grid search results
    grid_results = {
        'best_alpha': float(best_alpha),
        'best_score': float(grid_search.best_score_),
        'n_features_selected': int(n_nonzero),
        'n_features_eliminated': int(n_zero),
        'sparsity_percent': float(n_zero / len(best_model.coef_) * 100),
        'all_scores': {float(alpha): float(score)
                       for alpha, score in zip(
            Config.RIDGE_LASSO_ALPHAS,
            grid_search.cv_results_['mean_test_score']
        )}
    }
    with open(Config.MODELS_DIR / 'lasso_grid_results.json', 'w') as f:
        json.dump(grid_results, f, indent=2)
    print("  ✓ Grid search results saved: lasso_grid_results.json")

    return {
        'model_name': 'Lasso Regression',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_coefficients_nonzero': n_nonzero,
        'n_features_eliminated': n_zero,
        'best_alpha': best_alpha,
        'notes': 'L1 regularization - automatic feature selection, sparse models'}


# ==============================================================================
# SECTION 8: POLYNOMIAL REGRESSION (DEGREE 2)
# ==============================================================================


def train_polynomial_model(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray) -> Dict[str,
                                                       Any]:
    """
    Train polynomial regression (degree 2) using pipeline.

    ⚠️ WARNING: Polynomial features can cause OVERFITTING!
    - Degree 2 creates ~n*(n+1)/2 new features (interactions + squares)
    - With 25 features → ~325 polynomial features
    - Model becomes much more complex
    - Risk: Great training performance, poor test performance

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("6. TRAINING POLYNOMIAL REGRESSION (DEGREE 2)")

    print("📊 POLYNOMIAL REGRESSION STRATEGY:")
    print("  - Creates polynomial features: x, x², x*y, etc.")
    print("  - Captures NON-LINEAR relationships")
    print("  - Captures FEATURE INTERACTIONS")
    print("  - Good when: Relationships are quadratic or have interactions")

    print("\n⚠️  OVERFITTING WARNING:")
    print("  - Polynomial degree 2 dramatically increases feature count")
    print(f"  - Original features: {X_train.shape[1]}")
    print(
        f"  - Estimated polynomial features: ~{X_train.shape[1] * (X_train.shape[1] + 1) // 2}")
    print("  - Risk: Model memorizes training data")
    print("  - Watch for: Large gap between train and test performance")
    print("  - Mitigation: Use regularization (Ridge/Lasso) with polynomial features")

    print("\n▶ Creating polynomial pipeline...")
    start_time = time.time()

    # Create pipeline: PolynomialFeatures → LinearRegression
    pipeline = Pipeline([('poly_features',
                          PolynomialFeatures(degree=2,
                                             include_bias=False,
                                             interaction_only=False)),
                         ('linear_model',
                          LinearRegression())])

    print("  Pipeline steps:")
    print("    1. PolynomialFeatures(degree=2, include_bias=False)")
    print("    2. LinearRegression()")

    print("\n▶ Training polynomial model...")
    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Get the number of polynomial features created
    n_poly_features = pipeline.named_steps['poly_features'].n_output_features_
    linear_model = pipeline.named_steps['linear_model']

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print("\n  Feature expansion:")
    print(f"    Original features: {X_train.shape[1]}")
    print(f"    Polynomial features: {n_poly_features}")
    print(f"    Expansion ratio: {n_poly_features / X_train.shape[1]:.1f}x")
    print("  Model parameters:")
    print(f"    Number of coefficients: {len(linear_model.coef_)}")
    print(f"    Intercept: {linear_model.intercept_:.6f}")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Save model and predictions
    save_model(pipeline, 'model_polynomial_deg2.pkl')
    save_predictions(y_pred_train, 'y_pred_polynomial_deg2_train.npy')
    save_predictions(y_pred_test, 'y_pred_polynomial_deg2_test.npy')

    print("\n  💡 INTERPRETATION NOTE:")
    print("    - If test performance >> worse than train: OVERFITTING!")
    print("    - Solution: Use Ridge/Lasso polynomial or reduce degree")
    print("    - Alternative: Use Random Forest (handles non-linearity better)")

    return {
        'model_name': 'Polynomial Regression (degree 2)',
        'training_time': training_time,
        'n_features_original': X_train.shape[1],
        'n_features_polynomial': n_poly_features,
        'n_coefficients': len(linear_model.coef_),
        'notes': 'Non-linear model via polynomial expansion - HIGH RISK OF OVERFITTING!'
    }


# ==============================================================================
# SECTION 9: SUPPORT VECTOR REGRESSION (LINEAR)
# ==============================================================================


def train_svr_linear_model(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray) -> Dict[str,
                                                       Any]:
    """
    Train Support Vector Regression with linear kernel.

    SVR with linear kernel is similar to linear regression but:
    - Uses epsilon-insensitive loss (tolerates small errors)
    - Optimizes margin instead of least squares
    - Can be more robust to outliers

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("7. TRAINING SVR (LINEAR KERNEL)")

    print("📊 SVR LINEAR STRATEGY:")
    print("  - Support Vector Machine for regression")
    print("  - Linear kernel: similar to linear regression")
    print("  - Epsilon-insensitive loss: ignores small errors")
    print("  - Can be more ROBUST to outliers than OLS")
    print("  - Good when: Want linear model with outlier robustness")
    print("  - Limitation: Slower than LinearRegression for large datasets")

    print("\n▶ Training SVR(kernel='linear')...")
    start_time = time.time()

    model = SVR(kernel='linear', C=1.0, gamma='scale')
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"  Number of support vectors: {len(model.support_)}")
    print(
        f"  Support vector ratio: {len(model.support_) / len(y_train) * 100:.1f}%")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Save model and predictions
    save_model(model, 'model_svr_linear.pkl')
    save_predictions(y_pred_train, 'y_pred_svr_linear_train.npy')
    save_predictions(y_pred_test, 'y_pred_svr_linear_test.npy')

    return {
        'model_name': 'SVR (linear kernel)',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_support_vectors': len(model.support_),
        'notes': 'Linear SVR - robust to outliers, epsilon-insensitive loss'
    }


# ==============================================================================
# SECTION 10: SUPPORT VECTOR REGRESSION (RBF)
# ==============================================================================


def train_svr_rbf_model(X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray) -> Dict[str,
                                                    Any]:
    """
    Train Support Vector Regression with RBF (Radial Basis Function) kernel.

    SVR with RBF kernel:
    - Can capture COMPLEX NON-LINEAR relationships
    - Uses Gaussian kernel to map features to higher dimensions
    - Very flexible but computationally expensive
    - Risk of overfitting if C is too high

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("8. TRAINING SVR (RBF KERNEL)")

    print("📊 SVR RBF STRATEGY:")
    print("  - Radial Basis Function (Gaussian) kernel")
    print("  - Captures COMPLEX NON-LINEAR patterns")
    print("  - Maps data to infinite-dimensional space")
    print("  - Very flexible but computationally expensive")
    print("  - Good when: Relationships are highly non-linear")
    print("  - Limitation: Slower training, harder to interpret")
    print("  - Hyperparameters: C (regularization), gamma (kernel width)")

    print("\n▶ Training SVR(kernel='rbf', C=10, gamma='scale')...")
    start_time = time.time()

    model = SVR(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"  Number of support vectors: {len(model.support_)}")
    print(
        f"  Support vector ratio: {len(model.support_) / len(y_train) * 100:.1f}%")
    print(f"  C parameter: {model.C}")
    print(
        f"  Gamma: {
            model.gamma if isinstance(
                model.gamma, (int, float)) else 'scale'}")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Save model and predictions
    save_model(model, 'model_svr_rbf.pkl')
    save_predictions(y_pred_train, 'y_pred_svr_rbf_train.npy')
    save_predictions(y_pred_test, 'y_pred_svr_rbf_test.npy')

    print("\n  💡 NOTE:")
    print("    - If support vector ratio is high (>50%): model is complex")
    print("    - May benefit from hyperparameter tuning (C, gamma)")
    print("    - Watch for overfitting in evaluation metrics")

    return {
        'model_name': 'SVR (RBF kernel)',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_support_vectors': len(model.support_),
        'C': float(model.C),
        'notes': 'Non-linear SVR with RBF kernel - captures complex patterns'
    }


# ==============================================================================
# SECTION 11: RANDOM FOREST REGRESSION
# ==============================================================================


def train_random_forest_model(X_train: np.ndarray,
                              X_test: np.ndarray,
                              y_train: np.ndarray,
                              y_test: np.ndarray) -> Dict[str,
                                                          Any]:
    """
    Train Random Forest Regressor.

    Random Forest:
    - Ensemble of decision trees
    - Each tree trained on random subset of data and features
    - Predictions averaged across all trees
    - Excellent for non-linear relationships
    - Provides feature importance scores
    - Robust to outliers and overfitting

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with model info and training time
    """
    print_section("9. TRAINING RANDOM FOREST")

    print("📊 RANDOM FOREST STRATEGY:")
    print("  - Ensemble of decision trees")
    print("  - Each tree: random data subset + random feature subset")
    print("  - Final prediction: AVERAGE of all tree predictions")
    print("  - Captures COMPLEX NON-LINEAR relationships")
    print("  - Handles FEATURE INTERACTIONS naturally")
    print("  - ROBUST to outliers and noise")
    print("  - Provides FEATURE IMPORTANCE rankings")
    print("  - Good when: Relationships are complex and non-linear")
    print("  - Good when: Don't know which features are important")
    print("  - Limitation: Slower prediction, less interpretable")

    print("\n▶ Training RandomForestRegressor...")
    print("  Parameters:")
    print(f"    n_estimators: {Config.RF_N_ESTIMATORS} (number of trees)")
    print(
        f"    min_samples_leaf: {
            Config.RF_MIN_SAMPLES_LEAF} (regularization)")
    print(f"    random_state: {Config.RANDOM_STATE}")
    print("    n_jobs: -1 (use all CPU cores)")

    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=Config.RF_N_ESTIMATORS,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        max_features=1.0,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.4f} seconds")
    print(f"  Number of trees: {model.n_estimators}")

    # Get max depth (handle None values)
    depths = [tree.tree_.max_depth for tree in model.estimators_]
    max_depth = max(d for d in depths if d is not None) if any(
        d is not None for d in depths) else "unlimited"
    print(f"  Max depth: {max_depth}")
    print(
        f"  Total nodes: {sum(tree.tree_.node_count for tree in model.estimators_):,}")

    # Feature importance
    feature_importance = model.feature_importances_
    top_5_idx = np.argsort(feature_importance)[-5:][::-1]

    print("\n  🔍 TOP 5 MOST IMPORTANT FEATURES:")
    for i, idx in enumerate(top_5_idx, 1):
        print(
            f"    {i}. Feature {idx}: importance = {
                feature_importance[idx]:.6f}")

    # Make predictions
    print("\n▶ Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Save model and predictions
    save_model(model, 'model_random_forest.pkl')
    save_predictions(y_pred_train, 'y_pred_random_forest_train.npy')
    save_predictions(y_pred_test, 'y_pred_random_forest_test.npy')

    # Save feature importance
    importance_dict = {
        'feature_indices': list(range(len(feature_importance))),
        'importances': feature_importance.tolist()
    }
    with open(Config.MODELS_DIR / 'random_forest_feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)
    print("  ✓ Feature importance saved: random_forest_feature_importance.json")

    return {
        'model_name': 'Random Forest',
        'training_time': training_time,
        'n_features': X_train.shape[1],
        'n_estimators': Config.RF_N_ESTIMATORS,
        'max_depth': str(max_depth),
        'notes': 'Ensemble of trees - excellent for non-linear patterns and feature importance'
    }


# ==============================================================================
# SECTION 12: SAVE TRAINING SUMMARY
# ==============================================================================


def save_training_summary(training_results: list) -> None:
    """
    Save comprehensive training summary to JSON file.

    Args:
        training_results: List of dictionaries with model training info
    """
    print_section("10. SAVING TRAINING SUMMARY")

    # Sort by training time
    training_results_sorted = sorted(
        training_results,
        key=lambda x: x['training_time'])

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    # Create summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models': len(training_results),
        'total_training_time': sum(
            r['training_time'] for r in training_results),
        'models': convert_to_native(training_results_sorted)}

    # Save to JSON
    with open(Config.TRAINING_TIMES_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Training summary saved: {Config.TRAINING_TIMES_FILE}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("TRAINING TIME SUMMARY (sorted by speed)")
    print(f"{'=' * 80}")
    print(f"{'Model':<40} {'Time (s)':<12} {'Notes':<30}")
    print(f"{'-' * 80}")

    for result in training_results_sorted:
        model_name = result['model_name']
        training_time = result['training_time']
        # Get first 30 chars of notes
        notes = result.get('notes', '')[:30]
        print(f"{model_name:<40} {training_time:<12.4f} {notes:<30}")

    print(f"{'-' * 80}")
    print(f"{'TOTAL':<40} {summary['total_training_time']:<12.4f}")
    print(f"{'=' * 80}")

    # Model recommendations
    print(f"\n{'=' * 80}")
    print("MODEL SELECTION GUIDE")
    print(f"{'=' * 80}")
    print("""
When to use each model:

1. LINEAR BASELINE (1 feature)
   → Quick baseline, interpretable, minimum acceptable performance

2. LINEAR REGRESSION (all features)
   → Fast, interpretable, good for linear relationships
   → Start here if relationships seem linear

3. RIDGE
   → Many correlated features (multicollinearity)
   → Want to keep all features but prevent overfitting
   → Similar to linear but more stable

4. LASSO
   → Want automatic feature selection
   → Suspect many features are irrelevant
   → Need sparse, interpretable model

5. POLYNOMIAL (degree 2)
   → Relationships are quadratic
   → Need to capture feature interactions
   → ⚠️ Watch for overfitting!

6. SVR LINEAR
   → Want robustness to outliers
   → Linear relationships but with noise
   → Alternative to OLS with better outlier handling

7. SVR RBF
   → Highly complex non-linear relationships
   → Don't know the functional form
   → Willing to sacrifice interpretability for accuracy

8. RANDOM FOREST
   → Complex non-linear patterns
   → Many feature interactions
   → Want feature importance rankings
   → Most reliable for unknown relationships

RECOMMENDATION FOR THIS PROBLEM:
- Start with: Linear Regression (baseline)
- If underfits: Try Random Forest or SVR RBF
- If overfits: Try Ridge or Lasso
- For production: Random Forest (robust) or best regularized linear model
""")
    print(f"{'=' * 80}")


# ==============================================================================
# SECTION 13: MAIN EXECUTION
# ==============================================================================


def main():
    """Main training pipeline execution."""
    print("=" * 80)
    print("ELECTRICITY BILL PREDICTION - MODEL TRAINING")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create output directory
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output Models directory: {Config.MODELS_DIR}")
        Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output Predictions directory: {Config.PREDICTIONS_DIR}")

        # 1. Load data
        X_train, X_test, y_train, y_test, feature_names = load_data()

        # 2. Train all models
        training_results = []

        # Model 1: Baseline (1 feature)
        result = train_baseline_model(
            X_train, X_test, y_train, y_test, feature_names)
        training_results.append(result)

        # Model 2: Linear Multiple
        result = train_linear_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 3: Ridge
        result = train_ridge_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 4: Lasso
        result = train_lasso_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 5: Polynomial
        result = train_polynomial_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 6: SVR Linear
        result = train_svr_linear_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 7: SVR RBF
        result = train_svr_rbf_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # Model 8: Random Forest
        result = train_random_forest_model(X_train, X_test, y_train, y_test)
        training_results.append(result)

        # 3. Save training summary
        save_training_summary(training_results)

        print("\n" + "=" * 80)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nNext steps:")
        print(f"  1. Review training summary: {Config.TRAINING_TIMES_FILE}")
        print("  2. Run evaluation script to calculate metrics (MAE, RMSE, R², MAPE)")
        print("  3. Compare model performances")
        print("  4. Analyze residuals and prediction errors")
        print("  5. Select best model for production")

    except Exception as e:
        print("\n✗ ERROR during training:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
