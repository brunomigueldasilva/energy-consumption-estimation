#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - METRICS EVALUATION
==============================================================================

Purpose: Calculate and compare performance metrics for all trained models

This script:
1. Automatically discovers all model predictions (y_pred_*.npy)
2. Loads ground truth labels (y_test.npy)
3. Calculates 4 key regression metrics: R², MAE, MSE, RMSE
4. Creates comparative table sorted by RMSE (best performance first)
5. Highlights best performing model
6. Saves results to CSV and Markdown formats
7. Provides detailed metric interpretations and recommendations

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, List
import textwrap

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Evaluation configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    PROCESSED_DATA_DIR = OUTPUT_DIR / 'data_processed'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
    RESULTS_DIR = OUTPUT_DIR / 'results'

    # Input files
    Y_TEST_FILE = PROCESSED_DATA_DIR / 'y_test.npy'

    # Output files
    METRICS_CSV = RESULTS_DIR / 'metricas.csv'
    METRICS_MD = RESULTS_DIR / 'metricas.md'

    # Formatting
    DECIMAL_PLACES = 4

    # Thresholds for warnings
    LOW_R2_THRESHOLD = 0.5  # Warn if R² below this
    HIGH_RMSE_RATIO = 1.5    # Warn if RMSE/MAE ratio is high


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def print_wrapped(text: str, indent: int = 0) -> None:
    """
    Print text with word wrapping and indentation.

    Args:
        text: Text to print
        indent: Number of spaces to indent
    """
    wrapper = textwrap.TextWrapper(
        width=80 - indent,
        initial_indent=' ' * indent,
        subsequent_indent=' ' * indent
    )
    print(wrapper.fill(text))


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_ground_truth() -> np.ndarray:
    """
    Load ground truth test labels.

    Returns:
        np.ndarray: True test labels

    Raises:
        SystemExit: If ground truth file not found
    """
    print_section("1. LOADING GROUND TRUTH")

    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Ground truth file not found: {Config.Y_TEST_FILE}")
        print("  Please run 02_preprocessamento.py first.")
        exit(1)

    y_test = np.load(Config.Y_TEST_FILE)

    print(f"✓ Loaded ground truth: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Type: {y_test.dtype}")

    # Show target statistics
    print("\n📊 Target distribution (y_test):")
    print(f"  Mean:   {y_test.mean():.4f}")
    print(f"  Std:    {y_test.std():.4f}")
    print(f"  Min:    {y_test.min():.4f}")
    print(f"  Max:    {y_test.max():.4f}")
    print(f"  Median: {np.median(y_test):.4f}")

    # Calculate skewness (simple implementation)
    from scipy import stats
    try:
        skewness = stats.skew(y_test)
        print(f"  Skewness: {skewness:.4f}")

        if abs(skewness) > 1.0:
            print(f"\n  ⚠️  High skewness detected (|{skewness:.2f}| > 1.0)")
            print("      → Target distribution is heavily skewed")
            print("      → Consider log transformation: y_log = log1p(y)")
            print("      → This may improve model performance")
    except ImportError:
        # If scipy not available, calculate simple skewness
        mean = y_test.mean()
        std = y_test.std()
        skewness = np.mean(((y_test - mean) / std) ** 3)
        print(f"  Skewness (approx): {skewness:.4f}")

    return y_test


def discover_prediction_files() -> List[Path]:
    """
    Automatically discover all prediction files (y_pred_*.npy).

    Returns:
        List[Path]: List of prediction file paths

    Raises:
        SystemExit: If no prediction files found
    """
    print_section("2. DISCOVERING PREDICTION FILES")

    if not Config.PREDICTIONS_DIR.exists():
        print(
            f"✗ ERROR: Models prediction directory not found: {
                Config.PREDICTIONS_DIR}")
        print("  Please run 03_treino_modelos.py first.")
        exit(1)

    # Find all y_pred_*_test.npy files
    prediction_files = list(Config.PREDICTIONS_DIR.glob('y_pred_*_test.npy'))

    if len(prediction_files) == 0:
        print(
            f"✗ ERROR: No prediction files found in {
                Config.PREDICTIONS_DIR}")
        print("  Expected pattern: y_pred_*_test.npy")
        print("  Please run 03_treino_modelos.py first.")
        exit(1)

    # Sort alphabetically
    prediction_files.sort()

    print(f"✓ Found {len(prediction_files)} prediction files:\n")
    for i, filepath in enumerate(prediction_files, 1):
        print(f"  {i}. {filepath.name}")

    return prediction_files


def extract_model_name(filepath: Path) -> str:
    """
    Extract model name from prediction file path.

    Args:
        filepath: Path to prediction file (e.g., y_pred_linear_baseline_test.npy)

    Returns:
        str: Model name (e.g., "linear_baseline")
    """
    # Remove 'y_pred_' prefix and '_test.npy' suffix
    filename = filepath.stem  # Remove .npy
    if filename.endswith('_test'):
        filename = filename[:-5]  # Remove '_test'
    if filename.startswith('y_pred_'):
        filename = filename[7:]  # Remove 'y_pred_'

    return filename


def load_all_predictions(
    prediction_files: List[Path],
    y_test: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Load all prediction files and validate shapes.

    Args:
        prediction_files: List of prediction file paths
        y_test: Ground truth array (for shape validation)

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping model names to predictions
    """
    print_section("3. LOADING PREDICTIONS")

    predictions = {}

    print("Loading and validating predictions:\n")

    for filepath in prediction_files:
        model_name = extract_model_name(filepath)

        # Load predictions
        y_pred = np.load(filepath)

        # Validate shape
        if y_pred.shape != y_test.shape:
            print(f"  ✗ {model_name}: Shape mismatch!")
            print(f"      Expected: {y_test.shape}, Got: {y_pred.shape}")
            continue

        # Validate no NaN or Inf
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print(f"  ⚠️  {model_name}: Contains NaN or Inf values!")
            print(
                f"      NaN: {
                    np.isnan(y_pred).sum()}, Inf: {
                    np.isinf(y_pred).sum()}")

        predictions[model_name] = y_pred

        print(f"  ✓ {model_name:<30} Shape: {y_pred.shape}")

    print(f"\n✓ Successfully loaded {len(predictions)} model predictions")

    return predictions


# ==============================================================================
# SECTION 4: METRIC CALCULATION
# ==============================================================================


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics for a single model.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict[str, float]: Dictionary with R², MAE, MSE, RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }


def calculate_all_metrics(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Calculate metrics for all models.

    Args:
        predictions: Dictionary mapping model names to predictions
        y_test: Ground truth values

    Returns:
        pd.DataFrame: DataFrame with metrics for all models
    """
    print_section("4. CALCULATING METRICS")

    print("Calculating metrics for each model:\n")

    metrics_list = []

    for model_name, y_pred in predictions.items():
        # Calculate all metrics
        metrics = calculate_metrics(y_test, y_pred)
        metrics['Model'] = model_name

        # Calculate RMSE/MAE ratio for outlier sensitivity analysis
        rmse_mae_ratio = metrics['RMSE'] / metrics['MAE']
        metrics['RMSE_MAE_Ratio'] = rmse_mae_ratio

        metrics_list.append(metrics)

        # Display metrics
        print(f"{model_name}:")
        print(f"  R²:   {metrics['R2']:>10.{Config.DECIMAL_PLACES}f}")
        print(f"  MAE:  {metrics['MAE']:>10.{Config.DECIMAL_PLACES}f}")
        print(f"  MSE:  {metrics['MSE']:>10.{Config.DECIMAL_PLACES}f}")
        print(f"  RMSE: {metrics['RMSE']:>10.{Config.DECIMAL_PLACES}f}")
        print(f"  RMSE/MAE: {rmse_mae_ratio:>6.2f}")
        print()

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Reorder columns
    df = df[['Model', 'R2', 'MAE', 'MSE', 'RMSE', 'RMSE_MAE_Ratio']]

    print(f"✓ Metrics calculated for {len(df)} models")

    return df


# ==============================================================================
# SECTION 5: RANKING AND BEST MODEL IDENTIFICATION
# ==============================================================================


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort models by RMSE (ascending) and mark best model.

    Args:
        metrics_df: DataFrame with metrics

    Returns:
        pd.DataFrame: Sorted DataFrame with 'is_best' column
    """
    print_section("5. RANKING MODELS")

    # Sort by RMSE (lower is better)
    df_sorted = metrics_df.sort_values(
        'RMSE', ascending=True).reset_index(
        drop=True)

    # Mark best model
    df_sorted['is_best'] = False
    df_sorted.loc[0, 'is_best'] = True

    # Add rank column
    df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))

    print("Models ranked by RMSE (lower is better):\n")
    print(df_sorted[['Rank', 'Model', 'RMSE',
          'R2', 'MAE']].to_string(index=False))

    # Highlight best model
    best_model = df_sorted.iloc[0]
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_model['Model']}")
    print(f"{'=' * 80}")
    print(
        f"  RMSE: {
            best_model['RMSE']:.{
                Config.DECIMAL_PLACES}f} ⭐ (primary metric)")
    print(f"  R²:   {best_model['R2']:.{Config.DECIMAL_PLACES}f}")
    print(f"  MAE:  {best_model['MAE']:.{Config.DECIMAL_PLACES}f}")
    print(f"  MSE:  {best_model['MSE']:.{Config.DECIMAL_PLACES}f}")

    return df_sorted


# ==============================================================================
# SECTION 6: METRIC INTERPRETATION
# ==============================================================================


def print_metric_interpretations(
        metrics_df: pd.DataFrame,
        y_test: np.ndarray) -> None:
    """
    Print detailed interpretations of metrics and results.

    Args:
        metrics_df: DataFrame with sorted metrics
        y_test: Ground truth values
    """
    print_section("6. METRIC INTERPRETATIONS")

    best_model = metrics_df.iloc[0]

    print("\n📊 UNDERSTANDING THE METRICS:\n")

    # R² Interpretation
    print("1. R² (Coefficient of Determination):")
    print(f"   Best model R²: {best_model['R2']:.{Config.DECIMAL_PLACES}f}")
    print()
    print_wrapped(
        "R² measures the proportion of variance in the target variable that is "
        "explained by the model. It ranges from -∞ to 1.", indent=3)
    print()

    r2 = best_model['R2']
    if r2 >= 0.9:
        print("   ✓ R² ≥ 0.9: EXCELLENT - Model explains >90% of variance")
    elif r2 >= 0.7:
        print("   ✓ R² ≥ 0.7: GOOD - Model explains >70% of variance")
    elif r2 >= 0.5:
        print("   ⚠️  R² ≥ 0.5: MODERATE - Model explains >50% of variance")
    elif r2 >= 0.0:
        print("   ⚠️  R² ≥ 0.0: WEAK - Model barely better than mean baseline")
    else:
        print("   ✗ R² < 0.0: POOR - Model worse than predicting mean!")

    print()
    print_wrapped(
        f"Interpretation: The model explains {r2 * 100:.2f}% of the variance in "
        f"monthly electricity bills. The remaining {(1 - r2) * 100:.2f}% is unexplained.",
        indent=3
    )

    if r2 < Config.LOW_R2_THRESHOLD:
        print()
        print(
            f"   ⚠️  WARNING: R² below threshold ({
                Config.LOW_R2_THRESHOLD})")
        print("   Recommendations:")
        print("     • Add more relevant features (weather, occupancy, appliances)")
        print("     • Try non-linear models (Random Forest, Gradient Boosting)")
        print("     • Check for data quality issues")
        print("     • Consider feature engineering")

    print("\n" + "-" * 80 + "\n")

    # MAE Interpretation
    print("2. MAE (Mean Absolute Error):")
    print(f"   Best model MAE: {best_model['MAE']:.{Config.DECIMAL_PLACES}f}")
    print()
    print_wrapped(
        "MAE is the average absolute difference between predictions and actual values. "
        "It's in the same units as the target variable (€ for bills).", indent=3)
    print()

    mae = best_model['MAE']
    mean_bill = y_test.mean()
    mae_percentage = (mae / mean_bill) * 100

    print(f"   Interpretation: On average, predictions are off by €{mae:.2f}")
    print(
        f"   Relative error: {
            mae_percentage:.2f}% of mean bill (€{
            mean_bill:.2f})")

    if mae_percentage < 5:
        print("   ✓ Excellent accuracy (<5% error)")
    elif mae_percentage < 10:
        print("   ✓ Good accuracy (<10% error)")
    elif mae_percentage < 20:
        print("   ⚠️  Moderate accuracy (10-20% error)")
    else:
        print("   ⚠️  High error (>20% of mean)")

    print("\n" + "-" * 80 + "\n")

    # RMSE Interpretation
    print("3. RMSE (Root Mean Squared Error):")
    print(
        f"   Best model RMSE: {
            best_model['RMSE']:.{
                Config.DECIMAL_PLACES}f}")
    print()
    print_wrapped(
        "RMSE is the square root of the average squared errors. It penalizes large "
        "errors more heavily than MAE, making it more sensitive to outliers.",
        indent=3)
    print()

    rmse = best_model['RMSE']
    rmse_percentage = (rmse / mean_bill) * 100

    print(f"   Interpretation: RMS error is €{rmse:.2f}")
    print(f"   Relative RMSE: {rmse_percentage:.2f}% of mean bill")

    print("\n" + "-" * 80 + "\n")

    # MAE vs RMSE Comparison
    print("4. MAE vs RMSE Comparison (Outlier Sensitivity):")
    print()
    print_wrapped(
        "Comparing MAE and RMSE reveals how the model handles outliers. "
        "If RMSE >> MAE, the model has difficulty with large errors.",
        indent=3
    )
    print()

    rmse_mae_ratio = best_model['RMSE_MAE_Ratio']
    print(f"   RMSE/MAE Ratio: {rmse_mae_ratio:.2f}")
    print()

    if rmse_mae_ratio < 1.2:
        print("   ✓ Ratio < 1.2: Errors are consistent, few outliers")
        print("   → Model performs uniformly across all samples")
    elif rmse_mae_ratio < Config.HIGH_RMSE_RATIO:
        print("   ⚠️  Ratio 1.2-1.5: Some large errors present")
        print("   → Model struggles with certain edge cases")
    else:
        print("   ⚠️  Ratio > 1.5: Significant outlier errors!")
        print("   → Model has trouble with extreme values")
        print()
        print("   Recommendations:")
        print("     • Identify and analyze large error cases")
        print("     • Consider robust regression methods")
        print("     • Check for data anomalies")
        print("     • Try log transformation: y_log = log1p(y)")

    print("\n" + "-" * 80 + "\n")

    # Model comparison
    if len(metrics_df) > 1:
        print("5. Model Performance Comparison:")
        print()

        worst_model = metrics_df.iloc[-1]

        rmse_improvement = (1 - best_model['RMSE'] / worst_model['RMSE']) * 100
        mae_improvement = (1 - best_model['MAE'] / worst_model['MAE']) * 100

        print(f"   Best model: {best_model['Model']}")
        print(f"   Worst model: {worst_model['Model']}")
        print()
        print(f"   RMSE improvement: {rmse_improvement:.2f}%")
        print(f"   MAE improvement: {mae_improvement:.2f}%")

        if rmse_improvement > 20:
            print("   ✓ Significant improvement - model selection matters!")
        elif rmse_improvement > 10:
            print("   ✓ Moderate improvement - right model helps")
        else:
            print("   → Small improvement - models perform similarly")


# ==============================================================================
# SECTION 7: SAVE RESULTS
# ==============================================================================


def save_results(metrics_df: pd.DataFrame) -> None:
    """
    Save metrics to CSV and Markdown files.

    Args:
        metrics_df: DataFrame with sorted metrics
    """
    print_section("7. SAVING RESULTS")

    # Prepare DataFrame for saving (remove helper columns)
    df_to_save = metrics_df[['Rank', 'Model', 'R2',
                             'MAE', 'MSE', 'RMSE', 'is_best']].copy()

    # Save CSV
    df_to_save.to_csv(
        Config.METRICS_CSV,
        index=False,
        float_format=f'%.{
            Config.DECIMAL_PLACES}f')
    print(f"✓ Metrics saved to CSV: {Config.METRICS_CSV}")

    # Create Markdown file
    with open(Config.METRICS_MD, 'w', encoding='utf-8') as f:
        f.write("# Model Performance Metrics\n\n")
        f.write("## Comparative Table (Sorted by RMSE)\n\n")

        # Create markdown table
        f.write("| Rank | Model | R² | MAE | MSE | RMSE | Best |\n")
        f.write("|------|-------|-----|-----|-----|------|------|\n")

        for _, row in df_to_save.iterrows():
            best_marker = "✓" if row['is_best'] else ""
            f.write(
                f"| {row['Rank']} | {row['Model']} | "
                f"{row['R2']:.{Config.DECIMAL_PLACES}f} | "
                f"{row['MAE']:.{Config.DECIMAL_PLACES}f} | "
                f"{row['MSE']:.{Config.DECIMAL_PLACES}f} | "
                f"{row['RMSE']:.{Config.DECIMAL_PLACES}f} | "
                f"{best_marker} |\n"
            )

        f.write("\n---\n\n")
        f.write("## Metric Definitions\n\n")
        f.write("- **R²** (Coefficient of Determination): Proportion of variance explained (0-1, higher is better)\n")
        f.write(
            "- **MAE** (Mean Absolute Error): Average absolute error in original units (lower is better)\n")
        f.write(
            "- **MSE** (Mean Squared Error): Average squared error (lower is better)\n")
        f.write("- **RMSE** (Root Mean Squared Error): Square root of MSE, same units as target (lower is better)\n\n")

        f.write("## Key Insights\n\n")
        best_model = df_to_save.iloc[0]
        f.write(f"🏆 **Best Model**: {best_model['Model']}\n\n")
        f.write(f"- RMSE: {best_model['RMSE']:.{Config.DECIMAL_PLACES}f}\n")
        f.write(
            f"- R²: {
                best_model['R2']:.{
                    Config.DECIMAL_PLACES}f} ({
                best_model['R2'] *
                100:.2f}% variance explained)\n")
        f.write(f"- MAE: {best_model['MAE']:.{Config.DECIMAL_PLACES}f}\n\n")

        f.write("## Interpretation\n\n")
        f.write("**RMSE is the primary metric** for comparing regression models:\n")
        f.write("- Lower RMSE indicates better overall predictions\n")
        f.write("- RMSE penalizes large errors more than MAE\n")
        f.write("- Compare RMSE/MAE ratio to assess outlier sensitivity\n\n")

        f.write("**R² interpretation**:\n")
        r2 = best_model['R2']
        if r2 >= 0.9:
            f.write("- R² ≥ 0.9: Excellent explanatory power\n")
        elif r2 >= 0.7:
            f.write("- R² ≥ 0.7: Good explanatory power\n")
        elif r2 >= 0.5:
            f.write("- R² ≥ 0.5: Moderate explanatory power\n")
        else:
            f.write("- R² < 0.5: Weak explanatory power (consider more features)\n")

        f.write("\n")
        f.write("**Recommendations**:\n")
        f.write(f"- Use **{best_model['Model']}** for production deployment\n")
        f.write("- Monitor prediction errors on new data\n")
        f.write("- Retrain periodically with updated historical data\n")
        f.write("- Consider ensemble methods if performance needs improvement\n")

    print(f"✓ Metrics saved to Markdown: {Config.METRICS_MD}")

    # Summary
    print("\nOutput files:")
    print(f"  • {Config.METRICS_CSV.name} (spreadsheet format)")
    print(f"  • {Config.METRICS_MD.name} (formatted report)")


# ==============================================================================
# SECTION 8: RECOMMENDATIONS
# ==============================================================================


def print_recommendations(
        metrics_df: pd.DataFrame,
        y_test: np.ndarray) -> None:
    """
    Print actionable recommendations based on results.

    Args:
        metrics_df: DataFrame with sorted metrics
        y_test: Ground truth values
    """
    print_section("8. RECOMMENDATIONS")

    best_model = metrics_df.iloc[0]
    r2 = best_model['R2']
    rmse_mae_ratio = best_model['RMSE_MAE_Ratio']

    print("\n📋 ACTIONABLE RECOMMENDATIONS:\n")

    # Performance assessment
    print("1. Model Performance Assessment:")
    if r2 >= 0.7 and rmse_mae_ratio < 1.5:
        print("   ✓ Model performance is GOOD")
        print("   → Ready for production deployment")
        print("   → Continue monitoring on new data")
    elif r2 >= 0.5:
        print("   ⚠️  Model performance is MODERATE")
        print("   → Acceptable for initial deployment")
        print("   → Prioritize improvements in next iteration")
    else:
        print("   ⚠️  Model performance is WEAK")
        print("   → Not recommended for production")
        print("   → Focus on model improvements before deployment")

    print()

    # Specific recommendations based on metrics
    print("2. Improvement Strategies:")

    if r2 < 0.7:
        print("   • Feature Engineering:")
        print("     - Add weather data (temperature, humidity)")
        print("     - Include temporal features (day of week, holidays)")
        print("     - Create interaction features")
        print("     - Add lagged consumption values")

    if rmse_mae_ratio > Config.HIGH_RMSE_RATIO:
        print("   • Outlier Handling:")
        print("     - Investigate samples with large errors")
        print("     - Consider log transformation: y_log = log1p(y)")
        print("     - Try robust regression methods")
        print("     - Remove or cap extreme outliers")

    # Check skewness
    from scipy import stats
    try:
        skewness = stats.skew(y_test)
    except ImportError:
        skewness = np.mean(((y_test - y_test.mean()) / y_test.std()) ** 3)

    if abs(skewness) > 1.0:
        print("   • Target Transformation:")
        print(f"     - Target is skewed (skewness = {skewness:.2f})")
        print("     - Apply log transformation: y_log = np.log1p(y)")
        print("     - Remember to inverse transform predictions")
        print("     - This may improve model performance")

    print()

    # Model comparison insights
    if len(metrics_df) > 1:
        print("3. Model Selection Insights:")

        # Check if multiple models perform similarly
        rmse_values = metrics_df['RMSE'].values
        rmse_cv = (rmse_values.std() / rmse_values.mean()) * 100

        if rmse_cv < 5:
            print("   • All models perform similarly (CV < 5%)")
            print("   → Choose simplest model for interpretability")
            print("   → Consider ensemble methods")
        else:
            print(
                f"   • Significant performance differences (CV = {
                    rmse_cv:.2f}%)")
            print("   → Model selection is important")
            print(f"   → Use {best_model['Model']} for best results")

    print()

    # Next steps
    print("4. Next Steps:")
    print("   • Create residual analysis plots:")
    print("     - Predicted vs Actual scatter plot")
    print("     - Residual distribution histogram")
    print("     - Residual vs Predicted plot (check for patterns)")
    print("   • Perform error analysis:")
    print("     - Identify samples with largest errors")
    print("     - Look for patterns in misclassified samples")
    print("   • Feature importance:")
    print("     - Analyze which features drive predictions")
    print("     - Consider removing irrelevant features")
    print("   • Cross-validation:")
    print("     - Ensure robustness across different data splits")
    print("   • Production deployment:")
    print("     - Set up monitoring for prediction drift")
    print("     - Define retraining schedule")


# ==============================================================================
# SECTION 9: FINAL SUMMARY
# ==============================================================================


def print_final_summary(metrics_df: pd.DataFrame) -> None:
    """
    Print final evaluation summary.

    Args:
        metrics_df: DataFrame with sorted metrics
    """
    print_section("EVALUATION SUMMARY")

    best_model = metrics_df.iloc[0]
    n_models = len(metrics_df)

    summary = f"""
✅ Model evaluation completed successfully!

MODELS EVALUATED: {n_models}

🏆 BEST PERFORMING MODEL: {best_model['Model']}

Key Metrics:
  • RMSE:  {best_model['RMSE']:.{Config.DECIMAL_PLACES}f} ⭐ (primary metric)
  • R²:    {best_model['R2']:.{Config.DECIMAL_PLACES}f} ({best_model['R2'] * 100:.2f}% variance explained)
  • MAE:   {best_model['MAE']:.{Config.DECIMAL_PLACES}f}
  • MSE:   {best_model['MSE']:.{Config.DECIMAL_PLACES}f}

OUTPUT FILES:
  • {Config.METRICS_CSV.name} (CSV format)
  • {Config.METRICS_MD.name} (Markdown report)

INTERPRETATION:
  • RMSE: Root Mean Squared Error (lower = better)
  • R²: Proportion of variance explained (higher = better, max 1.0)
  • MAE: Mean Absolute Error (average error magnitude)
  • MSE: Mean Squared Error (penalizes large errors)

NEXT STEPS:
  1. Review detailed metrics in {Config.METRICS_MD.name}
  2. Analyze residuals to identify error patterns
  3. Create visualization plots (predicted vs actual)
  4. Deploy {best_model['Model']} for production use
  5. Set up monitoring for prediction quality

RECOMMENDATIONS:
  • Use {best_model['Model']} for predictions
  • Monitor performance on new data
  • Retrain periodically with updated data
  • Consider ensemble methods for robustness
"""

    print(summary)


# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================


def main():
    """Main evaluation pipeline execution."""
    print("=" * 80)
    print("ELECTRICITY BILL PREDICTION - METRICS EVALUATION")
    print("=" * 80)

    try:
        # 1. Load ground truth
        y_test = load_ground_truth()

        # 2. Discover prediction files
        prediction_files = discover_prediction_files()

        # 3. Load all predictions
        predictions = load_all_predictions(prediction_files, y_test)

        # 4. Calculate metrics
        metrics_df = calculate_all_metrics(predictions, y_test)

        # 5. Rank models
        metrics_df_sorted = rank_models(metrics_df)

        # 6. Print interpretations
        print_metric_interpretations(metrics_df_sorted, y_test)

        # 7. Save results
        save_results(metrics_df_sorted)

        # 8. Print recommendations
        print_recommendations(metrics_df_sorted, y_test)

        # 9. Print final summary
        print_final_summary(metrics_df_sorted)

        print("=" * 80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print("\n✗ ERROR during evaluation:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
