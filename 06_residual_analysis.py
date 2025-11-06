#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - RESIDUAL ANALYSIS
==============================================================================

Purpose: Detailed analysis of prediction residuals for model diagnostics

This script:
1. Identifies best model from metrics CSV
2. Loads ground truth and predictions
3. Calculates residuals (y_test - y_pred)
4. Creates diagnostic visualizations:
   - Residual histogram with normal distribution overlay
   - Residuals vs Predicted values scatter plot
5. Performs statistical analysis:
   - Mean, std, skewness, kurtosis
   - Tests for heteroscedasticity
   - Identifies outliers
6. Provides actionable interpretation:
   - Bias detection
   - Heteroscedasticity patterns
   - Non-linearity indicators
   - Outlier analysis
7. Saves statistics and recommendations to Markdown report

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11


# Configuration Constants
class Config:
    """Residual analysis configuration parameters."""
    # Directories
    OUTPUT_DIR = Path('outputs')
    PROCESSED_DATA_DIR = OUTPUT_DIR / 'data_processed'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'

    # Input files
    METRICS_CSV = RESULTS_DIR / 'metricas.csv'
    Y_TEST_FILE = PROCESSED_DATA_DIR / 'y_test.npy'

    # Output files
    HIST_PNG = GRAPHICS_DIR / 'residuos_hist.png'
    HIST_PDF = GRAPHICS_DIR / 'residuos_hist.pdf'
    SCATTER_PNG = GRAPHICS_DIR / 'residuos_vs_previstos.png'
    SCATTER_PDF = GRAPHICS_DIR / 'residuos_vs_previstos.pdf'
    STATS_MD = RESULTS_DIR / 'residuos_stats.md'

    # Plot settings
    DPI = 300
    HIST_BINS = 30
    SCATTER_SIZE = 80
    SCATTER_ALPHA = 0.6


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_best_model_data() -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Load best model predictions and ground truth.

    Returns:
        Tuple[str, np.ndarray, np.ndarray]: (model_name, y_test, y_pred)

    Raises:
        SystemExit: If required files not found
    """
    print_section("1. LOADING DATA")

    # Load metrics to identify best model
    if not Config.METRICS_CSV.exists():
        print(f"✗ ERROR: Metrics file not found: {Config.METRICS_CSV}")
        print("  Please run 04_avaliacao_metricas.py first.")
        exit(1)

    metrics_df = pd.read_csv(Config.METRICS_CSV)
    best_idx = metrics_df['RMSE'].idxmin()
    model_name = metrics_df.loc[best_idx, 'Model']

    print(f"✓ Best model identified: {model_name}")

    # Load ground truth
    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Ground truth not found: {Config.Y_TEST_FILE}")
        exit(1)

    y_test = np.load(Config.Y_TEST_FILE)
    print(f"✓ Loaded ground truth: {len(y_test)} samples")

    # Load predictions
    pred_file = Config.PREDICTIONS_DIR / f'y_pred_{model_name}_test.npy'
    if not pred_file.exists():
        print(f"✗ ERROR: Predictions not found: {pred_file}")
        exit(1)

    y_pred = np.load(pred_file)
    print(f"✓ Loaded predictions: {len(y_pred)} samples")

    # Validate
    if y_test.shape != y_pred.shape:
        print("✗ ERROR: Shape mismatch!")
        exit(1)

    return model_name, y_test, y_pred


# ==============================================================================
# SECTION 4: RESIDUAL CALCULATION
# ==============================================================================


def calculate_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (actual - predicted).

    Args:
        y_test: Ground truth values
        y_pred: Predicted values

    Returns:
        np.ndarray: Residuals
    """
    print_section("2. CALCULATING RESIDUALS")

    residuals = y_test - y_pred

    print("✓ Residuals calculated")
    print("  Formula: residual = actual - predicted")
    print("  Positive residual → Model underestimates (predicts too low)")
    print("  Negative residual → Model overestimates (predicts too high)")

    return residuals


# ==============================================================================
# SECTION 5: STATISTICAL ANALYSIS
# ==============================================================================


def analyze_residuals(residuals: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Perform comprehensive statistical analysis of residuals.

    Args:
        residuals: Residual values
        y_pred: Predicted values

    Returns:
        Dict: Dictionary with statistical results
    """
    print_section("3. STATISTICAL ANALYSIS")

    # Basic statistics
    mean_res = np.mean(residuals)
    median_res = np.median(residuals)
    std_res = np.std(residuals)
    min_res = np.min(residuals)
    max_res = np.max(residuals)

    print("\n📊 DESCRIPTIVE STATISTICS:")
    bias_msg = '✓ (close to zero)' if abs(
        mean_res) < 0.5 else '⚠️  (bias detected)'
    print(f"  Mean:     {mean_res:>10.4f} € {bias_msg}")
    print(f"  Median:   {median_res:>10.4f} €")
    print(f"  Std Dev:  {std_res:>10.4f} €")
    print(f"  Min:      {min_res:>10.4f} €")
    print(f"  Max:      {max_res:>10.4f} €")

    # Quartiles and IQR
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1

    print("\n📐 DISTRIBUTION MEASURES:")
    print(f"  Q1 (25%): {q1:>10.4f} €")
    print(f"  Q3 (75%): {q3:>10.4f} €")
    print(f"  IQR:      {iqr:>10.4f} €")

    # Shape statistics
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    print("\n📈 SHAPE STATISTICS:")
    skew_msg = ('✓ (symmetric)' if abs(skewness) < 0.5
                else '⚠️  (right-skewed)' if skewness > 0
                else '⚠️  (left-skewed)')
    print(f"  Skewness: {skewness:>10.4f} {skew_msg}")

    kurt_msg = ('✓ (normal-like)' if abs(kurtosis) < 0.5
                else '⚠️  (heavy tails)' if kurtosis > 0
                else '⚠️  (light tails)')
    print(f"  Kurtosis: {kurtosis:>10.4f} {kurt_msg}")

    # Normality test (Shapiro-Wilk for n <= 5000, Anderson-Darling otherwise)
    if len(residuals) <= 5000:
        _, normality_p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
    else:
        # For large samples, use Anderson-Darling
        result = stats.anderson(residuals, dist='norm')
        # Approximate p-value conversion (rough estimate)
        normality_p = 1.0 if result.statistic < result.critical_values[2] else 0.01
        test_name = "Anderson-Darling"

    print("\n🔬 NORMALITY TEST:")
    norm_msg = '✓ (normally distributed)' if normality_p > 0.05 else '⚠️  (non-normal)'
    print(f"  {test_name} p-value: {normality_p:.4f} {norm_msg}")

    # Heteroscedasticity analysis
    # Split predictions into low, mid, high groups
    n = len(y_pred)
    sorted_idx = np.argsort(y_pred)
    third = n // 3

    low_group = residuals[sorted_idx[:third]]
    high_group = residuals[sorted_idx[-third:]]

    var_low = np.var(low_group)
    var_high = np.var(high_group)
    variance_ratio = var_high / var_low if var_low > 0 else 1.0

    print("\n🔍 HETEROSCEDASTICITY:")
    print(f"  Variance (low predictions):  {var_low:.4f}")
    print(f"  Variance (high predictions): {var_high:.4f}")
    hetero_msg = ('✓ (homoscedastic)' if variance_ratio < 1.5
                  else '⚠️  (mild heteroscedasticity)' if variance_ratio < 2.5
                  else '⚠️  (strong heteroscedasticity)')
    print(f"  Variance Ratio:              {variance_ratio:.2f} {hetero_msg}")

    # Outlier detection (using IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (residuals < lower_bound) | (residuals > upper_bound)
    n_outliers = np.sum(outliers)
    pct_outliers = 100 * n_outliers / len(residuals)

    print("\n⚠️  OUTLIERS (IQR method):")
    print(f"  Lower bound: {lower_bound:.4f} €")
    print(f"  Upper bound: {upper_bound:.4f} €")
    outlier_msg = '✓ (acceptable)' if pct_outliers < 5 else '⚠️  (many outliers)'
    print(f"  Outliers:    {n_outliers} ({pct_outliers:.1f}%) {outlier_msg}")

    return {
        'mean': mean_res,
        'median': median_res,
        'std': std_res,
        'min': min_res,
        'max': max_res,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality_p': normality_p,
        'variance_ratio': variance_ratio,
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers
    }


# ==============================================================================
# SECTION 6: INTERPRETATION
# ==============================================================================


def interpret_residuals(stats: Dict) -> Dict:
    """
    Interpret residual statistics and provide recommendations.

    Args:
        stats: Dictionary with statistical results

    Returns:
        Dict: Interpretation with findings and recommendations
    """
    print_section("4. INTERPRETATION")

    findings = []
    recommendations = []

    # Bias check
    if abs(stats['mean']) < 0.5:
        findings.append("✓ No significant bias detected (mean residual ≈ 0)")
    else:
        bias_dir = "underestimation" if stats['mean'] > 0 else "overestimation"
        findings.append(
            f"⚠️  Systematic {bias_dir} detected (mean = {
                stats['mean']:.2f} €)")
        recommendations.append(
            f"Address systematic {bias_dir} by reviewing feature engineering or model calibration")

    # Normality check
    if stats['normality_p'] > 0.05:
        findings.append("✓ Residuals are normally distributed")
    else:
        findings.append("⚠️  Residuals deviate from normal distribution")
        if abs(stats['skewness']) > 0.5:
            skew_dir = "right" if stats['skewness'] > 0 else "left"
            findings.append(f"  - Distribution is {skew_dir}-skewed")
            recommendations.append(
                "Consider log transformation or Box-Cox transformation to address skewness"
            )

    # Heteroscedasticity check
    if stats['variance_ratio'] < 1.5:
        findings.append("✓ Residuals show constant variance (homoscedastic)")
    elif stats['variance_ratio'] < 2.5:
        findings.append(
            f"⚠️  Mild heteroscedasticity detected (variance ratio = {
                stats['variance_ratio']:.2f})")
        recommendations.append(
            "Consider variance-stabilizing transformations or weighted regression"
        )
    else:
        findings.append(
            f"⚠️  Strong heteroscedasticity detected "
            f"(variance ratio = {stats['variance_ratio']:.2f})"
        )
        recommendations.append(
            "Apply log transformation or use weighted least squares to handle heteroscedasticity"
        )

    # Outliers check
    if stats['pct_outliers'] < 5:
        findings.append(
            f"✓ Outliers are minimal ({
                stats['pct_outliers']:.1f}%)")
    else:
        findings.append(
            f"⚠️  Many outliers detected ({
                stats['pct_outliers']:.1f}%)")
        recommendations.append(
            "Investigate outliers - consider robust regression or outlier removal"
        )

    print("\n🔍 KEY FINDINGS:")
    for finding in findings:
        print(f"  {finding}")

    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n💡 RECOMMENDATIONS:")
        print("  ✓ Model residuals look good. Continue monitoring performance.")

    return {
        'findings': findings,
        'recommendations': recommendations
    }


# ==============================================================================
# SECTION 7: VISUALIZATION - HISTOGRAM
# ==============================================================================


def create_residual_histogram(residuals: np.ndarray, model_name: str,
                              stats: Dict) -> None:
    """
    Create residual histogram with normal distribution overlay.

    Args:
        residuals: Residual values
        model_name: Name of the model
        stats: Dictionary with statistical results
    """
    print_section("5. CREATING HISTOGRAM")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Histogram
    n, bins, patches = ax.hist(
        residuals,
        bins=Config.HIST_BINS,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )

    # Normal distribution overlay
    mu, sigma = stats['mean'], stats['std']
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats_scipy.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')

    # Zero line
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
               label='Zero (No Bias)')

    # Mean line
    ax.axvline(x=mu, color='orange', linestyle='--', linewidth=2,
               label=f'Mean = {mu:.2f} €')

    # Labels and title
    ax.set_xlabel('Residual (Actual - Predicted) [€]', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Residual Distribution - {model_name.upper()}', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = (
        f"Mean: {stats['mean']:.4f} €\n"
        f"Std Dev: {stats['std']:.4f} €\n"
        f"Skewness: {stats['skewness']:.4f}\n"
        f"Kurtosis: {stats['kurtosis']:.4f}"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    fig.savefig(Config.HIST_PNG, dpi=Config.DPI, bbox_inches='tight')
    fig.savefig(Config.HIST_PDF, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Histogram saved: {Config.HIST_PNG}")
    print(f"✓ Histogram saved: {Config.HIST_PDF}")


# ==============================================================================
# SECTION 8: VISUALIZATION - SCATTER PLOT
# ==============================================================================


def create_residual_scatter(residuals: np.ndarray, y_pred: np.ndarray,
                            model_name: str, stats: Dict) -> None:
    """
    Create residuals vs predicted values scatter plot.

    Args:
        residuals: Residual values
        y_pred: Predicted values
        model_name: Name of the model
        stats: Dictionary with statistical results
    """
    print_section("6. CREATING SCATTER PLOT")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    ax.scatter(y_pred, residuals,
               s=Config.SCATTER_SIZE,
               alpha=Config.SCATTER_ALPHA,
               c='steelblue',
               edgecolors='black',
               linewidths=0.5)

    # Zero line
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2,
               label='Zero (Perfect Predictions)')

    # Mean line
    ax.axhline(y=stats['mean'], color='orange', linestyle='--', linewidth=2,
               label=f'Mean Residual = {stats["mean"]:.2f} €')

    # +/- 2 std bands
    ax.axhline(y=2 * stats['std'], color='red', linestyle=':', linewidth=1,
               alpha=0.5, label='±2 Std Dev')
    ax.axhline(y=-2 * stats['std'], color='red', linestyle=':', linewidth=1,
               alpha=0.5)

    # LOWESS smoothing to detect patterns
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    residuals_sorted = residuals[sorted_idx]

    # Only apply smoothing if we have enough points
    if len(y_pred) > 50:
        window = min(51, len(y_pred) // 2 * 2 + 1)  # Ensure odd window
        smoothed = savgol_filter(residuals_sorted, window, 3)
        ax.plot(y_pred_sorted, smoothed, 'r-', linewidth=2,
                label='Trend (LOWESS)')

    # Labels and title
    ax.set_xlabel('Predicted Value [€]', fontsize=13)
    ax.set_ylabel('Residual (Actual - Predicted) [€]', fontsize=13)
    ax.set_title(
        f'Residuals vs Predicted Values - {model_name.upper()}',
        fontsize=16
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add interpretation box
    hetero_msg = (
        "Homoscedastic ✓" if stats['variance_ratio'] < 1.5
        else f"Heteroscedastic (ratio={stats['variance_ratio']:.2f})"
    )
    pattern_text = (
        f"Variance Ratio: {stats['variance_ratio']:.2f}\n"
        f"{hetero_msg}\n"
        f"Outliers: {stats['n_outliers']} ({stats['pct_outliers']:.1f}%)"
    )
    ax.text(0.98, 0.02, pattern_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    fig.savefig(Config.SCATTER_PNG, dpi=Config.DPI, bbox_inches='tight')
    fig.savefig(Config.SCATTER_PDF, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Scatter plot saved: {Config.SCATTER_PNG}")
    print(f"✓ Scatter plot saved: {Config.SCATTER_PDF}")


# ==============================================================================
# SECTION 9: MARKDOWN REPORT
# ==============================================================================


def create_markdown_report(model_name: str, stats: Dict,
                           interpretation: Dict) -> None:
    """
    Create detailed markdown report with findings and recommendations.

    Args:
        model_name: Name of the model
        stats: Dictionary with statistical results
        interpretation: Dictionary with interpretation results
    """
    print_section("7. CREATING MARKDOWN REPORT")

    report = f"""# Residual Analysis Report

**Model:** {model_name.upper()}
**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Residual Statistics

### Descriptive Statistics

| Statistic | Value |
|-----------|-------|
| **Mean** | {stats['mean']:.4f} € |
| **Median** | {stats['median']:.4f} € |
| **Std Dev** | {stats['std']:.4f} € |
| **Min** | {stats['min']:.4f} € |
| **Max** | {stats['max']:.4f} € |
| **Q1 (25%)** | {stats['q1']:.4f} € |
| **Q3 (75%)** | {stats['q3']:.4f} € |
| **IQR** | {stats['iqr']:.4f} € |

### Distribution Shape

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Skewness** | {stats['skewness']:.4f} | \
{'Symmetric' if abs(stats['skewness']) < 0.5
        else 'Right-skewed' if stats['skewness'] > 0 else 'Left-skewed'} |
| **Kurtosis** | {stats['kurtosis']:.4f} | \
{'Normal-like' if abs(stats['kurtosis']) < 0.5
        else 'Heavy tails' if stats['kurtosis'] > 0 else 'Light tails'} |
| **Normality (p)** | {stats['normality_p']:.4f} | {'Normal ✓' if stats['normality_p'] > 0.05 else 'Non-normal ⚠️'} |

### Heteroscedasticity

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Variance Ratio** | {stats['variance_ratio']:.2f} | \
{'Homoscedastic ✓' if stats['variance_ratio'] < 1.5
        else 'Mild heteroscedasticity ⚠️' if stats['variance_ratio'] < 2.5
        else 'Strong heteroscedasticity ⚠️'} |

### Outliers

| Metric | Value |
|--------|-------|
| **Count** | {stats['n_outliers']} |
| **Percentage** | {stats['pct_outliers']:.1f}% |

---

## 2. Key Findings

"""

    for finding in interpretation['findings']:
        report += f"- {finding}\n"

    report += "\n---\n\n## 3. Interpretation\n\n"

    # Bias interpretation
    report += "### Bias Assessment\n\n"
    if abs(stats['mean']) < 0.5:
        report += "✓ **No significant bias detected.** The mean residual is close to zero, "
        report += "indicating the model does not systematically over or underestimate.\n\n"
    else:
        bias_dir = "underestimation" if stats['mean'] > 0 else "overestimation"
        report += f"⚠️  **Systematic {bias_dir} detected.** "
        report += f"Mean residual = {stats['mean']:.2f} €. "
        predict_msg = "predict too low" if stats['mean'] > 0 else "predict too high"
        report += f"The model tends to {predict_msg}.\n\n"

    # Heteroscedasticity interpretation
    report += "### Heteroscedasticity Assessment\n\n"
    if stats['variance_ratio'] < 1.5:
        report += "✓ **Homoscedastic residuals.** Error variance is constant across prediction levels. "
        report += "This is ideal for regression models.\n\n"
    elif stats['variance_ratio'] < 2.5:
        report += f"⚠️  **Mild heteroscedasticity detected** (variance ratio = {
            stats['variance_ratio']:.2f}). "
        report += "Error variance increases slightly with prediction level. "
        report += "Consider variance-stabilizing transformations.\n\n"
    else:
        report += f"⚠️  **Strong heteroscedasticity detected** (variance ratio = {
            stats['variance_ratio']:.2f}). "
        report += "Error variance varies significantly with prediction level. "
        report += "This violates regression assumptions and may require:\n"
        report += "- Log transformation of target variable\n"
        report += "- Weighted least squares regression\n"
        report += "- Different models for different prediction ranges\n\n"

    # Normality interpretation
    report += "### Normality Assessment\n\n"
    if stats['normality_p'] > 0.05:
        report += "✓ **Residuals are normally distributed.** This validates model assumptions "
        report += "and ensures confidence intervals are reliable.\n\n"
    else:
        report += f"⚠️  **Residuals deviate from normality** (p = {
            stats['normality_p']:.4f}). "
        if abs(stats['skewness']) > 0.5:
            skew_dir = "right" if stats['skewness'] > 0 else "left"
            report += f"Distribution is {skew_dir}-skewed (skewness = {
                stats['skewness']:.2f}). "
        if abs(stats['kurtosis']) > 1.0:
            report += f"Heavy tails present (kurtosis = {
                stats['kurtosis']:.2f}), indicating outliers. "
        report += "\n\n"

    report += "---\n\n## 4. Recommendations\n\n"

    if len(interpretation['recommendations']) == 0:
        report += "✓ Model residuals look good. Continue monitoring performance on new data.\n"
    else:
        for i, rec in enumerate(interpretation['recommendations'], 1):
            report += f"{i}. {rec}\n"

    report += "\n---\n\n## 5. Visualizations\n\n"
    report += "The following diagnostic plots are available:\n\n"
    report += f"1. **{Config.HIST_PNG.name}** - Residual histogram with normal distribution overlay\n"
    report += f"2. **{Config.SCATTER_PNG.name}** - Residuals vs predicted values scatter plot\n\n"
    report += "These plots help identify:\n"
    report += "- Bias (histogram centered away from zero)\n"
    report += "- Non-normality (histogram shape differs from normal curve)\n"
    report += "- Heteroscedasticity (funnel shape in scatter plot)\n"
    report += "- Non-linearity (curvature in scatter plot)\n"
    report += "- Outliers (extreme points)\n\n"

    report += "---\n\n## 6. Next Steps\n\n"
    report += "Based on this analysis:\n\n"

    if (stats['variance_ratio'] > 2.0 or abs(stats['skewness']) > 1.0
            or stats['pct_outliers'] > 5):
        report += "**Model improvements recommended:**\n\n"
        report += "1. **Feature Engineering:**\n"
        report += "   - Create interaction terms between key features\n"
        report += "   - Add polynomial features for non-linear relationships\n"
        report += "   - Engineer domain-specific features\n\n"
        report += "2. **Transformation:**\n"
        report += "   - Try log transformation: `y_log = np.log1p(y)`\n"
        report += "   - Consider Box-Cox transformation\n"
        report += "   - Apply standardization/normalization\n\n"
        report += "3. **Advanced Models:**\n"
        report += "   - Random Forest (handles non-linearity naturally)\n"
        report += "   - Gradient Boosting (better for complex patterns)\n"
        report += "   - Robust regression (if outliers are issue)\n\n"
    else:
        report += "**Model performing well:**\n\n"
        report += "- Continue using current model for predictions\n"
        report += "- Monitor performance on new data\n"
        report += "- Retrain periodically with updated data\n"
        report += "- Consider ensemble methods for marginal improvements\n"

    # Save report
    with open(Config.STATS_MD, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Markdown report saved: {Config.STATS_MD}")


# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================


def main():
    """Main residual analysis pipeline execution."""
    print("=" * 80)
    print("RESIDUAL ANALYSIS - MODEL DIAGNOSTICS")
    print("=" * 80)

    try:
        # 1. Load data
        model_name, y_test, y_pred = load_best_model_data()

        # 2. Calculate residuals
        residuals = calculate_residuals(y_test, y_pred)

        # 3. Statistical analysis
        stats_results = analyze_residuals(residuals, y_pred)

        # 4. Interpretation
        interpretation = interpret_residuals(stats_results)

        # 5. Create histogram
        create_residual_histogram(residuals, model_name, stats_results)

        # 6. Create scatter plot
        create_residual_scatter(residuals, y_pred, model_name, stats_results)

        # 7. Create markdown report
        create_markdown_report(model_name, stats_results, interpretation)

        # Final summary
        print_section("SUMMARY")
        print("\n✅ RESIDUAL ANALYSIS COMPLETED!\n")
        print(f"Model: {model_name.upper()}")
        print("\nKey Statistics:")
        mean_msg = '✓' if abs(stats_results['mean']) < 0.5 else '⚠️'
        print(f"  Mean residual:     {stats_results['mean']:.4f} € {mean_msg}")
        print(f"  Std dev:           {stats_results['std']:.4f} €")
        norm_msg = '✓' if stats_results['normality_p'] > 0.05 else '⚠️'
        print(
            f"  Normality p-value: {stats_results['normality_p']:.4f} {norm_msg}")
        var_msg = '✓' if stats_results['variance_ratio'] < 1.5 else '⚠️'
        print(
            f"  Variance ratio:    {
                stats_results['variance_ratio']:.2f} {var_msg}")
        outlier_msg = '✓' if stats_results['pct_outliers'] < 5 else '⚠️'
        print(
            f"  Outliers:          {stats_results['n_outliers']} "
            f"({stats_results['pct_outliers']:.1f}%) {outlier_msg}"
        )

        print("\nOutput Files:")
        print(f"  📊 {Config.HIST_PNG}")
        print(f"  📊 {Config.HIST_PDF}")
        print(f"  📊 {Config.SCATTER_PNG}")
        print(f"  📊 {Config.SCATTER_PDF}")
        print(f"  📄 {Config.STATS_MD}")

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print("\n✗ ERROR during analysis:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Import scipy.stats properly
    import scipy.stats as stats_scipy
    main()
