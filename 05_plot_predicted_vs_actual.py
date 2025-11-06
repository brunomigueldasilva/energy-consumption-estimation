#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - PREDICTED VS ACTUAL VISUALIZATION
==============================================================================

Purpose: Create high-quality visualization comparing predicted vs actual values

This script:
1. Reads metrics CSV to identify best model (lowest RMSE)
2. Loads ground truth (y_test.npy) and best model predictions
3. Creates professional scatter plot with:
   - Predicted vs Actual values
   - 45-degree identity line (perfect prediction)
   - R² score annotation
   - Color-coded error regions
4. Saves plot in PNG and PDF formats (high resolution)
5. Exports plot data to CSV for auditing
6. Provides business-oriented interpretation of errors

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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11


# Configuration Constants
class Config:
    """Visualization configuration parameters."""
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
    PLOT_PNG = GRAPHICS_DIR / 'prev_vs_real.png'
    PLOT_PDF = GRAPHICS_DIR / 'prev_vs_real.pdf'
    DATA_CSV = RESULTS_DIR / 'prev_vs_real_points.csv'

    # Plot settings
    DPI = 300  # High resolution
    FIGSIZE = (12, 10)
    ANNOTATION_FONTSIZE = 14
    IDENTITY_LINE_WIDTH = 2.5
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
# SECTION 3: BEST MODEL IDENTIFICATION
# ==============================================================================


def identify_best_model() -> Tuple[str, float]:
    """
    Read metrics CSV and identify best model by RMSE.

    Returns:
        Tuple[str, float]: (best_model_name, best_rmse)

    Raises:
        SystemExit: If metrics file not found or invalid
    """
    print_section("1. IDENTIFYING BEST MODEL")

    if not Config.METRICS_CSV.exists():
        print(f"✗ ERROR: Metrics file not found: {Config.METRICS_CSV}")
        print("  Please run 04_avaliacao_metricas.py first.")
        exit(1)

    # Load metrics
    metrics_df = pd.read_csv(Config.METRICS_CSV)
    print(f"✓ Loaded metrics from: {Config.METRICS_CSV}")
    print(f"  Found {len(metrics_df)} models")

    # Find best model (lowest RMSE)
    best_idx = metrics_df['RMSE'].idxmin()
    best_model = metrics_df.loc[best_idx, 'Model']
    best_rmse = metrics_df.loc[best_idx, 'RMSE']
    best_r2 = metrics_df.loc[best_idx, 'R2']
    best_mae = metrics_df.loc[best_idx, 'MAE']

    print("\n🏆 BEST MODEL IDENTIFIED:")
    print(f"  Model:  {best_model}")
    print(f"  RMSE:   {best_rmse:.4f} €")
    print(f"  R²:     {best_r2:.4f}")
    print(f"  MAE:    {best_mae:.4f} €")

    return best_model, best_rmse


# ==============================================================================
# SECTION 4: DATA LOADING
# ==============================================================================


def load_test_data(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth and best model predictions.

    Args:
        model_name: Name of best model

    Returns:
        Tuple[np.ndarray, np.ndarray]: (y_test, y_pred)

    Raises:
        SystemExit: If required files not found
    """
    print_section("2. LOADING DATA")

    # Load ground truth
    if not Config.Y_TEST_FILE.exists():
        print(f"✗ ERROR: Ground truth file not found: {Config.Y_TEST_FILE}")
        print("  Please run 02_preprocessamento.py first.")
        exit(1)

    y_test = np.load(Config.Y_TEST_FILE)
    print(f"✓ Loaded ground truth: {Config.Y_TEST_FILE}")
    print(f"  Shape: {y_test.shape}")
    print(f"  Range: [{y_test.min():.2f}, {y_test.max():.2f}] €")

    # Load best model predictions
    pred_file = Config.PREDICTIONS_DIR / f'y_pred_{model_name}_test.npy'

    if not pred_file.exists():
        print(f"✗ ERROR: Prediction file not found: {pred_file}")
        print(f"  Expected file: y_pred_{model_name}_test.npy")
        print("  Please run 03_treino_modelos.py first.")
        exit(1)

    y_pred = np.load(pred_file)
    print(f"✓ Loaded predictions: {pred_file}")
    print(f"  Shape: {y_pred.shape}")
    print(f"  Range: [{y_pred.min():.2f}, {y_pred.max():.2f}] €")

    # Validate shapes match
    if y_test.shape != y_pred.shape:
        print("✗ ERROR: Shape mismatch!")
        print(f"  y_test: {y_test.shape}")
        print(f"  y_pred: {y_pred.shape}")
        exit(1)

    print(f"\n✓ Data loaded successfully ({len(y_test)} test samples)")

    return y_test, y_pred


# ==============================================================================
# SECTION 5: ERROR ANALYSIS
# ==============================================================================


def analyze_prediction_errors(y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Perform detailed error analysis with business interpretation.

    Args:
        y_test: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict: Dictionary with error analysis results
    """
    print_section("3. ANALYZING PREDICTION ERRORS")

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate residuals
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)

    # Percentage errors
    pct_errors = (residuals / y_test) * 100

    # Identify underestimation and overestimation
    underestimate_mask = residuals > 0  # Actual > Predicted
    overestimate_mask = residuals < 0   # Actual < Predicted

    n_underestimate = np.sum(underestimate_mask)
    n_overestimate = np.sum(overestimate_mask)

    # Calculate statistics
    mean_underestimate = residuals[underestimate_mask].mean(
    ) if n_underestimate > 0 else 0
    mean_overestimate = residuals[overestimate_mask].mean(
    ) if n_overestimate > 0 else 0

    print("\n📊 ERROR STATISTICS:")
    print(f"  R² Score:        {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print(f"  MAE:             {mae:.4f} €")
    print(f"  RMSE:            {rmse:.4f} €")
    print(f"  Mean Residual:   {residuals.mean():.4f} €")
    print(f"  Std Residual:    {residuals.std():.4f} €")

    print("\n🎯 PREDICTION BIAS:")
    print(
        f"  Underestimations: {n_underestimate}/{
            len(y_test)} ({
            n_underestimate / len(y_test) * 100:.1f}%)")
    print("    → Model predicts TOO LOW")
    print(f"    → Average error: {mean_underestimate:.2f} €")
    print(
        f"  Overestimations:  {n_overestimate}/{
            len(y_test)} ({
            n_overestimate / len(y_test) * 100:.1f}%)")
    print("    → Model predicts TOO HIGH")
    print(f"    → Average error: {mean_overestimate:.2f} €")

    # Identify where model errors most
    error_percentiles = np.percentile(abs_residuals, [50, 75, 90, 95])

    print("\n📈 ERROR DISTRIBUTION:")
    print(f"  50% of predictions within: ±{error_percentiles[0]:.2f} €")
    print(f"  75% of predictions within: ±{error_percentiles[1]:.2f} €")
    print(f"  90% of predictions within: ±{error_percentiles[2]:.2f} €")
    print(f"  95% of predictions within: ±{error_percentiles[3]:.2f} €")

    # Identify largest errors
    top_5_error_idx = np.argsort(abs_residuals)[-5:][::-1]

    print("\n⚠️  TOP 5 LARGEST ERRORS:")
    print(f"  {'Actual':>10} {'Predicted':>10} {'Error':>10} {'% Error':>10}")
    print(f"  {'-' * 44}")
    for idx in top_5_error_idx:
        actual = y_test[idx]
        pred = y_pred[idx]
        error = residuals[idx]
        pct = pct_errors[idx]
        print(f"  {actual:>10.2f} {pred:>10.2f} {error:>10.2f} {pct:>9.1f}%")

    # Business interpretation
    print_section("4. BUSINESS INTERPRETATION")

    print("\n💼 IMPLICATIONS FOR BUSINESS:\n")

    # Average bill context
    mean_bill = y_test.mean()

    if mae / mean_bill < 0.10:  # Error < 10%
        print("✓ EXCELLENT ACCURACY:")
        print(
            f"  Average error is {
                mae:.2f} € ({
                mae /
                mean_bill *
                100:.1f}% of mean bill)")
        print("  → Predictions are highly reliable")
        print("  → Suitable for customer billing estimates")
        print("  → Can be used for budget planning")
    elif mae / mean_bill < 0.20:  # Error 10-20%
        print("⚠️  MODERATE ACCURACY:")
        print(
            f"  Average error is {
                mae:.2f} € ({
                mae /
                mean_bill *
                100:.1f}% of mean bill)")
        print("  → Predictions are reasonably accurate")
        print("  → Acceptable for planning purposes")
        print("  → Consider adding safety margins for critical decisions")
    else:  # Error > 20%
        print("⚠️  HIGH UNCERTAINTY:")
        print(
            f"  Average error is {
                mae:.2f} € ({
                mae /
                mean_bill *
                100:.1f}% of mean bill)")
        print("  → Predictions have significant uncertainty")
        print("  → Not recommended for precise billing")
        print("  → Requires model improvement")

    print("\n💡 SPECIFIC INSIGHTS:\n")

    # Underestimation vs overestimation bias
    if n_underestimate > n_overestimate * 1.2:
        print("  ⚠️  SYSTEMATIC UNDERESTIMATION:")
        print("     → Model tends to predict lower than actual bills")
        print("     → Risk: Customers may be surprised by higher actual bills")
        print("     → Recommendation: Apply small upward correction factor")
    elif n_overestimate > n_underestimate * 1.2:
        print("  ⚠️  SYSTEMATIC OVERESTIMATION:")
        print("     → Model tends to predict higher than actual bills")
        print("     → Risk: Budget forecasts may be pessimistic")
        print("     → Benefit: Provides conservative estimates")
    else:
        print("  ✓ BALANCED PREDICTIONS:")
        print("     → No systematic bias detected")
        print("     → Errors are evenly distributed")

    # Error magnitude interpretation
    if error_percentiles[1] / mean_bill < 0.10:  # 75th percentile < 10%
        print("\n  ✓ CONSISTENT PERFORMANCE:")
        print(
            f"     → 75% of predictions within {
                error_percentiles[1]:.2f} € ({
                error_percentiles[1] /
                mean_bill *
                100:.1f}%)")
        print("     → Model is reliable across most scenarios")
    else:
        print("\n  ⚠️  VARIABLE PERFORMANCE:")
        print(
            f"     → 75% of predictions within {
                error_percentiles[1]:.2f} € ({
                error_percentiles[1] /
                mean_bill *
                100:.1f}%)")
        print("     → Some predictions have large errors")
        print("     → Investigate edge cases")

    # Return analysis results
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'residuals': residuals,
        'n_underestimate': n_underestimate,
        'n_overestimate': n_overestimate,
        'mean_underestimate': mean_underestimate,
        'mean_overestimate': mean_overestimate
    }


# ==============================================================================
# SECTION 6: VISUALIZATION
# ==============================================================================


def create_prediction_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    analysis: Dict
) -> None:
    """
    Create high-quality predicted vs actual scatter plot.

    Args:
        y_test: Ground truth values
        y_pred: Predicted values
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("5. CREATING VISUALIZATION")

    # Create figure
    fig, ax = plt.subplots(figsize=Config.FIGSIZE)

    # Calculate residuals for coloring
    residuals = y_test - y_pred

    # Color code by error magnitude
    # Green: small errors, Yellow: moderate, Red: large errors
    mae = analysis['mae']
    colors = np.where(
        np.abs(residuals) <= mae,
        'green',
        np.where(np.abs(residuals) <= 2 * mae, 'orange', 'red')
    )

    # Create scatter plot
    ax.scatter(
        y_test,
        y_pred,
        c=colors,
        s=Config.SCATTER_SIZE,
        alpha=Config.SCATTER_ALPHA,
        edgecolors='black',
        linewidth=0.5
    )

    # Perfect prediction line (45-degree identity line)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'b--',
        linewidth=Config.IDENTITY_LINE_WIDTH,
        label='Perfect Prediction (y=x)',
        zorder=5
    )

    # Labels and title
    ax.set_xlabel('Actual Bill (€)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Bill (€)', fontsize=14, fontweight='bold')

    title = f'Predicted vs Actual Monthly Bills - {model_name.upper()}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add R² annotation
    r2 = analysis['r2']
    mae = analysis['mae']
    rmse = analysis['rmse']

    textstr = '\n'.join([
        f'R² = {r2:.4f}',
        f'MAE = {mae:.2f} €',
        f'RMSE = {rmse:.2f} €',
        f'n = {len(y_test)}'
    ])

    # Position text box in upper left
    props = dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.9,
        edgecolor='black',
        linewidth=2)
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=Config.ANNOTATION_FONTSIZE,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )

    # Add error legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(
            facecolor='green',
            edgecolor='black',
            label=f'Good (|error| ≤ {
                mae:.2f} €)'),
        Patch(
            facecolor='orange',
            edgecolor='black',
            label=f'Moderate ({
                mae:.2f} < |error| ≤ {
                2 * mae:.2f} €)'),
        Patch(
            facecolor='red',
            edgecolor='black',
            label=f'Large (|error| > {
                2 * mae:.2f} €)'),
        plt.Line2D(
            [0],
            [0],
            color='blue',
            linestyle='--',
            linewidth=2,
            label='Perfect Prediction')]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=11,
        framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Equal aspect ratio for better visual comparison
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save PNG
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(Config.PLOT_PNG, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ Saved PNG: {Config.PLOT_PNG}")

    # Save PDF
    plt.savefig(
        Config.PLOT_PDF,
        dpi=Config.DPI,
        bbox_inches='tight',
        format='pdf')
    print(f"✓ Saved PDF: {Config.PLOT_PDF}")

    plt.close()


# ==============================================================================
# SECTION 7: DATA EXPORT
# ==============================================================================


def export_plot_data(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    analysis: Dict
) -> None:
    """
    Export plot data to CSV for auditing and transparency.

    Args:
        y_test: Ground truth values
        y_pred: Predicted values
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("6. EXPORTING PLOT DATA")

    # Create DataFrame
    residuals = analysis['residuals']
    abs_residuals = np.abs(residuals)
    pct_errors = (residuals / y_test) * 100

    df = pd.DataFrame({
        'sample_id': range(len(y_test)),
        'actual_bill_euros': y_test,
        'predicted_bill_euros': y_pred,
        'residual_euros': residuals,
        'absolute_error_euros': abs_residuals,
        'percentage_error': pct_errors,
        'error_category': np.where(
            abs_residuals <= analysis['mae'],
            'good',
            np.where(abs_residuals <= 2 * analysis['mae'], 'moderate', 'large')
        )
    })

    # Sort by absolute error (largest errors first)
    df = df.sort_values(
        'absolute_error_euros',
        ascending=False).reset_index(
        drop=True)

    # Save to CSV
    df.to_csv(Config.DATA_CSV, index=False, float_format='%.4f')
    print(f"✓ Saved plot data: {Config.DATA_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Summary statistics
    print("\n📊 DATA SUMMARY:")
    print(
        f"  Good predictions:     {
            (
                df['error_category'] == 'good').sum()} ({
            (
                df['error_category'] == 'good').sum() /
            len(df) *
            100:.1f}%)")
    print(
        f"  Moderate errors:      {
            (
                df['error_category'] == 'moderate').sum()} ({
            (
                df['error_category'] == 'moderate').sum() /
            len(df) *
            100:.1f}%)")
    print(
        f"  Large errors:         {
            (
                df['error_category'] == 'large').sum()} ({
            (
                df['error_category'] == 'large').sum() /
            len(df) *
            100:.1f}%)")


# ==============================================================================
# SECTION 8: FINAL SUMMARY
# ==============================================================================


def print_final_summary(model_name: str, analysis: Dict) -> None:
    """
    Print final summary of visualization generation.

    Args:
        model_name: Name of best model
        analysis: Dictionary with error analysis results
    """
    print_section("SUMMARY")

    r2 = analysis['r2']
    mae = analysis['mae']
    rmse = analysis['rmse']

    summary = f"""
✅ VISUALIZATION COMPLETED SUCCESSFULLY!

MODEL: {model_name.upper()}

KEY METRICS:
  • R² Score:     {r2:.4f} ({r2 * 100:.2f}% variance explained)
  • MAE:          {mae:.2f} €
  • RMSE:         {rmse:.2f} €

PREDICTION BIAS:
  • Underestimations: {analysis['n_underestimate']} samples (avg: {analysis['mean_underestimate']:.2f} €)
  • Overestimations:  {analysis['n_overestimate']} samples (avg: {analysis['mean_overestimate']:.2f} €)

OUTPUT FILES:
  📊 Visualizations:
     • {Config.PLOT_PNG.name} (high-res PNG, {Config.DPI} DPI)
     • {Config.PLOT_PDF.name} (vector PDF)

  📄 Data:
     • {Config.DATA_CSV.name} (audit data)

LOCATION: {Config.GRAPHICS_DIR.absolute()}

INTERPRETATION:
  The scatter plot shows the relationship between actual and predicted bills.
  Points closer to the blue dashed line (y=x) indicate better predictions.
  Color coding helps identify prediction quality at a glance.

NEXT STEPS:
  1. Review the visualization to assess model performance
  2. Investigate samples with large errors (red points)
  3. Consider the business implications of under/over-estimation
  4. Use audit CSV for detailed error analysis
  5. Deploy model if performance meets business requirements

RECOMMENDATIONS:
  • Monitor prediction accuracy on new data
  • Retrain model periodically with updated data
  • Set confidence intervals based on MAE/RMSE
  • Communicate uncertainty to stakeholders
"""

    print(summary)


# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================


def main():
    """Main visualization pipeline execution."""
    print("=" * 80)
    print("PREDICTED VS ACTUAL VISUALIZATION - BEST MODEL")
    print("=" * 80)

    try:
        # 1. Identify best model
        model_name, best_rmse = identify_best_model()

        # 2. Load data
        y_test, y_pred = load_test_data(model_name)

        # 3. Analyze errors
        analysis = analyze_prediction_errors(y_test, y_pred)

        # 4. Create visualization
        create_prediction_plot(y_test, y_pred, model_name, analysis)

        # 5. Export data
        export_plot_data(y_test, y_pred, model_name, analysis)

        # 6. Print summary
        print_final_summary(model_name, analysis)

        print("=" * 80)
        print("✅ VISUALIZATION GENERATION COMPLETED!")
        print("=" * 80)

    except Exception as e:
        print("\n✗ ERROR during visualization:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
