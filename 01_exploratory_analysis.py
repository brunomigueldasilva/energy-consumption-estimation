#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY CONSUMPTION - EXPLORATORY DATA ANALYSIS
==============================================================================

Purpose: Understand and visualize electricity consumption patterns (kWh analysis)

This script:
1. Loads raw electricity meter readings
2. Calculates daily consumption from cumulative readings
3. Aggregates to monthly consumption periods with consumption metrics
4. Loads and analyzes weather data patterns
5. Creates comprehensive visualizations and EDA
6. Analyzes consumption patterns (temporal, seasonal, tariff correlations)
7. Saves clean monthly data ready for forecasting

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Tuple, Dict, Any
import calendar
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Exploratory analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'

    # Input files
    INPUT_FILE = INPUT_DIR / 'leituras_unificadas.csv'
    WEATHER_FILE = INPUT_DIR / 'weather_data_montijo.csv'

    # Output files
    MONTHLY_DATASET_CSV = OUTPUT_DIR / 'base_mensal.csv'
    STATS_REPORT_MD = OUTPUT_DIR / 'eda_stats.md'

    # Visualization settings
    DPI = 300
    FIGSIZE = (12, 6)
    FONT_SIZE = 10

    # Outlier detection
    OUTLIER_IQR_THRESHOLD = 6.0


# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = Config.FIGSIZE
plt.rcParams['font.size'] = Config.FONT_SIZE


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filename: str, dpi: int = 300) -> None:
    """Save current plot to outputs folder."""
    filepath = Path(Config.GRAPHICS_DIR) / f"{filename}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Plot saved: {filepath}")
    plt.close()


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def sazonalidade(mes: int) -> str:
    """
    Determina a estação do ano baseada no mês.

    Args:
        mes (int): Número do mês (1-12)

    Returns:
        str: Estação correspondente em português

    Note:
        - Inverno: Dezembro, Janeiro, Fevereiro
        - Primavera: Março, Abril, Maio
        - Verão: Junho, Julho, Agosto
        - Outono: Setembro, Outubro, Novembro
    """
    if mes in [12, 1, 2]:
        return 'inverno'
    elif mes in [3, 4, 5]:
        return 'primavera'
    elif mes in [6, 7, 8]:
        return 'verão'
    else:  # [9, 10, 11]
        return 'outono'


# ==============================================================================
# SECTION 3: DATA LOADING FUNCTIONS
# ==============================================================================


def load_electricity_data() -> Tuple[pd.DataFrame, str, str, str]:
    """
    Load electricity consumption data from CSV file.

    Returns:
        Tuple of (df, date_column, off_peak_col, peak_combined_col)
    """
    print_section("1. DATA LOADING")

    # Check input folder
    if not Config.INPUT_DIR.exists():
        print(
            f"✗ ERROR: Input folder not found: {
                Config.INPUT_DIR.absolute()}")
        print(
            f"Please create the '{
                Config.INPUT_DIR}' folder and place CSV files there.")
        raise FileNotFoundError(
            f"Input directory not found: {
                Config.INPUT_DIR}")

    # Load electricity data
    print(f"Loading dataset: {Config.INPUT_FILE}\n")

    try:
        # Read with semicolon delimiter and parse dates
        df = pd.read_csv(Config.INPUT_FILE, delimiter=';',
                         parse_dates=['Data da Leitura'])
        print(f"✓ Electricity data: {len(df)} records loaded")
        print(
            f"  Period: {
                df['Data da Leitura'].min().date()} to {
                df['Data da Leitura'].max().date()}\n")

        # Sort by date
        df = df.sort_values('Data da Leitura').reset_index(drop=True)

        # Identify columns
        date_column = 'Data da Leitura'
        off_peak_col = 'Vazio'

        # Combine Ponta + Cheias into Peak
        df['Peak_Combined'] = df['Ponta'] + df['Cheias']
        peak_combined_col = 'Peak_Combined'

        print("✓ Combined tariff periods:")
        print(f"  - Date: {date_column}")
        print(f"  - Off-peak (Vazio): {off_peak_col}")
        print(f"  - Peak (Ponta + Cheias): {peak_combined_col}")

        # Check for duplicates
        duplicados = df.duplicated(subset=['Data da Leitura'], keep=False)
        n_duplicados = duplicados.sum()

        if n_duplicados > 0:
            print(f"\n⚠️  Found {n_duplicados} duplicate records by date!")
            print("   Aggregating consumptions by date (sum)...")

            # Aggregate duplicates by summing consumptions
            df = df.groupby('Data da Leitura').agg({
                'Vazio': 'sum',
                'Ponta': 'sum',
                'Cheias': 'sum',
                'Peak_Combined': 'sum'
            }).reset_index()

            print(f"✓ After aggregation: {len(df):,} unique records")
        else:
            print("✓ No duplicates by date")

        # Check for null/negative values
        print("\nData quality verification:")
        for col in ['Vazio', 'Ponta', 'Cheias']:
            n_nulos = df[col].isnull().sum()
            n_negativos = (df[col] < 0).sum()

            print(f"  {col}:")
            print(f"    Null values: {n_nulos}")
            print(f"    Negative values: {n_negativos}")

            # Handle null values
            if n_nulos > 0:
                print(
                    f"    ⚠️  Correcting {n_nulos} null values by forward-fill...")
                df[col] = df[col].ffill()
                # If still nulls at start, fill with 0
                df[col] = df[col].fillna(0)

                print("    IMPLICATION: Forward-fill assumes meter kept last")
                print(
                    "    valid value. May underestimate real consumption during failures.")

        return df, date_column, off_peak_col, peak_combined_col

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def load_weather_data() -> pd.DataFrame:
    """
    Load weather data from CSV file.

    Returns:
        pd.DataFrame: Weather data with parsed dates, or None if file not found
    """
    print_section("2. WEATHER DATA LOADING")

    if not Config.WEATHER_FILE.exists():
        print(f"⚠️  WARNING: Weather file not found: {Config.WEATHER_FILE}")
        print("   Skipping weather data analysis.")
        return None

    try:
        # Read with semicolon delimiter and parse dates
        df_weather = pd.read_csv(Config.WEATHER_FILE, delimiter=';',
                                 parse_dates=['Date'])

        print(f"✓ Weather data: {len(df_weather)} records loaded")
        print(
            f"  Period: {
                df_weather['Date'].min().date()} to {
                df_weather['Date'].max().date()}\n")

        # Sort by date
        df_weather = df_weather.sort_values('Date').reset_index(drop=True)

        # Display columns and basic statistics
        print("✓ Weather columns available:")
        for col in df_weather.columns:
            if col != 'Date':
                print(f"  - {col}:")
                print(
                    f"    Range: [{
                        df_weather[col].min():.1f}, {
                        df_weather[col].max():.1f}]")
                print(
                    f"    Mean: {
                        df_weather[col].mean():.1f}, Std: {
                        df_weather[col].std():.1f}")

        # Check for missing values
        missing = df_weather.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  Missing values found:")
            for col in missing[missing > 0].index:
                print(
                    f"  - {col}: {missing[col]} ({missing[col] / len(df_weather) * 100:.1f}%)")

            # Fill missing values with forward fill then backward fill
            df_weather = df_weather.ffill().bfill()
            print("✓ Missing values filled with forward/backward fill")
        else:
            print("\n✓ No missing values in weather data")

        return df_weather

    except Exception as e:
        print(f"✗ Error loading weather data: {e}")
        return None


# ==============================================================================
# SECTION 4: DATA PROCESSING FUNCTIONS
# ==============================================================================


def calculate_daily_consumption(df: pd.DataFrame,
                                date_column: str,
                                off_peak_col: str,
                                peak_combined_col: str) -> pd.DataFrame:
    """
    Calculate daily consumption from cumulative meter readings.

    Args:
        df: DataFrame with cumulative readings
        date_column: Name of date column
        off_peak_col: Name of off-peak column
        peak_combined_col: Name of peak combined column

    Returns:
        DataFrame with daily consumption values

    Note:
        Meter readings are cumulative and typically increasing. However, meters can
        reset to zero when replaced, causing negative differences. These are handled
        by taking absolute values.
    """
    df = df.copy()

    print("\nConverting cumulative readings to daily consumptions...")

    # Calculate daily consumption as difference from previous reading
    # Meter readings are cumulative (increasing counters)
    df['consumo_vazio_diario'] = df[off_peak_col].diff()
    df['consumo_peak_diario'] = df[peak_combined_col].diff()

    # First row will have NaN (no previous reading), remove it
    df = df.dropna()

    # Handle meter resets (when new meter is installed, readings start from 0)
    consumos_negativos = ((df['consumo_vazio_diario'] < 0) |
                          (df['consumo_peak_diario'] < 0)).sum()

    if consumos_negativos > 0:
        print(
            f"⚠️  {consumos_negativos} days with negative differences detected")
        print("   This indicates meter resets (meter replacement/rollover)")
        print("   Taking absolute values to recover actual consumption")

        # Take absolute value to handle meter resets
        df['consumo_vazio_diario'] = df['consumo_vazio_diario'].abs()
        df['consumo_peak_diario'] = df['consumo_peak_diario'].abs()

    # Calculate total daily consumption
    df['total_consumption'] = df['consumo_vazio_diario'] + \
        df['consumo_peak_diario']

    # Detect and replace outliers using IQR method
    print("\nDetecting and replacing daily consumption outliers...")

    # Calculate IQR for outlier detection
    q1 = df['total_consumption'].quantile(0.25)
    q3 = df['total_consumption'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers_mask = (
        df['total_consumption'] < lower_bound) | (
        df['total_consumption'] > upper_bound)
    n_outliers = outliers_mask.sum()

    if n_outliers > 0:
        print(
            f"⚠️  Found {n_outliers} days with outlier consumption ({
                n_outliers /
                len(df) *
                100:.1f}%):")

        # Calculate median values for replacement
        median_vazio = df.loc[~outliers_mask, 'consumo_vazio_diario'].median()
        median_peak = df.loc[~outliers_mask, 'consumo_peak_diario'].median()
        median_total = median_vazio + median_peak

        # Show some examples of outliers being replaced
        outlier_dates = df.loc[outliers_mask, date_column].head(10)
        outlier_values = df.loc[outliers_mask, 'total_consumption'].head(10)

        for date, value in zip(outlier_dates, outlier_values):
            print(
                f"    - {
                    date.date()}: {
                    value:.1f} kWh → {
                    median_total:.1f} kWh (replaced with median)")

        if n_outliers > 10:
            print(f"    ... and {n_outliers - 10} more outliers")

        # Replace outliers with median values
        df.loc[outliers_mask, 'consumo_vazio_diario'] = median_vazio
        df.loc[outliers_mask, 'consumo_peak_diario'] = median_peak
        df.loc[outliers_mask, 'total_consumption'] = median_total

        print(f"✓ Replaced {n_outliers} outlier days with median values")
        print(f"  Median off-peak: {median_vazio:.1f} kWh/day")
        print(f"  Median peak: {median_peak:.1f} kWh/day")
    else:
        print("✓ No outliers detected in daily consumption")

    # Data quality checks
    total_consumption_sum = df['total_consumption'].sum()
    zero_consumption_days = (df['total_consumption'] == 0).sum()

    print(f"✓ Daily consumption calculated for {len(df):,} days")
    print(f"  Total consumption: {total_consumption_sum:,.2f} kWh")
    print(
        f"  Days with zero consumption: {zero_consumption_days} ({
            zero_consumption_days /
            len(df) *
            100:.1f}%)")

    if zero_consumption_days > len(df) * 0.1:  # More than 10% zeros
        print("  ⚠️  WARNING: High proportion of zero-consumption days detected!")
        print(
            "     This may indicate data quality issues or extended periods of no usage")

    return df


def add_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Add temporal features extracted from reading date.

    Args:
        df: DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with new temporal columns
    """
    df = df.copy()

    # Extract temporal components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['week_of_year'] = df[date_column].dt.isocalendar().week

    # Season
    df['season'] = df['month'].apply(sazonalidade)

    # Add year-month period for aggregation
    df['ano_mes'] = df[date_column].dt.to_period('M')

    print("✓ Temporal features added: year, month, day, day_of_week, season")

    return df


def aggregate_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily consumption to monthly periods and calculate consumption metrics.
    For partial months (with missing days), extrapolate to full month consumption.
    """
    print_section("2. MONTHLY AGGREGATION AND CONSUMPTION METRICS")

    # Aggregate by month with day count
    print("Aggregating consumptions by month...")
    df_mensal = df.groupby('ano_mes').agg({
        'consumo_vazio_diario': 'sum',
        'consumo_peak_diario': 'sum',
        'year': 'first',
        'month': 'first',
        'day': 'count'  # Count actual days with data
    }).reset_index()

    # Rename columns for clarity
    df_mensal.rename(columns={
        'consumo_vazio_diario': 'vazio_kwh',
        'consumo_peak_diario': 'fora_vazio_kwh',
        'day': 'dias_com_dados'
    }, inplace=True)

    # Calculate total days in each month
    df_mensal['dias_no_mes'] = df_mensal['ano_mes'].apply(
        lambda x: calendar.monthrange(x.year, x.month)[1]
    )

    # Extrapolate for partial months (when dias_com_dados < dias_no_mes)
    partial_months = df_mensal[df_mensal['dias_com_dados']
                               < df_mensal['dias_no_mes']]
    if len(partial_months) > 0:
        print(
            f"\n⚠️  Found {
                len(partial_months)} partial months - extrapolating to full month:")
        for idx, row in partial_months.iterrows():
            factor = row['dias_no_mes'] / row['dias_com_dados']
            old_vazio = row['vazio_kwh']
            old_peak = row['fora_vazio_kwh']

            df_mensal.at[idx, 'vazio_kwh'] = row['vazio_kwh'] * factor
            df_mensal.at[idx,
                         'fora_vazio_kwh'] = row['fora_vazio_kwh'] * factor

            print(
                f"    - {row['ano_mes']}: {row['dias_com_dados']}/{row['dias_no_mes']} days")
            print(
                f"      Off-peak: {old_vazio:.1f} → {df_mensal.at[idx, 'vazio_kwh']:.1f} kWh (×{factor:.2f})")
            print(
                f"      Peak: {old_peak:.1f} → {df_mensal.at[idx, 'fora_vazio_kwh']:.1f} kWh (×{factor:.2f})")

    df_mensal['season'] = df_mensal['month'].apply(sazonalidade)

    # Calculate total monthly consumption
    df_mensal['total_consumption'] = df_mensal['vazio_kwh'] + \
        df_mensal['fora_vazio_kwh']

    # Calculate consumption per day
    df_mensal['consumption_per_day'] = df_mensal['total_consumption'] / \
        df_mensal['dias_no_mes']

    # Calculate off-peak ratio
    df_mensal['off_peak_ratio'] = df_mensal['vazio_kwh'] / \
        df_mensal['total_consumption']

    print(f"✓ {len(df_mensal)} months processed")

    # Show examples of first 3 months
    for idx, row in df_mensal.head(3).iterrows():
        print(f"\n  Example month {row['ano_mes']} ({row['season']}):")
        print(
            f"    Off-peak: {row['vazio_kwh']:.1f} kWh | Peak: {row['fora_vazio_kwh']:.1f} kWh")
        print(
            f"    Total: {
                row['total_consumption']:.1f} kWh | Daily avg: {
                row['consumption_per_day']:.1f} kWh/day")
        print(f"    Off-peak ratio: {row['off_peak_ratio'] * 100:.1f}%")

    # Remove duplicate columns if they exist
    df_mensal = df_mensal.loc[:, ~df_mensal.columns.duplicated()]

    # Select final columns for monthly base
    colunas_base = ['ano_mes', 'year', 'month', 'season', 'dias_no_mes',
                    'vazio_kwh', 'fora_vazio_kwh', 'total_consumption',
                    'consumption_per_day', 'off_peak_ratio']
    base_mensal = df_mensal[colunas_base].copy()

    return base_mensal


def process_data(df: pd.DataFrame, date_column: str, off_peak_col: str,
                 peak_combined_col: str) -> pd.DataFrame:
    """
    Process and consolidate all data.
    """
    print_section("DATA PROCESSING")

    # Calculate daily consumption
    df = calculate_daily_consumption(
        df, date_column, off_peak_col, peak_combined_col)

    # Add temporal features
    df = add_temporal_features(df, date_column)

    # Aggregate to monthly data with consumption metrics
    df_monthly = aggregate_monthly_data(df)

    return df_monthly


# ==============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ==============================================================================


def analyze_basic_statistics(df: pd.DataFrame) -> None:
    """Display basic statistics of the dataset."""
    print_section("3. BASIC STATISTICS")

    # Remove duplicate columns if they exist
    df = df.loc[:, ~df.columns.duplicated()]

    print("\nMonthly Consumption Statistics (kWh/month):")
    print("=" * 80)

    stats_df = df[['vazio_kwh', 'fora_vazio_kwh',
                   'total_consumption', 'consumption_per_day']].describe()
    print(stats_df.round(2))

    print("\n" + "=" * 80)
    print(f"Total months analyzed: {len(df)}")
    print(
        f"Average monthly consumption: {
            df['total_consumption'].mean():.2f} kWh")
    print(
        f"Median monthly consumption: {
            df['total_consumption'].median():.2f} kWh")
    print(
        f"Average daily consumption: {
            df['consumption_per_day'].mean():.2f} kWh/day")

    # Calculate total consumption
    total_consumption = df['total_consumption'].sum()
    print(f"Total consumption analyzed: {total_consumption:.2f} kWh")

    # Off-peak analysis
    print(
        f"\nAverage off-peak ratio: {df['off_peak_ratio'].mean() * 100:.1f}%")
    print(
        f"Average off-peak consumption: {df['vazio_kwh'].mean():.2f} kWh/month")
    print(
        f"Average peak consumption: {
            df['fora_vazio_kwh'].mean():.2f} kWh/month")

    # Data quality warnings
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS:")

    # Check for negative consumption
    negative_consumption = (df['total_consumption'] < 0).sum()
    if negative_consumption > 0:
        print(
            f"⚠️  WARNING: {negative_consumption} months with NEGATIVE consumption detected!")
        print("   This is physically impossible and indicates data quality issues.")
    else:
        print("✓ No negative consumption values detected")

    # Check for zero consumption
    zero_consumption = (df['total_consumption'] == 0).sum()
    if zero_consumption > 0:
        print(
            f"⚠️  WARNING: {zero_consumption} months with ZERO consumption detected!")
        print("   This may indicate missing data or vacant periods.")
    else:
        print("✓ No zero consumption months detected")

    # Check distribution symmetry
    median_consumption = df['total_consumption'].median()
    mean_consumption = df['total_consumption'].mean()
    if abs(mean_consumption - median_consumption) / \
            median_consumption > 0.1:  # >10% difference
        print(
            f"⚠️  NOTE: Mean ({
                mean_consumption:.2f} kWh) and Median ({
                median_consumption:.2f} kWh) differ by {
                abs(
                    mean_consumption - median_consumption) / median_consumption * 100:.1f}%")
        print("   This indicates some remaining variability in consumption patterns.")
    else:
        print(
            f"✓ Distribution is well-balanced (Mean: {
                mean_consumption:.2f} kWh, Median: {
                median_consumption:.2f} kWh)")


def visualize_time_series(df: pd.DataFrame) -> None:
    """Create sophisticated time series visualizations."""
    print_section("4. TIME SERIES ANALYSIS")

    print("\nCreating advanced time series plots...")

    # Convert ano_mes to datetime for better plotting
    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['ano_mes'].astype(str))

    # Plot 1: Multi-panel time series analysis
    print("\nCreating comprehensive temporal analysis...")
    _, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Monthly consumption evolution with trend and volatility
    ax1 = axes[0]
    ax1.plot(
        df_plot['date'],
        df_plot['total_consumption'],
        linewidth=1.5,
        alpha=0.8,
        color='steelblue',
        marker='o',
        markersize=4)

    # Add rolling statistics
    rolling_mean = df_plot['total_consumption'].rolling(
        window=6, center=True).mean()
    rolling_std = df_plot['total_consumption'].rolling(
        window=6, center=True).std()

    ax1.plot(df_plot['date'], rolling_mean,
             linewidth=3, color='darkred', label='6-Month Moving Average')

    # Add confidence bands
    ax1.fill_between(df_plot['date'],
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2, color='red', label='±1 Std Dev')

    ax1.set_ylabel('Monthly Consumption (kWh)', fontsize=12)
    ax1.set_title(
        'Evolution of Monthly Electricity Consumption with Trend Analysis',
        fontweight='bold',
        fontsize=14,
        pad=20)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Consumption components breakdown
    ax2 = axes[1]

    # Stacked area plot for consumption components
    ax2.fill_between(df_plot['date'], 0, df_plot['vazio_kwh'],
                     alpha=0.7, color='#3498db', label='Off-Peak (Vazio)')
    ax2.fill_between(df_plot['date'], df_plot['vazio_kwh'],
                     df_plot['vazio_kwh'] + df_plot['fora_vazio_kwh'],
                     alpha=0.7, color='#e74c3c', label='Peak (Ponta+Cheias)')

    ax2.set_ylabel('Monthly Consumption (kWh)', fontsize=12)
    ax2.set_title('Monthly Consumption Breakdown by Tariff Period',
                  fontweight='bold', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Daily average consumption analysis
    ax3 = axes[2]

    ax3.plot(df_plot['date'], df_plot['consumption_per_day'],
             linewidth=1.5, color='green', marker='s', markersize=4, alpha=0.8)
    ax3.axhline(
        y=df_plot['consumption_per_day'].median(),
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'Median: {
            df_plot["consumption_per_day"].median():.1f} kWh/day')

    ax3.set_ylabel('Daily Average (kWh/day)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Daily Average Consumption Analysis',
                  fontweight='bold', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot('eda_01_comprehensive_temporal_analysis')

    # Plot 2: Seasonal decomposition-style analysis
    print("\nCreating seasonal decomposition analysis...")
    _, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Monthly patterns
    ax = axes[0, 0]
    monthly_stats = df.groupby('month')['total_consumption'].agg([
        'mean', 'std', 'count'])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    bars = ax.bar(range(1, 13), monthly_stats['mean'],
                  yerr=monthly_stats['std'], capsize=5,
                  color=plt.cm.viridis(np.linspace(0, 1, 12)), alpha=0.8)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, monthly_stats['count'])):
        height = bar.get_height()
        ax.text(
            bar.get_x() +
            bar.get_width() /
            2.,
            height +
            monthly_stats['std'].iloc[i],
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=9)

    ax.set_xlabel('Month')
    ax.set_ylabel('Average Monthly Consumption (kWh)')
    ax.set_title('Monthly Consumption Patterns with Variability')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Seasonal patterns
    ax = axes[0, 1]
    season_order = ['inverno', 'primavera', 'verão', 'outono']
    season_colors = {'inverno': '#3498db', 'primavera': '#2ecc71',
                     'verão': '#f39c12', 'outono': '#e74c3c'}

    seasonal_data = []
    season_labels = []
    colors = []

    for season in season_order:
        data = df[df['season'] == season]['total_consumption'].values
        if len(data) > 0:
            seasonal_data.append(data)
            season_labels.append(season.title())
            colors.append(season_colors[season])

    bp = ax.boxplot(seasonal_data, labels=season_labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Season')
    ax.set_ylabel('Monthly Consumption (kWh)')
    ax.set_title('Seasonal Distribution of Monthly Consumption')
    ax.grid(True, alpha=0.3, axis='y')

    # Year-over-year comparison
    ax = axes[1, 0]
    yearly_stats = df.groupby('year')['total_consumption'].agg(
        ['mean', 'median', 'std'])

    x_pos = range(len(yearly_stats))
    ax.bar([x - 0.2 for x in x_pos], yearly_stats['mean'],
           width=0.4, label='Mean', color='lightblue', alpha=0.8)
    ax.bar([x + 0.2 for x in x_pos], yearly_stats['median'],
           width=0.4, label='Median', color='lightcoral', alpha=0.8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Monthly Consumption (kWh)')
    ax.set_title('Year-over-Year Consumption Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(yearly_stats.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Off-peak ratio analysis
    ax = axes[1, 1]

    # Create scatter with color gradient based on season
    for season, color in season_colors.items():
        mask = df['season'] == season
        if mask.sum() > 0:
            ax.scatter(
                df['total_consumption'][mask],
                df['off_peak_ratio'][mask] * 100,
                c=color,
                label=season.title(),
                alpha=0.7,
                s=60)

    ax.set_xlabel('Total Monthly Consumption (kWh)')
    ax.set_ylabel('Off-Peak Consumption Ratio (%)')
    ax.set_title('Off-Peak Usage Pattern by Consumption Level')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(
        y=50,
        color='red',
        linestyle='--',
        alpha=0.5,
        label='50% threshold')

    plt.tight_layout()
    save_plot('eda_02_seasonal_decomposition_analysis')


def visualize_seasonal_patterns(df: pd.DataFrame) -> None:
    """Analyze seasonal consumption patterns."""
    print_section("5. SEASONAL ANALYSIS")

    print("\nCreating seasonal analysis plots...")

    # Plot 3: Seasonality by season
    _, ax = plt.subplots(figsize=(10, 6))

    season_order = ['inverno', 'primavera', 'verão', 'outono']
    season_colors = {'inverno': '#3498db', 'primavera': '#2ecc71',
                     'verão': '#f39c12', 'outono': '#e74c3c'}

    seasonal_data = [df[df['season'] == season]['total_consumption'].values
                     for season in season_order]

    bp = ax.boxplot(seasonal_data, labels=season_order, patch_artist=True)

    for patch, season in zip(bp['boxes'], season_order):
        patch.set_facecolor(season_colors[season])

    ax.set_xlabel('Season')
    ax.set_ylabel('Monthly Consumption (kWh)')
    ax.set_title('Distribution of Monthly Consumption by Season',
                 fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    save_plot('eda_03_seasonal_pattern')

    # Statistical analysis by season
    print("\nAverage consumption by season:")
    for season in season_order:
        mean_val = df[df['season'] == season]['total_consumption'].mean()
        print(f"  - {season}: {mean_val:.2f} kWh")

    # Seasonal statistics
    print("\nSeasonal Statistics:")
    print("=" * 80)
    seasonal_stats = df.groupby('season')['total_consumption'].agg([
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).reindex(season_order)
    print(seasonal_stats.round(2))


def visualize_tariff_distribution(df: pd.DataFrame) -> None:
    """Advanced tariff period distribution analysis."""
    print_section("6. TARIFF PERIOD ANALYSIS")

    print("\nCreating tariff distribution analysis...")

    periods = ['vazio_kwh', 'fora_vazio_kwh']
    period_labels = ['Off-Peak (Vazio)', 'Peak (Ponta + Cheias)']

    # Calculate key metrics
    total_consumption = df['vazio_kwh'] + df['fora_vazio_kwh']
    df_analysis = df.copy()
    df_analysis['total_kwh'] = total_consumption
    df_analysis['off_peak_ratio'] = df_analysis['vazio_kwh'] / \
        (df_analysis['total_kwh'] + 0.001)
    df_analysis['peak_ratio'] = df_analysis['fora_vazio_kwh'] / \
        (df_analysis['total_kwh'] + 0.001)

    # Descriptive statistics by period
    print("\nDescriptive statistics by tariff period (monthly totals):")
    for period, label in zip(periods, period_labels):
        print(f"\n{label}:")
        print(f"  - Mean: {df[period].mean():.2f} kWh/month")
        print(f"  - Median: {df[period].median():.2f} kWh/month")
        print(f"  - Std Dev: {df[period].std():.2f} kWh")
        print(f"  - Minimum: {df[period].min():.2f} kWh")
        print(f"  - Maximum: {df[period].max():.2f} kWh")

    # Average proportion analysis
    print("\nAverage proportion of each period in total monthly consumption:")
    for period, label in zip(periods, period_labels):
        proportion = (df[period].sum() / total_consumption.sum()) * 100
        print(f"  - {label}: {proportion:.2f}%")

    off_peak_pct = (df['vazio_kwh'].sum() / total_consumption.sum()) * 100
    print(
        f"\n💡 Insight: High off-peak consumption ({off_peak_pct:.1f}%) is consistent with")
    print("   electric vehicle charging during night hours (off-peak tariff period)")

    # Plot: Comprehensive tariff analysis (4 panels - reduced from 6)
    print("\nCreating tariff analysis dashboard...")
    _, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Monthly consumption patterns (stacked area)
    ax1 = axes[0, 0]

    months = df.groupby('month')[['vazio_kwh', 'fora_vazio_kwh']].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    x = range(1, 13)
    ax1.fill_between(
        x,
        0,
        months['vazio_kwh'],
        alpha=0.8,
        color='#3498db',
        label='Off-Peak')
    ax1.fill_between(x, months['vazio_kwh'],
                     months['vazio_kwh'] + months['fora_vazio_kwh'],
                     alpha=0.8, color='#e74c3c', label='Peak')

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Monthly Consumption (kWh)')
    ax1.set_title(
        'Monthly Consumption Patterns by Tariff Period',
        fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distribution comparison with violin plots
    ax2 = axes[0, 1]

    data_for_violin = [df['vazio_kwh'].values, df['fora_vazio_kwh'].values]
    parts = ax2.violinplot(
        data_for_violin,
        positions=[
            1,
            2],
        showmeans=True,
        showmedians=True)

    # Customize violin plot colors
    colors = ['#3498db', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax2.set_xlabel('Tariff Period')
    ax2.set_ylabel('Monthly Consumption (kWh)')
    ax2.set_title(
        'Distribution Comparison: Off-Peak vs Peak',
        fontweight='bold')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Off-Peak\n(Vazio)', 'Peak\n(Ponta+Cheias)'])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistical annotations
    for i, (data, pos, color) in enumerate(
            zip(data_for_violin, [1, 2], colors)):
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        ax2.text(pos, q3 + 200, f'Q3: {q3:.0f}', ha='center', fontsize=9)
        ax2.text(
            pos,
            median + 100,
            f'Med: {
                median:.0f}',
            ha='center',
            fontsize=9,
            fontweight='bold')
        ax2.text(pos, q1 - 100, f'Q1: {q1:.0f}', ha='center', fontsize=9)

    # Panel 3: Seasonal tariff patterns
    ax3 = axes[1, 0]

    seasonal_analysis = df.groupby(
        'season')[['vazio_kwh', 'fora_vazio_kwh']].mean()

    x_pos = np.arange(len(seasonal_analysis))
    width = 0.35

    bars1 = ax3.bar(x_pos - width / 2, seasonal_analysis['vazio_kwh'],
                    width, label='Off-Peak', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x_pos + width / 2, seasonal_analysis['fora_vazio_kwh'],
                    width, label='Peak', color='#e74c3c', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 50,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Season')
    ax3.set_ylabel('Average Monthly Consumption (kWh)')
    ax3.set_title('Seasonal Tariff Period Comparison', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([season.title() for season in seasonal_analysis.index])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Ratio evolution over time
    ax4 = axes[1, 1]

    ax4.plot(
        range(
            len(df)),
        df_analysis['off_peak_ratio'] *
        100,
        linewidth=2,
        color='#3498db',
        label='Off-Peak %',
        marker='o',
        markersize=4)
    ax4.plot(
        range(
            len(df)),
        df_analysis['peak_ratio'] *
        100,
        linewidth=2,
        color='#e74c3c',
        label='Peak %',
        marker='s',
        markersize=4)

    ax4.axhline(
        y=50,
        color='gray',
        linestyle='--',
        alpha=0.7,
        label='50% Reference')
    ax4.set_xlabel('Month Index')
    ax4.set_ylabel('Consumption Ratio (%)')
    ax4.set_title('Tariff Period Ratio Evolution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    save_plot('eda_03_tariff_analysis')

    # Plot 2: Distribution analysis with Q-Q plots (reduced from separate
    # figure)
    print("\nCreating distribution analysis...")
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Enhanced histograms with statistical overlays
    for i, (ax, period, label, color) in enumerate(
            zip(axes.flat[:2], periods, period_labels, ['#3498db', '#e74c3c'])):
        data = df[period].values

        # Histogram
        n, bins, patches = ax.hist(data, bins=20, color=color, alpha=0.7,
                                   edgecolor='black', density=True)

        # Add statistical overlays
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.0f} kWh')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.0f} kWh')

        # Add normal distribution overlay for comparison
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = ((1 / (std_val * np.sqrt(2 * np.pi))) *
                  np.exp(-0.5 * ((x_norm - mean_val) / std_val) ** 2))
        ax.plot(x_norm, y_norm, 'orange', linewidth=2, alpha=0.8,
                label='Normal Approx.')

        ax.set_xlabel('Monthly Consumption (kWh)')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Analysis: {label}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add skewness and kurtosis info
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        ax.text(0.7, 0.8, f'Skew: {skewness:.2f}\nKurt: {kurtosis:.2f}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Q-Q plots for normality assessment
    for i, (ax, period, label) in enumerate(
            zip(axes.flat[2:], periods, period_labels)):
        data = df[period].values
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {label}', fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot('eda_04_distribution_analysis')


def analyze_correlations_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced correlation analysis with multiple visualization techniques."""
    print_section("7. CORRELATION ANALYSIS")

    periods = ['vazio_kwh', 'fora_vazio_kwh']

    # Extended correlation analysis
    print("Calculating comprehensive correlation matrix...")

    # Correlation matrix with consumption metrics
    correlation_cols = periods + ['total_consumption', 'dias_no_mes',
                                  'off_peak_ratio', 'consumption_per_day']
    corr_matrix = df[correlation_cols].corr()

    print("\nCorrelation Matrix (top relationships with total consumption):")
    consumption_correlations = corr_matrix['total_consumption'].drop(
        'total_consumption').sort_values(ascending=False, key=abs)
    for var, corr_val in consumption_correlations.items():
        print(f"  {var:25s}: {corr_val:6.3f}")

    # Check for suspicious correlations
    print("\n" + "=" * 80)
    print("CORRELATION QUALITY CHECKS:")

    perfect_corr_count = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.999 and abs(corr_val) < 1.0:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                print(
                    f"⚠️  Near-perfect correlation: {var1} <-> {var2} (r={corr_val:.4f})")
                print(
                    "   This may indicate redundant features or data quality issues.")
                perfect_corr_count += 1

    if perfect_corr_count == 0:
        print("✓ No suspicious near-perfect correlations detected")

    # Plot 1: Enhanced correlation heatmap with clustering
    print("\nCreating enhanced correlation heatmap...")
    _, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Standard correlation heatmap
    ax1 = axes[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.8},
        ax=ax1)
    ax1.set_title('Correlation Matrix (Lower Triangle)', fontweight='bold')

    # Clustermap for correlation structure
    ax2 = axes[1]
    # Create a dendrogram-ordered correlation matrix
    corr_linkage = hierarchy.linkage(corr_matrix.values, method='ward')
    hierarchy.dendrogram(
        corr_linkage, labels=corr_matrix.columns, ax=ax2)
    ax2.set_title('Hierarchical Clustering of Variables', fontweight='bold')
    ax2.set_xlabel('Variables')
    ax2.set_ylabel('Distance')

    plt.tight_layout()
    save_plot('eda_05_enhanced_correlation_analysis')

    # Interpretation of correlations
    print("\nDetailed correlation interpretation:")

    # Focus on relationships with total consumption
    print("\nKey relationships with Total Consumption:")
    for var in correlation_cols:
        if var != 'total_consumption':
            corr_value = corr_matrix.loc['total_consumption', var]

            if abs(corr_value) > 0.7:
                intensity = "STRONG"
            elif abs(corr_value) > 0.4:
                intensity = "moderate"
            elif abs(corr_value) > 0.2:
                intensity = "weak"
            else:
                intensity = "negligible"

            direction = "positive" if corr_value > 0 else "negative"

            print(
                f"  - {var:25s}: {corr_value:6.3f} ({intensity:10s} {direction} correlation)")

    return corr_matrix


def create_scatter_plots(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> None:
    """Create scatter plots for main relationships."""
    print_section("8. SCATTER PLOT ANALYSIS")

    print("\nCreating scatter plots...")

    vars_scatter = ['vazio_kwh', 'fora_vazio_kwh', 'dias_no_mes']

    for var in vars_scatter:
        print(f"Generating scatter plot: {var} vs monthly bill...")

        _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Scatter plot with colors by season
        seasons_colors = {'inverno': '#3498db', 'primavera': '#2ecc71',
                          'verão': '#f39c12', 'outono': '#e74c3c'}

        for season in df['season'].unique():
            mask = df['season'] == season
            ax.scatter(df.loc[mask, var],
                       df.loc[mask, 'preco_fatura'],
                       color=seasons_colors.get(season, 'gray'),
                       label=season.title(), alpha=0.7, s=50)

        # Trend line
        if len(df) > 1:
            z = np.polyfit(df[var], df['preco_fatura'], 1)
            p = np.poly1d(z)
            ax.plot(df[var], p(df[var]), "k--", alpha=0.8, linewidth=2)

        # Correlation
        corr = corr_matrix.loc[var,
                               'preco_fatura'] if var in corr_matrix.index else 0

        ax.set_xlabel(f'{var.replace("_", " ").title()}')
        ax.set_ylabel('Monthly Bill (€)')
        ax.set_title(
            f'Relationship: {
                var.replace(
                    "_",
                    " ").title()} vs Monthly Bill\n(r = {
                corr:.3f})')
        ax.legend(title='Season')
        ax.grid(True, alpha=0.3)

        save_plot(f'eda_07_scatter_{var}')


def analyze_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze anomalies and outliers in monthly billing data."""
    print_section("9. ANOMALY DETECTION")

    # Target variable statistics for outlier detection
    alvo = df['total_consumption']
    stats = {
        'count': len(alvo),
        'mean': alvo.mean(),
        'std': alvo.std(),
        'min': alvo.min(),
        'q1': alvo.quantile(0.25),
        'median': alvo.median(),
        'q3': alvo.quantile(0.75),
        'max': alvo.max(),
        'skewness': alvo.skew(),
        'kurtosis': alvo.kurtosis(),
        'iqr': alvo.quantile(0.75) - alvo.quantile(0.25)
    }

    # Identify outliers (Tukey method)
    limite_inferior = stats['q1'] - 1.5 * stats['iqr']
    limite_superior = stats['q3'] + 1.5 * stats['iqr']
    outliers = (alvo < limite_inferior) | (alvo > limite_superior)
    n_outliers = outliers.sum()

    print("Outlier Detection Statistics (Tukey Method - IQR * 1.5):")
    print(f"  Lower limit: {limite_inferior:.2f} kWh")
    print(f"  Upper limit: {limite_superior:.2f} kWh")
    print(
        f"  Outliers detected: {n_outliers} ({
            n_outliers /
            len(alvo) *
            100:.1f}%)")

    if n_outliers > 0:
        outliers_valores = alvo[outliers]
        print(
            f"  Outlier values: {', '.join([f'{x:.2f}' for x in outliers_valores])}")

    stats['n_outliers'] = n_outliers
    stats['limite_inferior'] = limite_inferior
    stats['limite_superior'] = limite_superior

    # Plot 8: Enhanced boxplot with detailed statistics
    print("\nCreating enhanced boxplot...")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot for monthly consumption
    bp1 = ax1.boxplot(alvo, patch_artist=True, labels=['Monthly Consumption'])
    bp1['boxes'][0].set_facecolor('lightblue')

    # Add outlier points if they exist
    if n_outliers > 0:
        outliers_mask = outliers
        ax1.scatter([1] * n_outliers, alvo[outliers_mask],
                    color='red', s=50, alpha=0.7, label='Outliers')
        ax1.legend()

    ax1.set_ylabel('Monthly Consumption (kWh)')
    ax1.set_title('Boxplot of Monthly Consumption')
    ax1.grid(True, alpha=0.3)

    # Histogram with distribution analysis
    ax2.hist(alvo, bins=15, alpha=0.7, color='skyblue', edgecolor='black')

    # Reference lines
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean"]:.2f} kWh')
    ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
                label=f'Median: {stats["median"]:.2f} kWh')

    ax2.set_xlabel('Monthly Consumption (kWh)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Monthly Consumption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Skewness: {
        stats["skewness"]:.3f}\nKurtosis: {
        stats["kurtosis"]:.3f}\nOutliers: {n_outliers}'
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.8))

    plt.tight_layout()
    save_plot('eda_08_distribution_analysis')

    return stats


def analyze_weather_data(df_weather: pd.DataFrame) -> None:
    """Comprehensive weather data EDA analysis."""
    if df_weather is None:
        return

    print_section("10. WEATHER DATA ANALYSIS")

    print("\nWeather Statistics Summary:")
    print("=" * 80)

    # Basic statistics
    weather_cols = [col for col in df_weather.columns if col != 'Date']
    stats_df = df_weather[weather_cols].describe()
    print(stats_df.round(2))

    # Create comprehensive weather visualizations
    print("\nCreating weather visualizations...")

    # Plot 1: Temperature trends
    _, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Temperature evolution
    ax1 = axes[0, 0]
    ax1.plot(df_weather['Date'], df_weather['Temperature_Max_C'],
             label='Max Temperature', color='red', alpha=0.7, linewidth=1)
    ax1.plot(df_weather['Date'], df_weather['Temperature_Min_C'],
             label='Min Temperature', color='blue', alpha=0.7, linewidth=1)

    # Add rolling averages
    temp_max_rolling = df_weather['Temperature_Max_C'].rolling(
        window=30, center=True).mean()
    temp_min_rolling = df_weather['Temperature_Min_C'].rolling(
        window=30, center=True).mean()

    ax1.plot(df_weather['Date'], temp_max_rolling,
             label='30-day Max Avg', color='darkred', linewidth=2.5)
    ax1.plot(df_weather['Date'], temp_min_rolling,
             label='30-day Min Avg', color='darkblue', linewidth=2.5)

    ax1.fill_between(df_weather['Date'],
                     df_weather['Temperature_Min_C'],
                     df_weather['Temperature_Max_C'],
                     alpha=0.2, color='gray')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(
        'Temperature Evolution (Max/Min)',
        fontweight='bold',
        fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Precipitation and UV Index
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()

    # Precipitation as bars
    precipitation_weekly = df_weather.set_index(
        'Date')['Precipitation_mm'].resample('W').sum()
    ax2.bar(
        precipitation_weekly.index,
        precipitation_weekly.values,
        width=5,
        alpha=0.6,
        color='steelblue',
        label='Weekly Precipitation')

    # UV Index as line
    uv_weekly = df_weather.set_index(
        'Date')['UV_Index_Max'].resample('W').mean()
    ax2_twin.plot(uv_weekly.index, uv_weekly.values,
                  color='orange', linewidth=2, marker='o', markersize=3,
                  label='Weekly Avg UV Index')

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Precipitation (mm)', color='steelblue')
    ax2_twin.set_ylabel('UV Index', color='orange')
    ax2.set_title(
        'Precipitation and UV Index Evolution',
        fontweight='bold',
        fontsize=14)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Humidity distribution
    ax3 = axes[1, 0]
    ax3.hist(df_weather['Humidity_Mean_%'], bins=30, alpha=0.7,
             color='cyan', edgecolor='black')

    mean_humidity = df_weather['Humidity_Mean_%'].mean()
    median_humidity = df_weather['Humidity_Mean_%'].median()

    ax3.axvline(mean_humidity, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_humidity:.1f}%')
    ax3.axvline(median_humidity, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_humidity:.1f}%')

    ax3.set_xlabel('Humidity (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Humidity Distribution', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Monthly averages
    ax4 = axes[1, 1]

    df_weather['month'] = df_weather['Date'].dt.month
    monthly_stats = df_weather.groupby('month').agg({
        'Temperature_Max_C': 'mean',
        'Temperature_Min_C': 'mean',
        'Precipitation_mm': 'sum',
        'Humidity_Mean_%': 'mean'
    })

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    x = range(1, len(monthly_stats) + 1)

    ax4_twin = ax4.twinx()

    # Temperature bars
    width = 0.35
    bars1 = ax4.bar([i - width / 2 for i in x],
                    monthly_stats['Temperature_Max_C'],
                    width,
                    label='Avg Max Temp',
                    color='red',
                    alpha=0.7)
    bars2 = ax4.bar([i + width / 2 for i in x],
                    monthly_stats['Temperature_Min_C'],
                    width,
                    label='Avg Min Temp',
                    color='blue',
                    alpha=0.7)

    # Precipitation line
    ax4_twin.plot(x, monthly_stats['Precipitation_mm'],
                  color='steelblue', marker='o', linewidth=2.5,
                  label='Total Precipitation', markersize=6)

    ax4.set_xlabel('Month')
    ax4.set_ylabel('Temperature (°C)')
    ax4_twin.set_ylabel('Precipitation (mm)', color='steelblue')
    ax4.set_title('Monthly Weather Patterns', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([month_names[i - 1]
                        for i in monthly_stats.index], rotation=45)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_plot('eda_09_weather_analysis')

    # Plot 2: Correlation matrix for weather variables
    print("\nCreating weather correlation analysis...")
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Weather correlations
    ax1 = axes[0]
    weather_corr = df_weather[weather_cols].corr()

    sns.heatmap(
        weather_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.8},
        ax=ax1)
    ax1.set_title(
        'Weather Variables Correlation Matrix',
        fontweight='bold',
        fontsize=14)

    # Seasonal weather patterns
    ax2 = axes[1]

    # Define seasons based on month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df_weather['season'] = df_weather['month'].apply(get_season)

    seasonal_temp = df_weather.groupby(
        'season')[['Temperature_Max_C', 'Temperature_Min_C']].mean()
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_temp = seasonal_temp.reindex(season_order)

    x_pos = np.arange(len(seasonal_temp))
    width = 0.35

    bars1 = ax2.bar(x_pos - width / 2, seasonal_temp['Temperature_Max_C'],
                    width, label='Avg Max Temp', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width / 2, seasonal_temp['Temperature_Min_C'],
                    width, label='Avg Min Temp', color='blue', alpha=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.1f}°C', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Season')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title(
        'Seasonal Temperature Patterns',
        fontweight='bold',
        fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seasonal_temp.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_plot('eda_10_weather_correlations')

    # Summary insights
    print("\n" + "=" * 80)
    print("WEATHER DATA INSIGHTS:")
    print("=" * 80)

    print("\nTemperature:")
    print(f"  • Average Max: {df_weather['Temperature_Max_C'].mean():.1f}°C")
    print(f"  • Average Min: {df_weather['Temperature_Min_C'].mean():.1f}°C")
    print(
        f"  • Temperature Range: {
            df_weather['Temperature_Min_C'].min():.1f}°C to {
            df_weather['Temperature_Max_C'].max():.1f}°C")

    print("\nPrecipitation:")
    print(f"  • Total: {df_weather['Precipitation_mm'].sum():.1f} mm")
    print(
        f"  • Days with rain: {
            (
                df_weather['Precipitation_mm'] > 0).sum()} ({
            (
                df_weather['Precipitation_mm'] > 0).sum() /
            len(df_weather) *
            100:.1f}%)")
    print(f"  • Max daily: {df_weather['Precipitation_mm'].max():.1f} mm")

    print("\nHumidity:")
    print(f"  • Average: {df_weather['Humidity_Mean_%'].mean():.1f}%")
    print(
        f"  • Range: {
            df_weather['Humidity_Mean_%'].min():.1f}% to {
            df_weather['Humidity_Mean_%'].max():.1f}%")

    print("\nUV Index:")
    print(f"  • Average Max: {df_weather['UV_Index_Max'].mean():.2f}")
    print(f"  • Highest: {df_weather['UV_Index_Max'].max():.1f}")

    # Seasonal breakdown
    print("\nSeasonal Patterns:")
    for season in season_order:
        season_data = df_weather[df_weather['season'] == season]
        if len(season_data) > 0:
            print(f"  • {season}:")
            print(
                f"    - Avg Temp: {
                    season_data['Temperature_Max_C'].mean():.1f}°C / {
                    season_data['Temperature_Min_C'].mean():.1f}°C")
            print(
                f"    - Total Precipitation: {season_data['Precipitation_mm'].sum():.1f} mm")
            print(
                f"    - Avg Humidity: {season_data['Humidity_Mean_%'].mean():.1f}%")


def create_all_visualizations(df: pd.DataFrame,
                              df_weather: pd.DataFrame = None) -> Tuple[Dict[str,
                                                                             Any],
                                                                        pd.DataFrame]:
    """Create all analysis visualizations."""
    print_section("CREATING COMPREHENSIVE VISUALIZATIONS")

    # Basic statistics
    analyze_basic_statistics(df)

    # Time series analysis
    visualize_time_series(df)

    # Seasonal patterns
    visualize_seasonal_patterns(df)

    # Tariff distribution
    visualize_tariff_distribution(df)

    # Correlation analysis (includes scatter plots)
    corr_matrix = analyze_correlations_detailed(df)

    # Anomaly detection
    stats = analyze_anomalies(df)

    # Weather data analysis
    if df_weather is not None:
        analyze_weather_data(df_weather)

    print("\n" + "=" * 80)
    print("✓ Visualization suite complete")
    total_plots = 10 if df_weather is not None else 7
    print(
        f"  Total plots generated: ~{total_plots} comprehensive visualizations")

    return stats, corr_matrix


# ==============================================================================
# SECTION 6: DATA EXPORT AND REPORTING FUNCTIONS
# ==============================================================================


def save_monthly_dataset(df: pd.DataFrame) -> None:
    """Save the monthly dataset to CSV."""
    print_section("5. SAVING MONTHLY DATASET")

    # Save full dataset
    df.to_csv(Config.MONTHLY_DATASET_CSV, index=False)

    print(f"✓ Monthly dataset saved: {Config.MONTHLY_DATASET_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Period: {df['ano_mes'].min()} to {df['ano_mes'].max()}")


def create_stats_report(df: pd.DataFrame, stats: Dict[str, Any],
                        corr_matrix: pd.DataFrame) -> None:
    """Create detailed statistics report in Markdown."""
    print_section("6. CREATING STATISTICS REPORT")

    with open(Config.STATS_REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Análise Exploratória - Energy Coast Forecast\n\n")
        f.write(
            f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Estatísticas Descritivas do Consumo (kWh)\n\n")
        f.write("| Estatística | Valor |\n")
        f.write("|-------------|-------|\n")
        f.write(f"| Contagem | {stats['count']} |\n")
        f.write(f"| Média | {stats['mean']:.2f} kWh |\n")
        f.write(f"| Desvio Padrão | {stats['std']:.2f} kWh |\n")
        f.write(f"| Mínimo | {stats['min']:.2f} kWh |\n")
        f.write(f"| Q1 (25%) | {stats['q1']:.2f} kWh |\n")
        f.write(f"| Mediana (50%) | {stats['median']:.2f} kWh |\n")
        f.write(f"| Q3 (75%) | {stats['q3']:.2f} kWh |\n")
        f.write(f"| Máximo | {stats['max']:.2f} kWh |\n")
        f.write(f"| IQR | {stats['iqr']:.2f} kWh |\n")
        f.write(f"| Skewness | {stats['skewness']:.3f} |\n")
        f.write(f"| Curtose | {stats['kurtosis']:.3f} |\n")

        f.write("\n## 2. Detecção de Outliers (Método de Tukey)\n\n")
        f.write(f"- **Limite Inferior:** {stats['limite_inferior']:.2f} kWh\n")
        f.write(f"- **Limite Superior:** {stats['limite_superior']:.2f} kWh\n")
        f.write(
            f"- **Outliers Detectados:** {
                stats['n_outliers']} ({
                stats['n_outliers'] /
                len(df) *
                100:.1f}%)\n")

        f.write("\n## 3. Matriz de Correlações\n\n")
        vars_numericas = [
            'vazio_kwh',
            'fora_vazio_kwh',
            'dias_no_mes',
            'total_consumption']
        f.write(
            "| Variável | Vazio kWh | Fora-Vazio kWh | Dias no Mês | Consumo Total |\n")
        f.write(
            "|----------|-----------|-----------------|-------------|---------------|\n")
        for var in vars_numericas:
            if var in corr_matrix.columns:
                linha = f"| {var.replace('_', ' ').title()} |"
                for var2 in vars_numericas:
                    if var2 in corr_matrix.columns:
                        linha += f" {corr_matrix.loc[var, var2]:.3f} |"
                    else:
                        linha += " N/A |"
                f.write(linha + "\n")

        f.write("\n## 4. Distribuição por Estação\n\n")
        season_stats = df.groupby('season')['total_consumption'].agg(
            ['count', 'mean', 'std']).round(2)
        f.write("| Estação | Contagem | Média (kWh) | Desvio Padrão (kWh) |\n")
        f.write("|---------|----------|-------------|---------------------|\n")
        for season, row in season_stats.iterrows():
            f.write(
                f"| {
                    season.title()} | {
                    row['count']} | {
                    row['mean']:.2f} | {
                    row['std']:.2f} |\n")

        # Interpretations and recommendations
        f.write("\n## 5. Interpretação e Recomendações\n\n")

        # Skewness analysis
        if abs(stats['skewness']) < 0.5:
            skew_interp = "aproximadamente simétrica"
            skew_rec = "Distribuição próxima da normal, MAE e RMSE devem ter comportamento similar."
        elif stats['skewness'] > 0.5:
            skew_interp = "assimétrica positiva (cauda à direita)"
            skew_rec = "Considerar transformação log1p() para normalizar. RMSE pode ser mais sensível a outliers altos."
        else:
            skew_interp = "assimétrica negativa (cauda à esquerda)"
            skew_rec = "RMSE pode penalizar mais erros em valores baixos."

        f.write("### 5.1 Análise de Assimetria\n")
        f.write(
            f"- **Skewness = {stats['skewness']:.3f}:** Distribuição {skew_interp}\n")
        f.write(f"- **Implicação:** {skew_rec}\n\n")

        # Outlier analysis
        f.write("### 5.2 Impacto de Outliers\n")
        if stats['n_outliers'] > 0:
            f.write(
                f"- **{stats['n_outliers']} outliers detectados** podem influenciar modelos sensíveis\n")
            f.write(
                "- **RMSE será mais penalizado** por estes valores extremos que MAE\n")
            f.write(
                "- **Recomendação:** Considerar modelos robustos ou tratamento de outliers\n\n")
        else:
            f.write(
                "- **Sem outliers significativos:** MAE e RMSE devem ter comportamento equilibrado\n")
            f.write("- **Recomendação:** Modelos padrão devem funcionar bem\n\n")

        # Data leakage prevention
        f.write("### 5.3 Prevenção de Data Leakage\n")
        f.write(
            "- **CRÍTICO:** Qualquer split treino/teste deve **respeitar a temporalidade**\n")
        f.write(
            "- **Método recomendado:** Time Series Split ou separação por data\n")
        f.write(
            "- **Evitar:** Random split que pode usar dados futuros para prever o passado\n\n")

    print(f"✓ Statistics report saved: {Config.STATS_REPORT_MD}")


def print_summary(df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """Print final analysis summary."""
    print_section("EXPLORATORY ANALYSIS SUMMARY")

    # Calculate key insights
    strongest_corr_var = None
    strongest_corr_val = 0

    # Find strongest correlation with total consumption
    for col in ['vazio_kwh', 'fora_vazio_kwh', 'dias_no_mes']:
        if col in df.columns:
            corr_val = df[col].corr(df['total_consumption'])
            if abs(corr_val) > abs(strongest_corr_val):
                strongest_corr_val = corr_val
                strongest_corr_var = col

    print("MAIN FINDINGS:")
    print(f"  • Period analyzed: {len(df)} months")
    print(
        f"  • Average monthly consumption: {
            stats['mean']:.2f} kWh (±{
            stats['std']:.2f})")
    print(
        f"  • Distribution: {
            'Asymmetric' if abs(
                stats['skewness']) > 0.5 else 'Approximately symmetric'}")
    print(
        f"  • Outliers: {
            stats['n_outliers']} ({
            stats['n_outliers'] /
            len(df) *
            100:.1f}% of data)")
    if strongest_corr_var:
        print(
            f"  • Strongest correlation with consumption: {
                strongest_corr_var.replace(
                    '_', ' ')} (r={
                strongest_corr_val:.3f})")

    print("\nRECOMMENDED TRANSFORMATIONS:")
    if stats['skewness'] > 0.5:
        print("  • Consider log1p(total_consumption) to reduce asymmetry")
    if stats['n_outliers'] > 0:
        print("  • Investigate outliers or use robust models")
    print("  • Create derived features: off_peak_ratio, consumption_per_day, seasonality")

    print("\nARTIFACTS GENERATED:")
    print(f"  • Monthly base: {Config.MONTHLY_DATASET_CSV.name}")
    print(f"  • Statistical report: {Config.STATS_REPORT_MD.name}")
    num_plots = len(list(Config.GRAPHICS_DIR.glob('*.png'))
                    ) if Config.GRAPHICS_DIR.exists() else 0
    print(f"  • Plots: {num_plots} images in {Config.GRAPHICS_DIR.name}/")

    print("\n⚠️  IMPORTANT - DATA LEAKAGE PREVENTION:")
    print("     Any posterior split MUST respect temporal order!")
    print("     Use TimeSeriesSplit or manual separation by date.")

    print("\n✅ EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY!")


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire exploratory analysis."""
    # Create output folders
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output folders created/verified: {Config.OUTPUT_DIR.absolute()}")

    print("=" * 80)
    print("EXPLORATORY ANALYSIS - ELECTRICITY CONSUMPTION PATTERNS (KWH)")
    print("=" * 80)

    try:
        # 1. Load data
        df, date_column, off_peak_col, peak_combined_col = load_electricity_data()

        # 2. Load weather data
        df_weather = load_weather_data()

        # 3. Process data
        df_monthly = process_data(
            df, date_column, off_peak_col, peak_combined_col)

        # 4. Create visualizations and analysis
        stats, corr_matrix = create_all_visualizations(df_monthly, df_weather)

        # 5. Save consolidated dataset
        save_monthly_dataset(df_monthly)

        # 6. Create statistics report
        create_stats_report(df_monthly, stats, corr_matrix)

        # 7. Print summary
        print_summary(df_monthly, stats)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
