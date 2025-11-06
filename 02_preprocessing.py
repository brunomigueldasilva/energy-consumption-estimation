#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - DATA PREPROCESSING
==============================================================================

Purpose: Prepare training dataset for monthly bill prediction with partial month data

This script:
1. Loads daily electricity readings and monthly aggregated data
2. Simulates partial month scenarios (cuts at day 10, 15, 20, 25)
3. Calculates accumulated consumption and engineered features
4. Integrates weather data (temperature, humidity)
5. Calculates target: monthly bill price (preco_fatura)
6. Prepares train/test split with proper temporal validation
7. Applies encoding and scaling without data leakage
8. Saves preprocessed datasets and transformation objects

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Preprocessing configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    PROCESSED_DATA_DIR = OUTPUT_DIR / 'data_processed'

    # Input files
    DAILY_FILE = INPUT_DIR / 'leituras_unificadas.csv'
    MONTHLY_FILE = OUTPUT_DIR / 'base_mensal.csv'
    WEATHER_FILE = INPUT_DIR / 'weather_data_montijo.csv'

    # Output files
    DATASET_FILE = OUTPUT_DIR / 'dataset_modelagem.csv'
    X_TRAIN_FILE = PROCESSED_DATA_DIR / 'X_train.npy'
    X_TEST_FILE = PROCESSED_DATA_DIR / 'X_test.npy'
    Y_TRAIN_FILE = PROCESSED_DATA_DIR / 'y_train.npy'
    Y_TEST_FILE = PROCESSED_DATA_DIR / 'y_test.npy'
    TRANSFORMER_FILE = PROCESSED_DATA_DIR / 'col_transformer.pkl'
    FEATURE_NAMES_FILE = PROCESSED_DATA_DIR / 'feature_names_after_transform.json'

    # Preprocessing parameters
    CUTOFF_DAYS = [10, 15, 20, 25]  # Days to simulate partial months
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ROLLING_WINDOW = 7  # Days for rolling averages

    # Portuguese electricity tariff rules (2024/2025)
    TARIFF_RULES = {
        'price_off_peak': 0.0999,      # €/kWh (Vazio: 22:00-08:00)
        'price_peak': 0.1799,          # €/kWh (Fora de Vazio)
        'price_power': 0.38,           # €/day (Contracted power)
        'network_access': 0.00,        # €/day (Network access)
        'audiovisual': 2.85,           # €/month (Audiovisual contribution)
        'tax_exploitation': 0.0023,    # €/month (DGEG tax)
        'special_tax': 0.001,          # €/kWh (Special consumption tax)
        'iva_reduced': 0.06,           # 6% VAT (First 200 kWh)
        'iva_normal': 0.23,            # 23% VAT (Above 200 kWh)
        'threshold_iva': 200,          # kWh threshold
        'discount_direct': 1.8,        # €/day (Social discount)
    }


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def sazonalidade(mes: int) -> str:
    """
    Determina a estação do ano baseada no mês.

    Args:
        mes: Número do mês (1-12)

    Returns:
        Nome da estação
    """
    if mes in [12, 1, 2]:
        return 'inverno'
    elif mes in [3, 4, 5]:
        return 'primavera'
    elif mes in [6, 7, 8]:
        return 'verão'
    else:
        return 'outono'


def calcula_fatura_mensal(vazio_kwh: float,
                          fora_vazio_kwh: float,
                          dias: int,
                          regras: Dict[str, float] = None) -> Dict[str, float]:
    """
    Calcula o preço da fatura mensal de eletricidade seguindo as regras portuguesas.
    """
    # ==================== DEBUG: INÍCIO DO CÁLCULO ====================
    print(f"\n{'=' * 80}")
    print("DEBUG: CÁLCULO DE FATURA MENSAL")
    print(f"{'=' * 80}")

    # ==================== VALIDAÇÃO DE INPUTS ====================
    print("\n[1] VALIDAÇÃO DE INPUTS")
    print(f"  • Consumo Vazio (entrada): {vazio_kwh:.2f} kWh")
    print(f"  • Consumo Fora de Vazio (entrada): {fora_vazio_kwh:.2f} kWh")
    print(f"  • Dias (entrada): {dias}")

    if vazio_kwh < 0 or fora_vazio_kwh < 0:
        print(
            f"  ⚠️  WARNING: Negative consumption detected (vazio={
                vazio_kwh:.2f}, fora_vazio={
                fora_vazio_kwh:.2f})")
        print("   Returning zero consumption bill (only fixed charges)")
        vazio_kwh = 0
        fora_vazio_kwh = 0
        print(f"  • Consumo Vazio (corrigido): {vazio_kwh:.2f} kWh")
        print(
            f"  • Consumo Fora de Vazio (corrigido): {
                fora_vazio_kwh:.2f} kWh")

    if dias <= 0:
        print(
            f"  ⚠️  ERROR: Invalid number of days ({dias}). Using 30 as default.")
        dias = 30

    if regras is None:
        regras = {
            'price_off_peak': 0.0999,
            'price_peak': 0.1799,
            'price_power': 0.38,
            'network_access': 0.00,
            'audiovisual': 2.85,
            'tax_exploitation': 0.0023,
            'special_tax': 0.001,
            'iva_reduced': 0.06,
            'iva_normal': 0.23,
            'threshold_iva': 200,
            'discount_direct': 1.8,
        }
        print("  ✓ Usando regras de tarifa padrão")
    else:
        print("  ✓ Usando regras de tarifa personalizadas")

    # ==================== CÁLCULO DO CONSUMO TOTAL ====================
    consumo_total_kwh = vazio_kwh + fora_vazio_kwh
    print("\n[2] CONSUMO TOTAL")
    print(f"  • Consumo Total: {consumo_total_kwh:.2f} kWh")

    if consumo_total_kwh > 2000:
        print(
            f"  ⚠️  WARNING: Very high monthly consumption detected ({
                consumo_total_kwh:.1f} kWh)")
        print("   This may indicate data quality issues or exceptional usage.")

    month_weight = dias / (365 / 12)
    print(f"  • Peso do mês (prorateio): {month_weight:.4f} (dias={dias})")

    # ==================== STEP 1: CUSTOS BASE DE ENERGIA (SEM IVA) ==========
    print("\n[3] STEP 1: CUSTOS BASE DE ENERGIA (sem IVA)")
    cost_off_peak_base = vazio_kwh * regras['price_off_peak']
    cost_peak_base = fora_vazio_kwh * regras['price_peak']
    cost_energy_base = cost_off_peak_base + cost_peak_base

    print(
        f"  • Vazio: {
            vazio_kwh:.2f} kWh × €{
            regras['price_off_peak']:.4f}/kWh = €{
                cost_off_peak_base:.2f}")
    print(
        f"  • Fora-Vazio: {
            fora_vazio_kwh:.2f} kWh × €{
            regras['price_peak']:.4f}/kWh = €{
                cost_peak_base:.2f}")
    print(f"  • Custo Energia Base: €{cost_energy_base:.2f}")

    # ==================== STEP 2: APLICAR IVA BASEADO NO THRESHOLD ==========
    print("\n[4] STEP 2: APLICAÇÃO DE IVA NA ENERGIA")
    threshold = regras['threshold_iva']
    discount = regras.get('discount_direct', 0)
    print(f"  • Threshold IVA: {threshold} kWh")

    if consumo_total_kwh <= threshold:
        # Aplicar desconto NO 1º ESCALÃO (antes do IVA)
        cost_first_tier_base = max(0, cost_energy_base - discount)
        cost_energy = cost_first_tier_base * (1 + regras['iva_reduced'])

        print(
            f"  • Consumo ≤ {threshold} kWh → IVA reduzido em TODO o consumo")
        print(f"  • Custo Energia Base: €{cost_energy_base:.2f}")
        print(
            f"  • Desconto aplicado AO 1º ESCALÃO (antes do IVA): -€{discount:.2f}")
        print(f"  • Base após desconto: €{cost_first_tier_base:.2f}")
        print(
            f"  • Custo Energia Final (com IVA {
                regras['iva_reduced']:.0%}): €{
                cost_energy:.2f}")
    else:
        # Aplicar desconto NO 2º ESCALÃO (antes do IVA)
        ratio_first_tier = threshold / consumo_total_kwh

        cost_first_tier_base = cost_energy_base * ratio_first_tier
        cost_first_tier = cost_first_tier_base * (1 + regras['iva_reduced'])

        cost_second_tier_base = cost_energy_base * (1 - ratio_first_tier)
        cost_second_tier_base_com_desconto = max(
            0, cost_second_tier_base - discount)
        cost_second_tier = cost_second_tier_base_com_desconto * \
            (1 + regras['iva_normal'])

        cost_energy = cost_first_tier + cost_second_tier

        print(f"  • Consumo > {threshold} kWh → IVA misto")
        print(f"    - Ratio 1º escalão: {ratio_first_tier:.2%}")
        print(
            f"    - 1º escalão ({threshold} kWh): €{
                cost_first_tier_base:.2f} × (1 + {
                regras['iva_reduced']:.0%}) = €{
                cost_first_tier:.2f}")
        print(f"    - 2º escalão base: €{cost_second_tier_base:.2f}")
        print(
            f"    - Desconto aplicado AO 2º ESCALÃO (antes do IVA): -€{discount:.2f}")
        print(
            f"    - 2º escalão após desconto: €{cost_second_tier_base_com_desconto:.2f}")
        print(
            f"    - 2º escalão final (com IVA {regras['iva_normal']:.0%}): €{cost_second_tier:.2f}")
        print(f"  • Custo Energia Total: €{cost_energy:.2f}")

    # ==================== STEP 3: CUSTOS FIXOS E ENCARGOS MENSAIS ===========
    print("\n[5] STEP 3: CUSTOS FIXOS E ENCARGOS MENSAIS")

    cost_power_base = regras['price_power'] * dias
    cost_power = cost_power_base * (1 + regras['iva_normal'])
    print(
        f"  • Potência contratada (base): €{
            regras['price_power']:.4f}/dia × {dias} dias = €{
            cost_power_base:.2f}")
    print(
        f"  • Potência contratada (com IVA {
            regras['iva_normal']:.0%}): €{
            cost_power:.2f}")

    cost_network = regras.get('network_access', 0) * dias
    print(
        f"  • Acesso à rede: €{
            regras.get(
                'network_access',
                0):.4f}/dia × {dias} dias = €{
            cost_network:.2f}")

    cost_audiovisual_base = month_weight * regras['audiovisual']
    cost_audiovisual = cost_audiovisual_base * (1 + regras['iva_reduced'])
    print(
        f"  • Contrib. audiovisual (base): €{
            regras['audiovisual']:.2f}/mês × {
            month_weight:.4f} = €{
                cost_audiovisual_base:.2f}")
    print(
        f"  • Contrib. audiovisual (com IVA {
            regras['iva_reduced']:.0%}): €{
            cost_audiovisual:.2f}")

    # ==================== STEP 4: TAXAS COM IVA NORMAL ====================
    print("\n[6] STEP 4: TAXAS (COM IVA NORMAL)")

    tax_exploitation_base = month_weight * regras['tax_exploitation']
    tax_exploitation = tax_exploitation_base * (1 + regras['iva_normal'])
    print(
        f"  • Taxa exploração (DGEG - base): €{
            regras['tax_exploitation']:.4f}/mês × {
            month_weight:.4f} = €{
                tax_exploitation_base:.4f}")
    print(
        f"  • Taxa exploração (com IVA {
            regras['iva_normal']:.0%}): €{
            tax_exploitation:.4f}")

    tax_special_base = consumo_total_kwh * regras['special_tax']
    tax_special = tax_special_base * (1 + regras['iva_normal'])
    print(
        f"  • Imposto especial consumo (base): {
            consumo_total_kwh:.2f} kWh × €{
            regras['special_tax']:.4f}/kWh = €{
                tax_special_base:.2f}")
    print(
        f"  • Imposto especial consumo (com IVA {
            regras['iva_normal']:.0%}): €{
            tax_special:.2f}")

    # ==================== STEP 5: CÁLCULO FINAL ====================
    print("\n[7] STEP 5: CÁLCULO FINAL DA FATURA")
    total_bill = (
        cost_energy +
        cost_power +
        cost_network +
        cost_audiovisual +
        tax_exploitation +
        tax_special
    )

    print("  Componentes:")
    print(f"    + Energia (com IVA e desconto): €{cost_energy:.2f}")
    print(f"    + Potência: €{cost_power:.2f}")
    print(f"    + Rede: €{cost_network:.2f}")
    print(
        f"    + Audiovisual (IVA {regras['iva_reduced']:.0%}): €{cost_audiovisual:.2f}")
    print(
        f"    + Taxa exploração (IVA {regras['iva_normal']:.0%}): €{tax_exploitation:.4f}")
    print(
        f"    + Imposto especial (IVA {regras['iva_normal']:.0%}): €{tax_special:.2f}")
    print(f"  {'─' * 76}")
    print(f"  = TOTAL: €{total_bill:.2f}")

    preco_fatura = max(0, total_bill)

    if preco_fatura != total_bill:
        print(
            f"  ⚠️  Ajuste: Fatura não pode ser negativa → €{
                preco_fatura:.2f}")

    print(f"\n{'=' * 80}")
    print(f"RESULTADO FINAL: €{preco_fatura:.2f}")
    print(f"{'=' * 80}\n")

    return {
        'preco_fatura': preco_fatura,
        'custo_energia': cost_energy,
        'custo_energia_base': cost_energy_base,
        'custo_power': cost_power,
        'custo_power_base': cost_power_base,
        'custo_network': cost_network,
        'custo_audiovisual': cost_audiovisual,
        'custo_audiovisual_base': cost_audiovisual_base,
        'taxa_exploracao': tax_exploitation,
        'taxa_exploracao_base': tax_exploitation_base,
        'taxa_special': tax_special,
        'taxa_special_base': tax_special_base,
        'desconto': discount,
        'consumo_total_kwh': consumo_total_kwh,
        'vazio_kwh': vazio_kwh,
        'fora_vazio_kwh': fora_vazio_kwh
    }


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_daily_data() -> pd.DataFrame:
    """
    Load and prepare daily electricity readings.

    Returns:
        DataFrame with daily consumption and time features
    """
    print_section("1. LOADING DAILY DATA")

    print(f"Loading daily readings: {Config.DAILY_FILE}")

    # Load data
    df = pd.read_csv(
        Config.DAILY_FILE,
        sep=';',
        parse_dates=['Data da Leitura'])
    print(f"✓ Loaded {len(df)} daily records")
    print(
        f"  Period: {
            df['Data da Leitura'].min()} to {
            df['Data da Leitura'].max()}")

    # Rename columns for clarity
    df.rename(columns={'Data da Leitura': 'date'}, inplace=True)

    # Sort by date (descending in original file)
    df = df.sort_values('date').reset_index(drop=True)

    # Create combined peak column (Ponta + Cheias = fora_vazio)
    df['fora_vazio'] = df['Ponta'] + df['Cheias']
    df.rename(columns={'Vazio': 'vazio'}, inplace=True)

    # Total daily reading
    df['total_reading'] = df['vazio'] + df['fora_vazio']

    print("\n✓ Combined tariff periods:")
    print("  - Off-peak (vazio): Vazio column")
    print("  - Peak (fora_vazio): Ponta + Cheias")

    # Calculate daily consumption from cumulative readings
    print("\nCalculating daily consumption from cumulative readings...")
    df['consumo_vazio_diario'] = df['vazio'].diff().abs()
    df['consumo_fora_vazio_diario'] = df['fora_vazio'].diff().abs()
    df['consumo_total_diario'] = df['consumo_vazio_diario'] + \
        df['consumo_fora_vazio_diario']

    # Handle first row (no previous reading)
    df.loc[0, 'consumo_vazio_diario'] = 0
    df.loc[0, 'consumo_fora_vazio_diario'] = 0
    df.loc[0, 'consumo_total_diario'] = 0

    # Remove first row with zero consumption
    df = df[df.index > 0].reset_index(drop=True)

    print(f"✓ Daily consumption calculated for {len(df)} days")
    print(f"  Total consumption: {df['consumo_total_diario'].sum():.2f} kWh")

    # Add time features
    print("\nAdding temporal features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['season'] = df['month'].apply(sazonalidade)
    df['ano_mes'] = df['date'].dt.to_period('M')

    print("✓ Temporal features added: year, month, day, day_of_week, season")

    # Add placeholder for occupancy (can be enhanced later)
    df['occupancy'] = 0  # Placeholder: 0 = unknown, 1 = occupied, 2 = vacation

    return df


def load_monthly_data() -> pd.DataFrame:
    """
    Load monthly aggregated data with bill prices.

    Returns:
        DataFrame with monthly consumption and bills
    """
    print_section("2. LOADING MONTHLY AGGREGATED DATA")

    print(f"Loading monthly data: {Config.MONTHLY_FILE}")

    df = pd.read_csv(Config.MONTHLY_FILE)
    df['ano_mes'] = pd.to_datetime(df['ano_mes']).dt.to_period('M')

    print(f"✓ Loaded {len(df)} monthly records")
    print(f"  Period: {df['ano_mes'].min()} to {df['ano_mes'].max()}")
    print(f"  Columns: {list(df.columns)}")

    # Calculate monthly bill if not present
    if 'preco_fatura' not in df.columns:
        print("\nCalculating monthly bills (preco_fatura)...")
        df['preco_fatura'] = df.apply(
            lambda row: calcula_fatura_mensal(
                row['vazio_kwh'],
                row['fora_vazio_kwh'],
                row['dias_no_mes'],
                Config.TARIFF_RULES
            )['preco_fatura'],
            axis=1
        )
        print("✓ Monthly bills calculated")
        print(f"  Mean bill: €{df['preco_fatura'].mean():.2f}")
        print(f"  Min bill: €{df['preco_fatura'].min():.2f}")
        print(f"  Max bill: €{df['preco_fatura'].max():.2f}")

    return df


def load_weather_data() -> pd.DataFrame:
    """
    Load weather data if available, otherwise return None.

    Returns:
        DataFrame with weather data or None
    """
    print_section("3. LOADING WEATHER DATA")

    if not Config.WEATHER_FILE.exists():
        print(f"⚠️  Weather file not found: {Config.WEATHER_FILE}")
        print("   Will use placeholders for temperature and humidity")
        return None

    print(f"Loading weather data: {Config.WEATHER_FILE}")

    df = pd.read_csv(Config.WEATHER_FILE, sep=';', parse_dates=['Date'])
    df.rename(columns={'Date': 'date'}, inplace=True)

    print(f"✓ Loaded {len(df)} weather records")
    print(f"  Period: {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns: {list(df.columns)}")

    # Keep relevant columns
    weather_cols = ['date', 'Temperature_Max_C', 'Temperature_Min_C',
                    'Humidity_Mean_%', 'Precipitation_mm']
    available_cols = [col for col in weather_cols if col in df.columns]
    df = df[available_cols].copy()

    # Calculate daily average temperature
    if 'Temperature_Max_C' in df.columns and 'Temperature_Min_C' in df.columns:
        df['temperature'] = (df['Temperature_Max_C'] +
                             df['Temperature_Min_C']) / 2
    elif 'Temperature_Max_C' in df.columns:
        df['temperature'] = df['Temperature_Max_C']
    else:
        df['temperature'] = 20.0  # Default fallback

    # Rename humidity column
    if 'Humidity_Mean_%' in df.columns:
        df['humidity'] = df['Humidity_Mean_%']
    else:
        df['humidity'] = 70.0  # Default fallback

    print("✓ Weather features prepared: temperature, humidity")

    return df


# ==============================================================================
# SECTION 4: PARTIAL MONTH SIMULATION
# ==============================================================================


def simulate_partial_months(df_daily: pd.DataFrame,
                            df_monthly: pd.DataFrame,
                            df_weather: pd.DataFrame = None) -> pd.DataFrame:
    """
    Simulate partial month scenarios for each historical month.

    For each month and cutoff day (10, 15, 20, 25), calculate:
    - Accumulated consumption up to cutoff
    - Days elapsed and remaining
    - Rolling averages
    - Weather features

    Args:
        df_daily: Daily consumption data
        df_monthly: Monthly aggregated data with targets
        df_weather: Weather data (optional)

    Returns:
        DataFrame with partial month scenarios
    """
    print_section("4. SIMULATING PARTIAL MONTH SCENARIOS")

    print(f"Cutoff days: {Config.CUTOFF_DAYS}")
    print("Creating scenarios for each month at each cutoff day...")

    scenarios = []

    # Group daily data by month
    df_daily_grouped = df_daily.groupby('ano_mes')

    for ano_mes, monthly_row in df_monthly.iterrows():
        period = monthly_row['ano_mes']

        # Get daily data for this month
        if period not in df_daily_grouped.groups:
            continue

        month_daily = df_daily_grouped.get_group(period).copy()
        month_daily = month_daily.sort_values('date').reset_index(drop=True)

        # Total days in month
        dias_no_mes = monthly_row['dias_no_mes']

        # Target: monthly bill
        preco_fatura = monthly_row['preco_fatura']

        # Try each cutoff day
        for cutoff_day in Config.CUTOFF_DAYS:
            # Skip if cutoff day exceeds month length
            if cutoff_day > dias_no_mes:
                continue

            # Filter data up to cutoff day
            data_up_to_cutoff = month_daily[month_daily['day_of_month'] <= cutoff_day]

            if len(data_up_to_cutoff) == 0:
                continue

            # Calculate accumulated consumption up to cutoff
            consumo_acumulado_vazio = data_up_to_cutoff['consumo_vazio_diario'].sum(
            )
            consumo_acumulado_fora = data_up_to_cutoff['consumo_fora_vazio_diario'].sum(
            )
            consumo_acumulado_total = consumo_acumulado_vazio + consumo_acumulado_fora

            # Days elapsed and remaining
            dias_decorridos = len(data_up_to_cutoff)
            dias_restantes = dias_no_mes - dias_decorridos

            # Calculate rolling averages (last 7 days)
            window = min(Config.ROLLING_WINDOW, len(data_up_to_cutoff))
            last_days = data_up_to_cutoff.tail(window)

            media_movel_vazio = last_days['consumo_vazio_diario'].mean()
            media_movel_fora = last_days['consumo_fora_vazio_diario'].mean()
            media_movel_total = last_days['consumo_total_diario'].mean()

            # Get last available date in cutoff
            last_date = data_up_to_cutoff['date'].max()

            # Weather features
            if df_weather is not None:
                # Get weather for this period
                weather_period = df_weather[
                    (df_weather['date'] >= month_daily['date'].min()) &
                    (df_weather['date'] <= last_date)
                ]

                if len(weather_period) > 0:
                    temperature = weather_period['temperature'].mean()
                    humidity = weather_period['humidity'].mean()
                else:
                    # Fallback to historical monthly average
                    temperature = 20.0
                    humidity = 70.0
            else:
                # Use seasonal defaults
                season = monthly_row['season']
                if season == 'inverno':
                    temperature = 12.0
                    humidity = 80.0
                elif season == 'verão':
                    temperature = 28.0
                    humidity = 60.0
                elif season == 'primavera':
                    temperature = 20.0
                    humidity = 70.0
                else:  # outono
                    temperature = 18.0
                    humidity = 75.0

            # Occupancy (placeholder - could be enhanced)
            occupancy = 0

            # Calculate engineered feature: projected bill based on current
            # consumption
            resultado_calculo = calcula_fatura_mensal(
                consumo_acumulado_vazio,
                consumo_acumulado_fora,
                dias_decorridos,
                Config.TARIFF_RULES
            )
            preco_fatura_projetado = resultado_calculo['preco_fatura']

            # Create scenario record
            scenario = {
                # Identifiers
                'ano_mes': period,
                'cutoff_day': cutoff_day,
                'simulation_date': last_date,

                # Temporal features
                'year': monthly_row['year'],
                'month': monthly_row['month'],
                'season': monthly_row['season'],
                'day_of_week': last_days['day_of_week'].iloc[-1] if len(last_days) > 0 else 0,

                # Days
                'dias_no_mes': dias_no_mes,
                'dias_decorridos': dias_decorridos,
                'dias_restantes': dias_restantes,

                # Accumulated consumption
                'consumo_acumulado_ate_corte_vazio_kwh': consumo_acumulado_vazio,
                'consumo_acumulado_ate_corte_fora_kwh': consumo_acumulado_fora,
                'consumo_acumulado_ate_corte_total_kwh': consumo_acumulado_total,

                # Rolling averages
                'media_movel_7d_vazio': media_movel_vazio,
                'media_movel_7d_fora': media_movel_fora,
                'media_movel_7d_total': media_movel_total,

                # Weather
                'temperature': temperature,
                'humidity': humidity,

                # Occupancy
                'occupancy': occupancy,

                # Engineered feature: projected bill for partial month
                'preco_fatura_parcial_projetado': preco_fatura_projetado,

                # Target: actual full month bill
                'preco_fatura': preco_fatura,
            }

            scenarios.append(scenario)

    # Create DataFrame
    df_scenarios = pd.DataFrame(scenarios)

    print(f"\n✓ Created {len(df_scenarios)} partial month scenarios")
    print(f"  Unique months: {df_scenarios['ano_mes'].nunique()}")
    print(
        f"  Scenarios per month: ~{
            len(df_scenarios) /
            df_scenarios['ano_mes'].nunique():.1f}")
    print("\nExample scenarios (first 3):")
    print(df_scenarios.head(3)[['ano_mes',
                                'cutoff_day',
                                'dias_decorridos',
                                'consumo_acumulado_ate_corte_total_kwh',
                                'preco_fatura']])

    return df_scenarios


# ==============================================================================
# SECTION 5: TRAIN/TEST SPLIT
# ==============================================================================


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets using temporal validation.

    Args:
        df: Complete dataset

    Returns:
        Tuple of (train_df, test_df)
    """
    print_section("5. TRAIN/TEST SPLIT (TEMPORAL VALIDATION)")

    # Sort by date to ensure temporal order
    df = df.sort_values('simulation_date').reset_index(drop=True)

    # Calculate split point (last 20% as test)
    n_total = len(df)
    n_test = int(n_total * Config.TEST_SIZE)
    n_train = n_total - n_test

    # Split
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()

    print("✓ Temporal split completed:")
    print(f"  Total samples: {n_total}")
    print(
        f"  Training samples: {n_train} ({100 * (1 - Config.TEST_SIZE):.0f}%)")
    print(f"  Test samples: {n_test} ({100 * Config.TEST_SIZE:.0f}%)")
    print(
        f"\n  Training period: {
            df_train['simulation_date'].min()} to {
            df_train['simulation_date'].max()}")
    print(
        f"  Test period: {
            df_test['simulation_date'].min()} to {
            df_test['simulation_date'].max()}")

    print("\n📊 Target distribution:")
    print(
        f"  Train - Mean: €{
            df_train['preco_fatura'].mean():.2f}, Std: €{
            df_train['preco_fatura'].std():.2f}")
    print(
        f"  Test  - Mean: €{
            df_test['preco_fatura'].mean():.2f}, Std: €{
            df_test['preco_fatura'].std():.2f}")

    return df_train, df_test


# ==============================================================================
# SECTION 6: ENCODING & SCALING
# ==============================================================================


def prepare_features_and_target(df_train: pd.DataFrame,
                                df_test: pd.DataFrame) -> Tuple:
    """
    Prepare features (X) and target (y), apply encoding and scaling.

    Args:
        df_train: Training data
        df_test: Test data

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, transformer, feature_names)
    """
    print_section("6. FEATURE PREPARATION AND SCALING")

    # Define feature columns
    numeric_features = [
        'consumo_acumulado_ate_corte_vazio_kwh',
        'consumo_acumulado_ate_corte_fora_kwh',
        'consumo_acumulado_ate_corte_total_kwh',
        'dias_decorridos',
        'dias_restantes',
        'media_movel_7d_vazio',
        'media_movel_7d_fora',
        'media_movel_7d_total',
        'temperature',
        'humidity',
        'preco_fatura_parcial_projetado',
    ]

    categorical_features = [
        'month',
        'season',
        'occupancy',
    ]

    # Target column
    target = 'preco_fatura'

    # Select features
    all_features = numeric_features + categorical_features

    print("Selected features:")
    print(
        f"  Numeric features ({len(numeric_features)}): {numeric_features[:3]}...")
    print(
        f"  Categorical features ({
            len(categorical_features)}): {categorical_features}")
    print(f"  Total features: {len(all_features)}")
    print(f"\n  Target: {target}")

    # Separate X and y
    X_train = df_train[all_features].copy()
    X_test = df_test[all_features].copy()
    y_train = df_train[target].values
    y_test = df_test[target].values

    print("\n✓ Features and target separated:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    print("  - StandardScaler for numeric features (fit on train only)")
    print("  - OneHotEncoder for categorical features (drop='first' to avoid multicollinearity)")

    # Numeric transformer: StandardScaler
    numeric_transformer = StandardScaler()

    # Categorical transformer: OneHotEncoder
    categorical_transformer = OneHotEncoder(
        drop='first',           # Drop first category to avoid dummy variable trap
        handle_unknown='ignore',  # Handle unseen categories in test set
        sparse_output=False
    )

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    # Fit on training data ONLY (critical to avoid data leakage)
    print("\n⚠️  AVOIDING DATA LEAKAGE:")
    print("   Fitting transformers on TRAINING DATA ONLY")
    print("   Test data will be transformed using training statistics")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("\n✓ Transformations applied:")
    print(f"  X_train_transformed shape: {X_train_transformed.shape}")
    print(f"  X_test_transformed shape: {X_test_transformed.shape}")

    # Get feature names after transformation
    feature_names = []

    # Numeric features keep their names
    feature_names.extend(numeric_features)

    # Categorical features get expanded with one-hot encoding
    if hasattr(
            preprocessor.named_transformers_['cat'],
            'get_feature_names_out'):
        cat_feature_names = preprocessor.named_transformers_[
            'cat'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)

    print(
        f"\n✓ Feature names after transformation ({
            len(feature_names)} total):")
    print(f"  First 5: {feature_names[:5]}")
    print(f"  Last 5: {feature_names[-5:]}")

    # Why scaling is important
    print("\n💡 WHY SCALING IS IMPORTANT:")
    print("   - Different features have different scales (e.g., kWh vs temperature)")
    print("   - Algorithms like Lasso, Ridge, SVR are sensitive to feature scales")
    print("   - StandardScaler: (X - mean) / std → zero mean, unit variance")
    print("   - Ensures all features contribute equally to model training")

    print("\n💡 WHY WE AVOID DATA LEAKAGE:")
    print("   - Scalers compute statistics (mean, std) from data")
    print("   - If we fit on ALL data, test statistics leak into training")
    print("   - This inflates model performance estimates artificially")
    print("   - SOLUTION: Fit on train only, transform both train and test")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, feature_names


# ==============================================================================
# SECTION 7: SAVE ARTIFACTS
# ==============================================================================


def save_artifacts(df_dataset: pd.DataFrame,
                   X_train: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   transformer: ColumnTransformer,
                   feature_names: List[str]) -> None:
    """
    Save all preprocessing artifacts.

    Args:
        df_dataset: Complete dataset before split
        X_train: Transformed training features
        X_test: Transformed test features
        y_train: Training target
        y_test: Test target
        transformer: Fitted column transformer
        feature_names: List of feature names after transformation
    """
    print_section("7. SAVING ARTIFACTS")

    # Create directories
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    Config.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    print("✓ Directories created/verified:")
    print(f"  - {Config.OUTPUT_DIR}")
    print(f"  - {Config.PROCESSED_DATA_DIR}")

    # Save complete dataset
    print("\nSaving complete dataset...")
    df_dataset.to_csv(Config.DATASET_FILE, index=False)
    print(f"✓ Dataset saved: {Config.DATASET_FILE}")
    print(f"  Rows: {len(df_dataset)}, Columns: {len(df_dataset.columns)}")

    # Save train/test arrays
    print("\nSaving train/test arrays...")
    np.save(Config.X_TRAIN_FILE, X_train)
    np.save(Config.X_TEST_FILE, X_test)
    np.save(Config.Y_TRAIN_FILE, y_train)
    np.save(Config.Y_TEST_FILE, y_test)

    print("✓ Arrays saved:")
    print(f"  - {Config.X_TRAIN_FILE} {X_train.shape}")
    print(f"  - {Config.X_TEST_FILE} {X_test.shape}")
    print(f"  - {Config.Y_TRAIN_FILE} {y_train.shape}")
    print(f"  - {Config.Y_TEST_FILE} {y_test.shape}")

    # Save transformer
    print("\nSaving column transformer...")
    with open(Config.TRANSFORMER_FILE, 'wb') as f:
        pickle.dump(transformer, f)
    print(f"✓ Transformer saved: {Config.TRANSFORMER_FILE}")

    # Save feature names
    print("\nSaving feature names...")
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(Config.FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✓ Feature names saved: {Config.FEATURE_NAMES_FILE}")
    print(f"  Total features: {len(feature_names)}")

    print("\n" + "=" * 80)
    print("ARTIFACTS SUMMARY")
    print("=" * 80)
    print("All preprocessing artifacts saved successfully!")
    print("\nDataset:")
    print(f"  - {Config.DATASET_FILE.name}: {len(df_dataset)} samples")
    print("\nTrain/Test arrays:")
    print(
        f"  - Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  - Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print("\nTransformation objects:")
    print(f"  - Column transformer: {Config.TRANSFORMER_FILE.name}")
    print(f"  - Feature names: {Config.FEATURE_NAMES_FILE.name}")


# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================


def main():
    """Main preprocessing pipeline execution."""
    print("=" * 80)
    print("ELECTRICITY BILL PREDICTION - DATA PREPROCESSING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Load data
        df_daily = load_daily_data()
        df_monthly = load_monthly_data()
        df_weather = load_weather_data()

        # 2. Simulate partial months
        df_scenarios = simulate_partial_months(
            df_daily, df_monthly, df_weather)

        # 3. Split train/test
        df_train, df_test = split_train_test(df_scenarios)

        # 4. Prepare features and apply transformations
        X_train, X_test, y_train, y_test, transformer, feature_names = \
            prepare_features_and_target(df_train, df_test)

        # 5. Save artifacts
        save_artifacts(df_scenarios, X_train, X_test, y_train, y_test,
                       transformer, feature_names)

        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nNext steps:")
        print(f"  1. Review dataset: {Config.DATASET_FILE}")
        print(
            f"  2. Train models using arrays in: {
                Config.PROCESSED_DATA_DIR}")
        print(
            f"  3. Use transformer for new predictions: {
                Config.TRANSFORMER_FILE}")

    except Exception as e:
        print("\n✗ ERROR during preprocessing:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
