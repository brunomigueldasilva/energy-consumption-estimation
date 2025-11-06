#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ELECTRICITY BILL PREDICTION - PIPELINE ORCHESTRATOR
==============================================================================

Purpose: Execute complete end-to-end machine learning pipeline

This script automates the execution of all ML pipeline scripts with dependency
checking, logging, and error handling.

Usage:
    python 08_orchestrator.py              # interactive mode
    python 08_orchestrator.py --all        # run complete pipeline
    python 08_orchestrator.py --steps 1,3  # run specific steps
    python 08_orchestrator.py --clean      # clean outputs

This script:
1. Runs 01_exploratory_analysis.py (EDA and feature engineering)
2. Runs 02_preprocessing.py (data preparation and partial month simulation)
3. Runs 03_train_models.py (model training)
4. Runs 04_evaluate_metrics.py (performance evaluation)
5. Runs 05_plot_predicted_vs_actual.py (visualization)
6. Runs 06_residual_analysis.py (residual analysis)
7. Runs 07_final_report.py (comprehensive documentation)
8. Provides error handling and progress tracking throughout pipeline

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import subprocess
import sys
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict


# Try to import colorama for colored output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    class Fore:
        RED = GREEN = BLUE = YELLOW = CYAN = MAGENTA = RESET = ""

    class Style:
        BRIGHT = RESET_ALL = ""


# Configuration Constants
class Config:
    """Orchestrator configuration parameters."""
    # Pipeline scripts
    SCRIPTS = [
        "01_exploratory_analysis.py",
        "02_preprocessing.py",
        "03_train_models.py",
        "04_evaluate_metrics.py",
        "05_plot_predicted_vs_actual.py",
        "06_residual_analysis.py",
        "07_final_report.py"
    ]

    SCRIPT_NAMES = [
        "Exploratory Analysis",
        "Preprocessing & Feature Engineering",
        "Model Training",
        "Metrics Evaluation",
        "Predicted vs Actual Plot",
        "Residual Analysis",
        "Final Report"
    ]

    # File and directory paths
    LOG_FILE = "execution.log"
    MONTHLY_DATASET = "outputs/base_mensal.csv"
    MODELING_DATASET = "outputs/dataset_modelagem.csv"
    PROCESSED_DATA_DIR = "outputs/data_processed/"
    MODELS_DIR = "outputs/models/"
    METRICS_FILE = "outputs/results/metricas.csv"
    FINAL_REPORT = "outputs/FINAL_REPORT.md"
    GRAPHICS_DIR = "outputs/graphics/"
    RESULTS_DIR = "outputs/results/"

    # User interaction prompts
    PROMPT_OPTION = "\nOption: "
    PRESS_ENTER = "\nPress Enter to continue..."

    # Execution settings
    # 20 minutes (EDA can take longer with large datasets)
    SCRIPT_TIMEOUT = 1200

    # Required libraries
    REQUIRED_LIBRARIES = [
        'pandas', 'numpy', 'sklearn', 'matplotlib',
        'seaborn', 'scipy'
    ]

    # Expected outputs after each step
    EXPECTED_OUTPUTS: Dict[int, List[str]] = {
        0: [MONTHLY_DATASET, GRAPHICS_DIR],
        1: [MODELING_DATASET, PROCESSED_DATA_DIR],
        2: [MODELS_DIR],
        3: [METRICS_FILE, RESULTS_DIR],
        4: ["outputs/graphics/prev_vs_real.png"],
        5: ["outputs/graphics/residuos_hist.png", "outputs/graphics/residuos_vs_previstos.png",
            "outputs/results/residuos_stats.md"],
        6: [FINAL_REPORT]
    }


# ==============================================================================
# SECTION 2: LOGGING FUNCTIONS
# ==============================================================================


def log(message: str, level: str = "INFO") -> None:
    """
    Write message to log file with timestamp.

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, CRITICAL, SUCCESS)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def print_colored(
        message: str,
        color: str = Fore.RESET,
        bright: bool = False) -> None:
    """
    Print colored message if colorama available.

    Args:
        message: Message to print
        color: Foreground color
        bright: Whether to use bright/bold style
    """
    if COLORS_AVAILABLE:
        style = Style.BRIGHT if bright else ""
        print(f"{style}{color}{message}{Style.RESET_ALL}")
    else:
        print(message)


def print_header(title: str) -> None:
    """
    Print formatted header.

    Args:
        title: Header title to display
    """
    print()
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print_colored(title.center(80), Fore.CYAN, bright=True)
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print()


def print_separator() -> None:
    """Print separator line."""
    print_colored("─" * 80, Fore.CYAN)


# ==============================================================================
# SECTION 3: DEPENDENCY CHECKING
# ==============================================================================


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    print_header("CHECKING DEPENDENCIES")
    log("Starting dependency check")

    missing = []

    for lib in Config.REQUIRED_LIBRARIES:
        try:
            __import__(lib)
            print_colored(f"  ✓ {lib}", Fore.GREEN)
        except ImportError:
            print_colored(f"  ✗ {lib} - NOT FOUND", Fore.RED)
            missing.append(lib)

    if missing:
        print()
        print_colored(
            f"[WARN]  Missing libraries: {', '.join(missing)}",
            Fore.YELLOW,
            bright=True)
        print_colored(
            f"Install with: pip install {' '.join(missing)}", Fore.YELLOW)
        log(f"Missing libraries: {missing}", "WARNING")
        return False

    print()
    print_colored("✓ All dependencies installed", Fore.GREEN, bright=True)
    log("All dependencies OK")
    return True


def check_input_data() -> bool:
    """
    Check if input data exists.

    Returns:
        bool: True if data found or user chooses to continue, False to abort
    """
    print_header("CHECKING INPUT DATA")
    log("Checking input data")

    # Check for input folder
    input_dir = Path('inputs')

    if input_dir.exists():
        csv_files = list(input_dir.glob('*.csv'))
        if csv_files:
            print_colored(
                f"✓ Found {len(csv_files)} CSV files in inputs/", Fore.GREEN)

            # Check specifically for expected files
            required_file = input_dir / 'leituras_unificadas.csv'
            weather_file = input_dir / 'weather_data_montijo.csv'

            if required_file.exists():
                print_colored(
                    f"  ✓ Main dataset: {
                        required_file.name}",
                    Fore.GREEN)
            else:
                print_colored(
                    f"  ⚠ Main dataset not found: {
                        required_file.name}", Fore.YELLOW)

            if weather_file.exists():
                print_colored(
                    f"  ✓ Weather data: {
                        weather_file.name}",
                    Fore.GREEN)
            else:
                print_colored(
                    f"  ⚠ Weather data not found (optional): {
                        weather_file.name}", Fore.YELLOW)

            log(f"Found {len(csv_files)} CSV files in inputs/")
            return True

    # Data not found
    print_colored("✗ Input data not found", Fore.RED)
    print()
    print_colored("Dataset not found. Options:", Fore.YELLOW)
    print("  [1] Specify path to data folder")
    print("  [2] Continue anyway (maybe data is already processed)")
    print("  [3] Abort")

    choice = input(Config.PROMPT_OPTION).strip()

    if choice == '1':
        custom_path = input("Enter path to data folder: ").strip()
        path = Path(custom_path)
        if path.exists() and any(path.glob('*.csv')):
            print_colored(f"✓ Data found at {custom_path}", Fore.GREEN)
            log(f"Using custom data path: {custom_path}")
            return True
        else:
            print_colored("✗ No data found at specified path", Fore.RED)
            return False

    elif choice == '2':
        print_colored("Continuing without input data check", Fore.YELLOW)
        log("User chose to continue without input data")
        return True

    else:
        print_colored("Aborting", Fore.RED)
        log("User aborted due to missing input data")
        return False


# ==============================================================================
# SECTION 4: SCRIPT EXECUTION
# ==============================================================================


def run_script(
        script_index: int,
        script_name: str,
        script_path: str) -> Tuple[bool, str]:
    """
    Execute a single pipeline script.

    Args:
        script_index: Index of script (0-based)
        script_name: Descriptive name of script
        script_path: Path to Python script

    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    print_separator()
    print_colored(
        f"[{script_index + 1}/{len(Config.SCRIPTS)}] Running: {script_name}",
        Fore.CYAN,
        bright=True)
    print_colored(f"Script: {script_path}", Fore.CYAN)
    print_separator()

    log(f"Starting script: {script_path}")
    start_time = time.time()

    try:
        # Run script
        subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            timeout=Config.SCRIPT_TIMEOUT,
            check=True
        )

        elapsed = time.time() - start_time
        print()
        print_colored(
            f"✓ {script_name} completed in {elapsed:.2f}s",
            Fore.GREEN,
            bright=True)
        log(f"Script completed: {script_path} ({elapsed:.2f}s)", "SUCCESS")

        # Check expected outputs
        check_outputs(script_index)

        return True, ""

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        error_msg = f"Script timeout after {
            elapsed:.0f}s (limit: {
            Config.SCRIPT_TIMEOUT}s)"
        print_colored(f"\n✗ {error_msg}", Fore.RED, bright=True)
        log(f"Script timeout: {script_path} ({elapsed:.0f}s)", "ERROR")
        return False, error_msg

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        error_msg = f"Script failed with return code {e.returncode}"
        print_colored(f"\n✗ {error_msg}", Fore.RED, bright=True)
        log(
            f"Script failed: {script_path} (code {
                e.returncode}, {
                elapsed:.2f}s)",
            "ERROR")
        return False, error_msg

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        print_colored(f"\n✗ {error_msg}", Fore.RED, bright=True)
        log(f"Script error: {script_path} - {str(e)} ({elapsed:.2f}s)", "ERROR")
        return False, error_msg


def check_outputs(script_index: int) -> None:
    """
    Check if expected outputs were created by script.

    Args:
        script_index: Index of script that just ran
    """
    if script_index not in Config.EXPECTED_OUTPUTS:
        return

    expected = Config.EXPECTED_OUTPUTS[script_index]
    print()
    print_colored("Checking outputs:", Fore.YELLOW)

    all_found = True
    for output_path in expected:
        path = Path(output_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024**2:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024**2):.1f} MB"
                print_colored(f"  ✓ {output_path} ({size_str})", Fore.GREEN)
            else:
                items = len(list(path.iterdir()))
                print_colored(f"  ✓ {output_path} ({items} items)", Fore.GREEN)
        else:
            print_colored(f"  ⚠ {output_path} (not found)", Fore.YELLOW)
            all_found = False

    if all_found:
        log(f"All expected outputs found for script {script_index}")
    else:
        log(f"Some outputs missing for script {script_index}", "WARNING")


def run_pipeline(selected_steps: Optional[List[int]] = None) -> None:
    """
    Run complete pipeline or selected steps.

    Args:
        selected_steps: List of step indices to run (None = all steps)
    """
    if selected_steps is None:
        steps_to_run = list(range(len(Config.SCRIPTS)))
        print_header("RUNNING COMPLETE PIPELINE")
    else:
        steps_to_run = selected_steps
        print_header("RUNNING SELECTED STEPS")

    log("=" * 80)
    log("PIPELINE EXECUTION STARTED")
    log("=" * 80)

    pipeline_start = time.time()
    successful_steps = []
    failed_steps = []

    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path(Config.GRAPHICS_DIR).mkdir(exist_ok=True, parents=True)
    Path(Config.RESULTS_DIR).mkdir(exist_ok=True, parents=True)
    Path(Config.MODELS_DIR).mkdir(exist_ok=True, parents=True)
    Path(Config.PROCESSED_DATA_DIR).mkdir(exist_ok=True, parents=True)

    # Execute selected steps
    for step_index in steps_to_run:
        script = Config.SCRIPTS[step_index]
        name = Config.SCRIPT_NAMES[step_index]

        success, error_msg = run_script(step_index, name, script)

        if success:
            successful_steps.append((step_index, name))
        else:
            failed_steps.append((step_index, name, error_msg))
            print()
            print_colored(
                f"[ERROR] Pipeline stopped at step {step_index + 1}",
                Fore.RED,
                bright=True)
            print_colored("Fix errors before continuing", Fore.YELLOW)
            break

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start
    print_header("PIPELINE EXECUTION SUMMARY")

    print_colored("Successful steps:", Fore.GREEN, bright=True)
    for idx, name in successful_steps:
        print_colored(f"  ✓ [{idx + 1}] {name}", Fore.GREEN)

    if failed_steps:
        print()
        print_colored("Failed steps:", Fore.RED, bright=True)
        for idx, name, error in failed_steps:
            print_colored(f"  ✗ [{idx + 1}] {name}", Fore.RED)
            print_colored(f"      Error: {error}", Fore.RED)

    print()
    print_colored(
        f"Total execution time: {pipeline_elapsed / 60:.2f} minutes",
        Fore.CYAN,
        bright=True)

    if not failed_steps:
        print()
        print_colored(
            "✓ PIPELINE COMPLETED SUCCESSFULLY!",
            Fore.GREEN,
            bright=True)
        print_colored(f"Final report: {Config.FINAL_REPORT}", Fore.CYAN)
        log("=" * 80)
        log("PIPELINE COMPLETED SUCCESSFULLY")
        log("=" * 80)
    else:
        log("=" * 80)
        log(f"PIPELINE FAILED at step {failed_steps[0][0] + 1}")
        log("=" * 80)


# ==============================================================================
# SECTION 5: INTERACTIVE MENU
# ==============================================================================


def show_main_menu() -> str:
    """
    Display main menu and get user choice.

    Returns:
        str: User's menu choice
    """
    print_header("ELECTRICITY BILL PREDICTION - PIPELINE ORCHESTRATOR")

    print("Choose an option:")
    print()
    print("  [1] Run complete pipeline (all 7 steps)")
    print("  [2] Run specific steps")
    print("  [3] Show pipeline overview")
    print("  [4] Clean outputs")
    print("  [5] Exit")
    print()

    return input(Config.PROMPT_OPTION).strip()


def show_pipeline_overview() -> None:
    """Display detailed pipeline overview."""
    print_header("PIPELINE OVERVIEW")

    steps = [
        ("01_exploratory_analysis.py",
         "Exploratory Analysis",
         "• Loads daily electricity meter readings\n"
         "  • Calculates daily consumption from cumulative readings\n"
         "  • Aggregates to monthly consumption periods\n"
         "  • Integrates weather data (temperature, humidity)\n"
         "  • Creates comprehensive EDA visualizations\n"
         "  • Analyzes consumption patterns (temporal, seasonal, tariff)\n"
         "  • Saves monthly dataset (base_mensal.csv)"),

        ("02_preprocessing.py",
         "Preprocessing & Feature Engineering",
         "• Simulates partial month scenarios (days 10, 15, 20, 25)\n"
         "  • Calculates accumulated consumption features\n"
         "  • Engineers temporal and weather features\n"
         "  • Calculates billing components (Portuguese tariff rules)\n"
         "  • Creates train/test split (temporal validation)\n"
         "  • Applies scaling and encoding (prevents data leakage)\n"
         "  • Saves modeling dataset (dataset_modelagem.csv)\n"
         "  • Saves preprocessed arrays (X_train, X_test, y_train, y_test)"),

        ("03_train_models.py",
         "Model Training",
         "• Trains 9 regression models:\n"
         "    - Baseline (single feature)\n"
         "    - Linear Regression\n"
         "    - Ridge Regression (L2 regularization)\n"
         "    - Lasso Regression (L1 regularization)\n"
         "    - Polynomial Regression (degree 2)\n"
         "    - SVR Linear (Support Vector Regression)\n"
         "    - SVR RBF (Radial Basis Function kernel)\n"
         "    - Random Forest (300 trees)\n"
         "    - Gradient Boosting (optional)\n"
         "  • Hyperparameter tuning (GridSearchCV)\n"
         "  • Saves trained models and predictions\n"
         "  • Records training times"),

        ("04_evaluate_metrics.py",
         "Metrics Evaluation",
         "• Calculates performance metrics:\n"
         "    - RMSE (Root Mean Squared Error)\n"
         "    - MAE (Mean Absolute Error)\n"
         "    - R² (Coefficient of Determination)\n"
         "    - MAPE (Mean Absolute Percentage Error)\n"
         "  • Creates comparative metrics table\n"
         "  • Ranks models by performance\n"
         "  • Identifies best model\n"
         "  • Calculates feature importance (tree-based models)\n"
         "  • Saves results to CSV and pickle"),

        ("05_plot_predicted_vs_actual.py",
         "Predicted vs Actual Visualization",
         "• Loads best model and test data\n"
         "  • Creates scatter plot (predicted vs actual)\n"
         "  • Adds perfect prediction line (y=x)\n"
         "  • Displays key metrics (RMSE, MAE, R²)\n"
         "  • Saves high-resolution plot"),

        ("06_residual_analysis.py",
         "Residual Analysis",
         "• Calculates prediction residuals (errors)\n"
         "  • Statistical tests:\n"
         "    - Shapiro-Wilk (normality test)\n"
         "    - Breusch-Pagan (heteroscedasticity test)\n"
         "  • Creates residual plots:\n"
         "    - Residuals vs Predicted\n"
         "    - Residual distribution histogram\n"
         "    - Q-Q plot (normality check)\n"
         "  • Identifies extreme errors and outliers\n"
         "  • Generates detailed analysis report (Markdown)"),

        ("07_final_report.py",
         "Final Report",
         "• Aggregates all pipeline results\n"
         "  • Compiles comprehensive Markdown report:\n"
         "    - Project introduction and objectives\n"
         "    - EDA findings and insights\n"
         "    - Preprocessing methodology\n"
         "    - Model descriptions and rationale\n"
         "    - Performance evaluation and comparison\n"
         "    - Best model analysis\n"
         "    - Residual analysis interpretation\n"
         "    - Feature importance insights\n"
         "    - Conclusions and recommendations\n"
         "  • Provides deployment guidelines\n"
         "  • Suggests improvement strategies")
    ]

    for i, (script, name, description) in enumerate(steps, 1):
        print(f"{i}. {name}")
        print(f"   Script: {script}")
        print(f"   {description}\n")

    input(Config.PRESS_ENTER)


def select_specific_steps() -> Optional[List[int]]:
    """
    Allow user to select specific steps.

    Returns:
        Optional[List[int]]: List of selected step indices, or None if invalid
    """
    print_header("SELECT STEPS TO EXECUTE")

    for i, name in enumerate(Config.SCRIPT_NAMES, 1):
        print(f"  [{i}] {name}")

    print("\nEnter step numbers separated by commas (e.g., 1,3,4):")
    selection = input("Steps: ").strip()

    try:
        steps = [int(s.strip()) - 1 for s in selection.split(',')]
        steps = [s for s in steps if 0 <= s < len(Config.SCRIPTS)]

        if not steps:
            print_colored("Invalid selection", Fore.RED)
            return None

        print("\nSelected steps:")
        for i in steps:
            print(f"  • {Config.SCRIPT_NAMES[i]}")

        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            return steps

    except Exception:
        print_colored("Invalid input", Fore.RED)

    return None


def clean_outputs() -> None:
    """Clean all output files and folders."""
    print_header("CLEAN OUTPUTS")

    items_to_clean = [
        "outputs/",
        Config.LOG_FILE
    ]

    print("The following items will be deleted:")
    existing_items = []
    for item in items_to_clean:
        if Path(item).exists():
            print_colored(f"  • {item}", Fore.YELLOW)
            existing_items.append(item)

    if not existing_items:
        print_colored("No output files found to clean", Fore.GREEN)
        return

    print()
    confirm = input("ARE YOU SURE? (yes/no): ").strip().lower()

    if confirm == 'yes':
        for item in existing_items:
            path = Path(item)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print_colored(f"  ✓ Deleted {item}", Fore.GREEN)
                elif path.is_file():
                    path.unlink()
                    print_colored(f"  ✓ Deleted {item}", Fore.GREEN)
            except Exception as e:
                print_colored(f"  ✗ Error deleting {item}: {e}", Fore.RED)

        print()
        print_colored("✓ Cleanup completed", Fore.GREEN, bright=True)
        log("Outputs cleaned by user")
    else:
        print_colored("Cleanup cancelled", Fore.YELLOW)


# ==============================================================================
# SECTION 6: COMMAND LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Orchestrator for ML Pipeline - Electricity Bill Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 08_orchestrator.py              # Interactive mode
  python 08_orchestrator.py --all        # Run complete pipeline
  python 08_orchestrator.py --steps 1,3  # Run steps 1 and 3
  python 08_orchestrator.py --clean      # Clean all outputs
        """)

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (non-interactive)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Run specific steps (comma-separated, e.g., 1,3,4)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean all outputs'
    )

    parser.add_argument(
        '--silent',
        action='store_true',
        help='Silent mode (no user interaction)'
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and data checks'
    )

    return parser.parse_args()


# ==============================================================================
# SECTION 7: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main orchestrator function.

    Handles both interactive and command-line modes for running the ML pipeline.
    """
    args = parse_arguments()

    # Handle command line arguments
    if args.clean:
        clean_outputs()
        return

    if args.all or args.steps or args.silent:
        # Non-interactive mode
        log("Starting in non-interactive mode")

        if not args.skip_checks:
            if not check_dependencies():
                print_colored(
                    "\n[WARN]  Missing dependencies. Use --skip-checks to bypass.",
                    Fore.RED)
                sys.exit(1)

        if args.all:
            run_pipeline()
        elif args.steps:
            try:
                steps = [int(s.strip()) - 1 for s in args.steps.split(',')]
                steps = [s for s in steps if 0 <= s < len(Config.SCRIPTS)]
                if not steps:
                    print_colored("Invalid steps specified", Fore.RED)
                    sys.exit(1)
                run_pipeline(steps)
            except Exception as e:
                print_colored(f"Invalid steps format: {e}", Fore.RED)
                sys.exit(1)
        return

    # Interactive mode
    log("Starting in interactive mode")
    print_colored(
        "\nWelcome to Electricity Bill Prediction Pipeline Orchestrator!",
        Fore.CYAN,
        bright=True)

    # Check dependencies
    if not check_dependencies():
        print()
        cont = input("Continue anyway? (y/n): ").strip().lower()
        if cont != 'y':
            sys.exit(1)

    # Check input data
    if not check_input_data():
        sys.exit(1)

    # Main menu loop
    while True:
        choice = show_main_menu()

        if choice == '1':
            run_pipeline()
            break
        elif choice == '2':
            steps = select_specific_steps()
            if steps:
                run_pipeline(steps)
            break
        elif choice == '3':
            show_pipeline_overview()
        elif choice == '4':
            clean_outputs()
            input(Config.PRESS_ENTER)
        elif choice == '5':
            print_colored("\nGoodbye!", Fore.CYAN)
            break
        else:
            print_colored("Invalid option", Fore.RED)
            input(Config.PRESS_ENTER)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("\n[WARN]  Execution interrupted by user", Fore.YELLOW)
        log("Execution interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n✗ Unexpected error: {e}", Fore.RED)
        log(f"Unexpected error: {e}", "CRITICAL")
        import traceback
        traceback.print_exc()
        sys.exit(1)
