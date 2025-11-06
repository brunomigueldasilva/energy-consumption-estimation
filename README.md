# Electricity Bill Prediction

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Machine Learning - Supervised Learning (Regression Models)
**Machine Learning pipeline for predicting monthly electricity bills using partial month consumption data.**

This project demonstrates a complete supervised learning workflow for regression, transforming electricity meter readings and weather data into accurate bill predictions that enable proactive budget management and early cost forecasting.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## Description

This project implements a **regression system** to predict monthly electricity bills based on partial month consumption data. The system analyzes patterns from meter readings, weather data, temporal features, and Portuguese electricity tariff structures to make accurate predictions at different points during the billing cycle.

The complete machine learning pipeline includes:
- **Data collection** from electricity meter readings and weather data
- **Exploratory data analysis** with comprehensive consumption visualizations
- **Feature engineering** including multi-scenario cutoff days (10, 15, 20, 25)
- **Training of multiple regression models** with thorough evaluation
- **Comprehensive evaluation** with multiple metrics and residual analysis
- **Automated reporting** with predicted vs actual plots and insights

### Why This Project?

Electricity bills can be unpredictable, making budget management challenging. This project demonstrates how machine learning can:
- **Predict final bills early** using only partial month data
- **Enable proactive budgeting** by forecasting costs mid-month
- **Identify unusual consumption** patterns before billing cycle ends
- **Optimize energy usage** through data-driven insights
- **Support financial planning** with accurate cost projections

---

## Features

### Core Capabilities
- ✅ **Automated ML Pipeline**: 7-step workflow from data loading to final report
- ✅ **Multiple Regression Models**: Comparison of various algorithms
- ✅ **Multi-Scenario Training**: Predictions at 10, 15, 20, and 25 days into month
- ✅ **Comprehensive Evaluation**: MAE, RMSE, R², MAPE, and residual analysis
- ✅ **Visual Analytics**: Predicted vs actual plots, residual analysis, feature correlations
- ✅ **Production-Ready**: Includes orchestrator for automated execution
- ✅ **Detailed Reporting**: Auto-generated Markdown reports with insights

### Technical Features
- **Data Preprocessing**: Handles cumulative readings, calculates daily consumption, weather integration
- **Feature Engineering**: Temporal features, tariff periods, weather correlations, partial month scenarios
- **Model Serialization**: Saves trained models and scalers using pickle
- **Stratified Splitting**: Train/test split maintaining temporal integrity
- **Standardization**: Prevents data leakage by fitting only on training data
- **Reproducibility**: Seeded random states for consistent results

---

## Business Problem

### Challenge
How can we accurately predict the final monthly electricity bill using only partial month consumption data?

### Impact Areas

| Area | Benefit | Example |
|------|---------|---------|
| **Budget Management** | Early bill forecasting | Know expected costs by mid-month |
| **Anomaly Detection** | Identify unusual patterns | Catch billing errors or consumption spikes |
| **Cost Optimization** | Understand consumption drivers | Adjust usage based on predictions |
| **Financial Planning** | Reduce uncertainty | Better household budget control |
| **Tariff Analysis** | Optimize rate plans | Compare peak vs off-peak savings |

### Target Audience
- Homeowners managing energy budgets
- Energy efficiency consultants
- Utility companies offering forecasting services
- Data scientists learning regression techniques
- IoT/ML practitioners in energy sector

---

## Dataset

### Data Sources

The project uses **2 primary data files**:

1. **leituras_unificadas.csv** - Electricity meter readings
   - Cumulative kWh readings (off-peak and peak periods)
   - Daily timestamps
   - Meter reading states

2. **weather_data_montijo.csv** - Weather data
   - Daily temperature readings
   - Humidity levels
   - Weather conditions
   - Seasonal patterns

### Data Characteristics

- **Temporal Data**: Daily meter readings with timestamps
- **Cumulative Readings**: Converted to daily consumption
- **Multi-Tariff Structure**: Portuguese bi-hourly rate system (peak/off-peak)
- **Weather Integration**: Temperature and humidity correlations
- **Billing Rules**: Network access fees, audiovisual contribution, taxes, VAT

### Target Variable

- **Variable**: `preco_fatura` (monthly electricity bill in €)
- **Type**: Continuous numeric (regression target)
- **Range**: Varies based on consumption and tariff rates

### Engineered Features

The pipeline creates multiple cutoff scenarios:
- **10-day cutoff**: Predict full month bill using first 10 days
- **15-day cutoff**: Predict using first 15 days
- **20-day cutoff**: Predict using first 20 days  
- **25-day cutoff**: Predict using first 25 days

### Example Data Structure

```csv
ano,mes,consumo_vazio,consumo_ponta,temperatura_media,preco_fatura
2024,1,150.5,85.3,12.5,87.45
2024,2,145.2,82.1,13.8,84.20
```

---

## Installation

### Requirements

- **Python**: 3.13 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: ~500 MB - 2 GB (depends on dataset size)
- **Storage**: ~10MB for dependencies + your dataset size

### Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Core Libraries:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

**Optional (for enhanced output):**
```
colorama>=0.4.4  # Colored terminal output
```

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/electricity-bill-prediction.git
cd electricity-bill-prediction
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare your data**:
```bash
mkdir inputs
# Place your CSV files in the inputs/ folder
```

### Directory Structure

```
electricity-bill-prediction/
│
├── inputs/                      # Place your CSV files here
│   ├── leituras_unificadas.csv
│   └── weather_data_montijo.csv
│
├── 01_exploratory_analysis.py   # Step 1: EDA
├── 02_preprocessing.py          # Step 2: Data preprocessing
├── 03_train_models.py           # Step 3: Model training
├── 04_evaluate_metrics.py       # Step 4: Metrics evaluation
├── 05_plot_predicted_vs_actual.py # Step 5: Prediction plots
├── 06_residual_analysis.py      # Step 6: Residual analysis
├── 07_final_report.py           # Step 7: Final report
├── 08_orchestrator.py           # Pipeline orchestrator
│
├── outputs/                     # Generated files (auto-created)
│   ├── graphics/                # Generated plots (auto-created)
│   ├── data_processed/          # Processed datasets (auto-created)
│   ├── models/                  # Trained models (auto-created)
│   └── predictions/             # Model predictions (auto-created)
│
├── requirements.txt
├── README.md
├── QUICK_START.md
├── CONTRIBUTING.md
├── LICENSE
└── PROJECT_OVERVIEW.md
```

---

## Usage

### Quick Start

Run the complete pipeline with one command:

```bash
python 08_orchestrator.py --all
```

### Interactive Mode

Launch the interactive menu:

```bash
python 08_orchestrator.py
```

**Menu Options:**
1. Run complete pipeline (all 7 scripts)
2. Run specific steps (choose which ones)
3. Clean outputs folder
4. Exit

### Run Individual Steps

Execute specific parts of the pipeline:

```bash
# Step 1: Exploratory analysis
python 01_exploratory_analysis.py

# Step 2: Preprocessing
python 02_preprocessing.py

# Step 3: Train models
python 03_train_models.py

# Step 4: Evaluate metrics
python 04_evaluate_metrics.py

# Step 5: Plot predictions
python 05_plot_predicted_vs_actual.py

# Step 6: Residual analysis
python 06_residual_analysis.py

# Step 7: Generate report
python 07_final_report.py
```

### Advanced Options

Run specific steps using orchestrator:

```bash
# Run only steps 1, 3, and 7
python 08_orchestrator.py --steps 1,3,7

# Clean outputs before running
python 08_orchestrator.py --clean --all
```

---

## Pipeline Architecture

### Overview

The pipeline follows a 7-step supervised learning workflow optimized for regression tasks:

```
┌──────────────────────────────────────────────────────────────────┐
│                    ELECTRICITY BILL PREDICTION                    │
│                      ML REGRESSION PIPELINE                       │
└──────────────────────────────────────────────────────────────────┘

Step 1: Exploratory Analysis → Step 2: Preprocessing → Step 3: Train Models
                ↓                        ↓                      ↓
         Understand Data          Feature Engineer      Multiple Algorithms
         Visualizations           Clean & Scale         Model Comparison
                                                              ↓
Step 7: Final Report ← Step 6: Residual Analysis ← Step 5: Predictions ← Step 4: Evaluate
        ↓                        ↓                        ↓                    ↓
   Complete Report       Error Patterns         Predicted vs Actual      MAE, RMSE, R²
   Recommendations       Diagnostic Plots       Visualization            MAPE Analysis
```

---

### Step 1: Exploratory Data Analysis
**Script**: `01_exploratory_analysis.py`

- Loads electricity meter readings and weather data
- Converts cumulative readings to daily consumption
- Aggregates to monthly billing periods
- Creates comprehensive visualizations
- Analyzes consumption patterns and correlations

**Outputs**:
- `outputs/base_mensal.csv` - Monthly consolidated dataset
- `outputs/eda_stats.md` - Statistical summary
- `outputs/graphics/*.png` - EDA visualizations

---

### Step 2: Data Preprocessing
**Script**: `02_preprocessing.py`

- Engineers features for multiple cutoff scenarios
- Creates temporal and tariff features
- Handles missing values and outliers
- Splits data into train/test sets
- Fits and applies StandardScaler

**Outputs**:
- `outputs/data_processed/X_train.csv` - Training features
- `outputs/data_processed/X_test.csv` - Testing features
- `outputs/data_processed/y_train.csv` - Training targets
- `outputs/data_processed/y_test.csv` - Testing targets
- `outputs/data_processed/scaler.pkl` - Fitted scaler

---

### Step 3: Model Training
**Script**: `03_train_models.py`

- Trains multiple regression models
- Saves trained models using pickle
- Generates predictions on test set
- Compares model performance

**Outputs**:
- `outputs/models/*.pkl` - Trained model files
- `outputs/predictions/*_predictions.csv` - Model predictions

---

### Step 4: Metrics Evaluation
**Script**: `04_evaluate_metrics.py`

- Calculates regression metrics for all models
- Compares performance across algorithms
- Identifies best performing model
- Generates comparative metrics table

**Outputs**:
- `outputs/comparative_metrics.csv` - Performance comparison

**Metrics Calculated**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

---

### Step 5: Predicted vs Actual Plots
**Script**: `05_plot_predicted_vs_actual.py`

- Visualizes predicted vs actual values
- Creates scatter plots with regression lines
- Compares all models side-by-side
- Highlights prediction accuracy

**Outputs**:
- `outputs/graphics/predicted_vs_actual.png` - Comparison visualization

---

### Step 6: Residual Analysis
**Script**: `06_residual_analysis.py`

- Analyzes prediction errors (residuals)
- Creates diagnostic plots
- Identifies systematic errors
- Tests residual normality

**Outputs**:
- `outputs/graphics/residual_analysis.png` - Residual plots
- `outputs/graphics/residual_distribution.png` - Error distribution

---

### Step 7: Final Report
**Script**: `07_final_report.py`

- Generates comprehensive Markdown report
- Includes all metrics, visualizations, and insights
- Provides recommendations and next steps
- Summarizes best model and deployment strategy

**Outputs**:
- `outputs/FINAL_REPORT.md` - Complete analysis report

---

## Results

### Performance Overview

The system achieves strong performance in predicting monthly electricity bills:

| Metric | Expected Range |
|--------|----------------|
| **MAE** | €3-8 |
| **RMSE** | €5-12 |
| **R² Score** | 0.85-0.95 |
| **MAPE** | 5-15% |

*Note: Exact values depend on your specific dataset and billing complexity.*

### Key Insights

1. **Early predictions are feasible**: 15-20 day cutoffs provide accurate forecasts
2. **Seasonal patterns are strong**: Winter heating and summer cooling dominate
3. **Weather correlates with consumption**: Temperature is a key predictor
4. **Tariff structure matters**: Peak vs off-peak usage impacts final bill
5. **Multi-scenario approach works**: Training on multiple cutoffs improves robustness

### Error Analysis

- **Positive Residuals**: Model underestimates bill
  - Impact: Customer surprised by higher-than-predicted bill
  
- **Negative Residuals**: Model overestimates bill
  - Impact: Customer pleasantly surprised by lower bill

**Recommendation**: Slight overestimation bias is preferable to avoid budget shortfalls.

---

## Model Performance

### Model Comparison

Different regression algorithms offer various trade-offs:

#### Typical Models Evaluated
1. **Linear Regression** - Baseline, interpretable
2. **Ridge Regression** - Handles multicollinearity
3. **Lasso Regression** - Feature selection capability
4. **Random Forest** - Non-linear patterns
5. **Gradient Boosting** - High accuracy potential

### Recommendations

**For Production Deployment:**
- Use the model with lowest MAPE for percentage accuracy
- Consider ensemble methods combining multiple models
- Implement confidence intervals for predictions

**For Interpretability:**
- Use Linear/Ridge Regression with feature importance analysis

**For Best Performance:**
- Fine-tune Gradient Boosting or Random Forest
- Consider stacking multiple models

---

## Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Make your changes**
4. **Run tests** (if available)
5. **Commit your changes**:
   ```bash
   git commit -m "Add YourFeature"
   ```
6. **Push to the branch**:
   ```bash
   git push origin feature/YourFeature
   ```
7. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Include comments for complex logic
- Update README.md if adding new features
- Test your changes thoroughly

### Areas for Contribution

- **Feature Engineering**: Add new consumption or weather features
- **Model Improvements**: Implement deep learning or advanced algorithms
- **Hyperparameter Tuning**: Add GridSearchCV or Bayesian optimization
- **Visualization**: Create interactive dashboards
- **Documentation**: Improve explanations or add tutorials
- **Testing**: Add unit tests and integration tests
- **Deployment**: Create REST API or web interface

---

## Roadmap

### Current Version (v1.0)
✅ Complete 7-step ML pipeline  
✅ Multiple regression models  
✅ Comprehensive evaluation metrics  
✅ Automated reporting  
✅ Multi-scenario training (cutoff days)

### Planned Features (v1.1)
- [ ] Real-time API for bill prediction
- [ ] Web dashboard for visualization
- [ ] Mobile app integration
- [ ] Multi-household support
- [ ] Advanced ensemble methods
- [ ] Deep learning models (LSTM for time series)

### Future Enhancements (v2.0)
- [ ] Causal inference capabilities
- [ ] Reinforcement learning optimizer
- [ ] Automated tariff plan recommendations
- [ ] Integration with smart meters
- [ ] Anomaly detection system

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary:**
- ✅ Free to use for personal and commercial projects
- ✅ Free to modify and distribute
- ✅ Must include original copyright notice
- ❌ No warranty or liability

---

## Authors

**Bruno Silva**
- Project Lead & Developer
- Contact: bruno_m_c_silva@proton.me
- GitHub: [@brunomigueldasilva](https://github.com/brunomigueldasilva)

---

## Acknowledgments

### Inspiration
- **Energy Data Science Community** - For innovative forecasting approaches
- **Scikit-learn Developers** - For the excellent machine learning library
- **Open Data Initiatives** - For making energy and weather data accessible

### References
1. **Portuguese Electricity Tariff System** - Official regulatory documentation
2. **Energy Forecasting Best Practices** - Academic research papers
3. **Time Series Regression** - Statistical modeling literature

### Tools & Libraries
- **Python** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Scientific computing

### Special Thanks
- Thank you to the open-source community for making projects like this possible
- Thanks to everyone who contributes to improving this project
- Appreciation for Portuguese energy data transparency

---

## Project Status

**Current Status**: 🟢 Active Development

This project is actively maintained and accepting contributions. New features are regularly added based on community feedback and emerging best practices in ML.

**Last Updated**: November 2025

---

## Support

If you encounter any issues or have questions:

1. **Check the Documentation**: Read this README thoroughly
2. **Search Issues**: Look for similar issues in the issue tracker
3. **Ask Questions**: Open a new issue with the `question` label
4. **Report Bugs**: Open an issue with detailed information:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

**Community Support:**
- GitHub Discussions: Project Discussions
- Email: bruno_m_c_silva@proton.me

---

## Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ for the Energy Data Science community**
