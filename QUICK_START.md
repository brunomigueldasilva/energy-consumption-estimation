# Quick Start Guide

## Get Running in 5 Minutes

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy colorama
```

### 2. Prepare Your Data
Place your CSV files in an `inputs/` folder:
```
inputs/
├── leituras_unificadas.csv       # Electricity meter readings
└── weather_data_montijo.csv      # Weather data
```

### 3. Run the Pipeline
```bash
python 08_orchestrator.py --all
```

That's it! ✅

---

## What Happens Next?

The system will automatically:
1. ✅ Load and analyze electricity consumption data
2. ✅ Integrate weather patterns
3. ✅ Engineer features for multiple scenarios
4. ✅ Train multiple regression models
5. ✅ Evaluate performance metrics
6. ✅ Generate predicted vs actual plots
7. ✅ Perform residual analysis
8. ✅ Build a comprehensive report

**Total time**: ~3-7 minutes depending on dataset size

---

## Expected Outputs

After completion, you'll find:

```
📊 outputs/base_mensal.csv              # Monthly consolidated dataset
📁 outputs/data_processed/              # Preprocessed data & scaler
📁 outputs/models/                      # Trained ML models
📁 outputs/predictions/                 # Model predictions
📁 outputs/graphics/                    # Visualizations (plots, charts)
📄 outputs/comparative_metrics.csv      # Performance comparison
📄 outputs/eda_stats.md                 # Statistical summary
📄 outputs/FINAL_REPORT.md              # Complete analysis report
📄 execution.log                        # Execution log
```

---

## View Results

Open `outputs/FINAL_REPORT.md` to see:
- Model performance comparison
- Best model recommendation
- Predicted vs actual visualizations
- Residual analysis
- Cost optimization insights
- Next steps and recommendations

---

## Common Issues

### Issue: Missing CSV files
**Solution**: Make sure both CSV files are in the `inputs/` folder:
- `leituras_unificadas.csv`
- `weather_data_montijo.csv`

### Issue: Import errors
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Memory errors
**Solution**: Reduce dataset size or increase available RAM

### Issue: Date parsing errors
**Solution**: Ensure date columns are in ISO format (YYYY-MM-DD)

---

## Data Format Requirements

### leituras_unificadas.csv
Should contain columns like:
- Date/timestamp column
- Cumulative consumption readings
- Peak/off-peak meter values

### weather_data_montijo.csv
Should contain columns like:
- Date column
- Temperature readings
- Humidity values
- Weather conditions

---

## Next Steps

1. Review `outputs/FINAL_REPORT.md` for insights
2. Check which model performed best (lowest MAPE)
3. Analyze predicted vs actual plots
4. Review residual analysis for systematic errors
5. Consider implementing recommendations
6. Use model for monthly bill forecasting

---

## Understanding the Results

### Key Metrics
- **MAE**: Average prediction error in €
- **RMSE**: Penalizes larger errors more
- **R²**: How well model explains variance (0-1)
- **MAPE**: Percentage error (easier to interpret)

### What to Look For
- **R² > 0.85**: Good model fit
- **MAPE < 15%**: Acceptable accuracy
- **Low residual bias**: Model doesn't systematically over/underestimate
- **Normal residual distribution**: Model assumptions met

---

## Need Help?

- 📖 Read the full [README.md](README.md)
- 📊 Check [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for details
- 🐛 Report issues on GitHub
- 💬 Join discussions
- 📧 Email: bruno_m_c_silva@proton.me

---

## Advanced Usage

### Run Specific Steps
```bash
# Only exploratory analysis
python 01_exploratory_analysis.py

# Only preprocessing and training
python 02_preprocessing.py
python 03_train_models.py

# Only evaluation and reporting
python 04_evaluate_metrics.py
python 07_final_report.py
```

### Custom Orchestrator Options
```bash
# Run specific steps
python 08_orchestrator.py --steps 1,3,5,7

# Clean outputs before running
python 08_orchestrator.py --clean --all
```

---

**Happy forecasting! ⚡💰🚀**
