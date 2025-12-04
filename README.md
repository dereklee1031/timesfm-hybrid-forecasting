# Macro Forecasting with TimesFM Inputs

This project aims to forecasting weekly S&P 500 log returns by blending macroeconomic indicators with TimesFM scenario predictions. The repository contains:

- Data collectors (`src/fetch_data.py`, `src/fred_utils.py`)
- Hybrid vs. historical-only training pipelines (transformer/CNN, GELU MLP, simple ReLU MLP, Ridge)
- Walk-forward evaluation utilities that log RMSE, R², and directional accuracy for every rolling window

All models use 16-week contexts, scenario noise, weight decay, early stopping (patience = 15), and per-window fine-tuning.

---

## 1. Requirements

1. Python 3.10+  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. A [FRED](https://fred.stlouisfed.org/) API key and internet access (for Yahoo Finance + FRED).

---

## 2. Configuration

Create `config.txt` in the project root with your FRED key:

```
FRED_API_KEY=YOUR_FRED_KEY_HERE
```

The fetch script reads this file and exports the key automatically.

---

## 3. Data Pipeline

1. **Download SP500 and FRED series**
   ```bash
   python src/fetch_data.py
   ```
   - `data/raw/sp500.csv` (Yahoo Finance history)
   - `data/raw/*.csv` for each FRED indicator
   - `data/processed/merged.csv` (aligned SP500 + macro frame)

2. **TimesFM predictions**
   - Place backtested TimesFM CSVs inside `data/predictions/` matching the filenames expected by each model (e.g., `dgs10_backtest.csv`). These are required for hybrid pipelines.

---

## 4. Training & Evaluation

All scripts share the same CLI style: historical variants only use truncated `_true` columns; hybrid variants train on truths but roll forward with TimesFM `_pred` features. Key commands:

### 4.1 Historical-only pipelines

```bash
# Transformer (historical truths only)
python src/transformer_hist.py --run-name hist_only_best

# GELU MLP (historical)
python src/gelu_MLP_hist.py --run-name gelu_hist

# Simple ReLU MLP (historical)
python src/simple_MLP_hist.py --run-name simple_hist

# Ridge baseline (historical)
python src/ridge_hist.py --run-name ridge_hist --alpha 1.0
```

### 4.2 TimesFM (deployable) pipelines

```bash
# Transformer/CNN + TimesFM predictions
python src/transformer.py --run-name hybrid_tfm

# GELU MLP hybrid
python src/gelu_MLP.py --run-name gelu_hybrid

# Simple MLP hybrid
python src/simple_MLP.py --run-name simple_hybrid

# Ridge hybrid
python src/ridge.py --run-name ridge_hybrid --alpha 1.0
```

> Each script supports grid sweeps via `--grid` (transformer) or built-in sweeps (MLPs). After the sweep, the best configuration is retrained on train + val before walk-forward testing.

### 4.3 Outputs

For every run:

- `results/rolling_predictions*.csv` and `results/rolling_metrics*.csv` (or `results_mlp_*/...` for MLPs)
- Hyperparameter logs, e.g., `results/hyperparam_runs_hist.csv`
- Diagnostic plots: prediction curves, residuals, reconstructed SP500 levels

Directional accuracy is computed window-wise as `mean(sign(y_true) == sign(y_pred))` and aggregated across all test windows for comparisons.

---

## 5. Reproducing Reported Runs

1. Run the data pipeline.
2. Execute the hybrid GELU MLP (best deployable DirAcc):
   ```bash
   python src/gelu_MLP.py --run-name gelu_best
   ```
3. Execute the historical GELU MLP (upper bound):
   ```bash
   python src/gelu_MLP_hist.py --run-name gelu_hist_best
   ```
4. Compare outputs in `results_mlp_gelu/` and `results/hyperparam_runs_hist.csv`.

---

## 6. Troubleshooting

- **Missing FRED key**: ensure `config.txt` is filled or `FRED_API_KEY` is exported.
- **TimesFM files**: verify every base in `BASES` has a `_backtest.csv` entry with `_true` and `_pred` columns.
- **Long Ridge fits**: ridge scripts re-fit after each window; runs take longer than neural models—let them finish.
- **Negative R²**: short-horizon returns have tiny variance; we focus on directional accuracy for actionable signals.

Feel free to adjust the hyperparameter grids or fine-tuning schedule to explore additional configurations.

> **Note**: Most code was authored with AI assistance under human supervision, with only small portions written manually.
