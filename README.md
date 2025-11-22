# MINDFULAI Internship — Sales Forecasting & Analysis

This repository contains a small pipeline for sales data preprocessing, anomaly detection, machine learning (classification & regression), and time-series forecasting (Prophet). The scripts produce diagnostic plots and cleaned datasets saved to `reports/` and the repository root.

**Quick overview:** run `main.py` or `main copy.py` to preprocess `data/train.csv`, detect and record anomalies, train models, and generate plots under `reports/`.

**Repository structure**
- `main.py`: primary pipeline (detects anomalies, keeps them for modeling).
- `main copy.py`: variant that removes detected anomalies before modeling.
- `data/train.csv`: expected input dataset (not included in this repo).
- `Cleaned_dataset.csv`: output produced by `main.py`.
- `Cleaned_dataset_no_anomalies.csv`: output produced by `main copy.py`.
- `reports/`: output folder for plots and `anomaly_records.csv`.
- `reports/anomaly_records.csv`: CSV of detected anomalous records.

**What this repo does**
- Load and preprocess sales data from `data/train.csv`.
- Detect anomalies and save them to `reports/anomaly_records.csv`.
- Train classification/regression models and evaluate results.
- Produce visual diagnostics (boxplots, scatter plots, confusion matrix, ROC curve, regression plots, and Prophet forecast plots).

Getting started
---------------

**Prerequisites**
- Python 3.8+ (3.10/3.11 recommended)
- `pip`

**Recommended packages**
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `prophet`

Install and setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet
```

If installing `prophet` on Windows fails, try one of:
- follow the official Prophet installation guide (may require C++ build tools or a supported wheel),
- use WSL or a Linux environment where Prophet wheels are easier to install.

Usage
-----

1. Ensure your dataset is at `data/train.csv` or update the `file_path` variable in the `main` function of `main.py` / `main copy.py` to the correct absolute path.

2. Run the pipeline you want:
```powershell
python main.py
# or
python "main copy.py"
```

Notes on scripts
- `main.py`: Detects anomalies but uses the original dataset when training models. Produces `Cleaned_dataset.csv`.
- `main copy.py`: Detects anomalies, removes them, and runs modeling on the cleaned data. Produces `Cleaned_dataset_no_anomalies.csv`.

Outputs
-------
- `reports/` (created if missing): PNG plots and `anomaly_records.csv`.
- `Cleaned_dataset.csv` or `Cleaned_dataset_no_anomalies.csv` in the repository root.

Troubleshooting & tips
- If plots or outputs are missing, confirm `data/train.csv` path and that the script has write permissions to the repo folder.
- Inspect `reports/anomaly_records.csv` to review which rows were flagged as anomalies.
- To make the project reproducible, consider generating a `requirements.txt` with `pip freeze > requirements.txt` after installing packages.

Suggested next improvements (optional)
- Add a small CLI (argparse / click) to choose `--remove-anomalies` and set `--data-path`.
- Add `requirements.txt` or `pyproject.toml` for reproducible installs.
- Merge the best parts of `main copy.py` into a single, well-documented `main.py`.
- Add unit tests for the preprocessing and anomaly detection functions.

Contact / Next steps
-------------------
If you'd like, I can:
- unify both scripts into a single CLI-driven script,
- add a `requirements.txt` and instructions for Windows & WSL installs,
- or run the scripts here (I will need a copy of `data/train.csv` or the correct absolute path).

Enjoy exploring the data — tell me which next step you prefer.