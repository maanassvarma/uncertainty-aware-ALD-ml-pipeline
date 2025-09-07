
# Uncertainty-Aware ALD ML Pipeline

A reproducible machine learning pipeline for **atomic layer deposition (ALD) process modeling**.  
This independent project rebuilds and extends earlier MATLAB prototypes into a modern Python workflow, adding **uncertainty quantification**, **cross-validation**, and **process-aware feature engineering**.

---

## Features
- **Multi-output ML**: MLP and Random Forest regressors trained jointly on Deposited Thickness, Refractive Index, and Wet Etch Rate.
- **Reproducibility**: Config-driven, fixed seeds, and versioned outputs.
- **Uncertainty quantification**: Bootstrap ensembles provide ±1σ predictive intervals.
- **Interpretability**: Permutation importance highlights wafer/process drivers.
- **Process-aware features**: Radial wafer geometry, quadratic and interaction terms.

---

## Example Results
- **Random Forest** achieved strong predictive performance:
  - Deposited Thickness: R² ≈ 0.93, RMSE ≈ 0.64
  - Refractive Index: R² ≈ 0.91, RMSE ≈ 0.03
  - Wet Etch Rate: R² ≈ 0.76, RMSE ≈ 0.13
- **MLP baseline** was unstable on this dataset, showing the need for careful tuning and model selection.
- **Key drivers** (from RF importance): Precursor Temperature, Deposition Temperature, and their quadratic terms.

<p align="center">
  <img src="outputs/perm_importance_top10.png" width="450"><br>
  <em>Top-10 permutation importances (Random Forest)</em>
</p>

---

## Quick Start

### 1. Clone & enter
```bash
git clone <your-repo-url>.git
cd uncertainty-aware-ALD-ml-pipeline
````

### 2. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add dataset

Place your Excel dataset (e.g. `ald_dataset.xlsx`) under `./data/`, then update
`src/ald_rebuild_config.yaml` → `paths.dataset_candidates`.

### 4. Run pipeline

```bash
bash run.sh
```

### 5. Outputs

All artifacts are saved to `./outputs/`:

* `metrics_cv.json` — cross-validation metrics
* `fold_predictions.parquet` — CV predictions vs truth
* `bootstrap_uncertainty.parquet` — predictions with ±σ intervals
* `permutation_importance.csv` — feature importances
* `*.png` — quick plots (metrics, uncertainty, importance)

---



