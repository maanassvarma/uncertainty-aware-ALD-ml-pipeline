#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ALD ML Rebuild Quickstart
# Adds:
#   (1) Deep modeling: scikit-learn MLPRegressor (multi-output) + bootstrap ensemble for uncertainty bands
#   (2) Robust pipeline: K-fold CV, config-driven, fixed seeds, versioned outputs
#   (3) Process-aware features: radial distance, interactions, quadratics; permutation importance (RF)
# No external heavy deps (TensorFlow/torch).
#
# Usage:
#   python ald_rebuild_quickstart.py --config /mnt/data/ald_rebuild_config.yaml
#
# Outputs (written under output_dir):
#   - metrics_cv.json: per-target CV scores (RMSE, R2) for MLP and RF baselines
#   - bootstrap_uncertainty.parquet: mean/std across ensemble on test fold predictions
#   - permutation_importance.csv: feature importances from RF
#   - fold_predictions.parquet: per-fold predictions vs truth
#
# Note:
#   Assumes the Excel dataset with the columns used in previous scripts.

import argparse
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=UserWarning)

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_dataset(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"Dataset not found in candidates: {candidates}")

def clean_and_standardize_columns(df):
    cols = ['Deposition Temperature', 'Precursor Temperature', 'x-axis Location', 'y-axis Location',
            'Deposited Thickness', 'Refractive Index', 'Wet Etch Rate']
    if len(df.columns) >= 7:
        df = df.copy()
        df.columns = cols + list(df.columns[len(cols):])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    return df[cols]

def engineer_features(df, cfg):
    X = df[['Deposition Temperature', 'Precursor Temperature', 'x-axis Location', 'y-axis Location']].copy()
    if cfg.get('radial_r', True):
        X['radial_r'] = np.sqrt((X['x-axis Location']**2) + (X['y-axis Location']**2))
    if cfg.get('r2', True):
        X['r2'] = X.get('radial_r', np.sqrt((X['x-axis Location']**2) + (X['y-axis Location']**2))) ** 2
    if cfg.get('xy_interaction', True):
        X['xy_inter'] = X['x-axis Location'] * X['y-axis Location']
    if cfg.get('temps_interaction', True):
        X['temp_inter'] = X['Deposition Temperature'] * X['Precursor Temperature']
    if cfg.get('quad_terms', True):
        for base in ['Deposition Temperature', 'Precursor Temperature', 'x-axis Location', 'y-axis Location']:
            X[f'{base}^2'] = X[base]**2
    return X

def build_models(cfg_models, seed):
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(cfg_models['mlp'].get('hidden_layer_sizes', [128, 64, 32])),
        activation=cfg_models['mlp'].get('activation', 'relu'),
        alpha=cfg_models['mlp'].get('alpha', 5e-4),
        max_iter=cfg_models['mlp'].get('max_iter', 800),
        early_stopping=cfg_models['mlp'].get('early_stopping', True),
        random_state=seed
    )
    rf = RandomForestRegressor(
        n_estimators=cfg_models['rf'].get('n_estimators', 300),
        max_depth=cfg_models['rf'].get('max_depth', 10),
        n_jobs=cfg_models['rf'].get('n_jobs', -1),
        random_state=seed
    )
    return mlp, rf

def rmse_cols(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))

def kfold_evaluate(X, Y, mlp, rf, k=5, seed=42, outdir=Path(".")):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    targets = list(Y.columns)
    fold_preds = []
    metrics = {'MLP': {'RMSE': [], 'R2': []}, 'RF': {'RMSE': [], 'R2': []}}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        Ytr, Yte = Y.iloc[train_idx], Y.iloc[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        mlp_model = MLPRegressor(**mlp.get_params())
        mlp_model.random_state = seed + fold
        mlp_model.fit(Xtr_s, Ytr)
        yp_mlp = mlp_model.predict(Xte_s)

        rf_model = RandomForestRegressor(**rf.get_params())
        rf_model.random_state = seed + fold
        rf_model.fit(Xtr, Ytr)
        yp_rf = rf_model.predict(Xte)

        rmse_mlp = rmse_cols(Yte.values, yp_mlp)
        rmse_rf  = rmse_cols(Yte.values, yp_rf)
        r2_mlp = [r2_score(Yte.iloc[:, j], yp_mlp[:, j]) for j in range(Y.shape[1])]
        r2_rf  = [r2_score(Yte.iloc[:, j], yp_rf[:, j]) for j in range(Y.shape[1])]

        metrics['MLP']['RMSE'].append(dict(zip(targets, map(float, rmse_mlp))))
        metrics['MLP']['R2'].append(dict(zip(targets, map(float, r2_mlp))))
        metrics['RF']['RMSE'].append(dict(zip(targets, map(float, rmse_rf))))
        metrics['RF']['R2'].append(dict(zip(targets, map(float, r2_rf))))

        df_fold = pd.DataFrame({
            'fold': fold,
            **{f'y_true_{t}': Yte[t].values for t in targets},
            **{f'y_mlp_{t}': yp_mlp[:, j] for j, t in enumerate(targets)},
            **{f'y_rf_{t}': yp_rf[:, j] for j, t in enumerate(targets)},
        })
        fold_preds.append(df_fold)

    preds = pd.concat(fold_preds, ignore_index=True)
    preds.to_parquet(outdir / "fold_predictions.parquet", index=False)

    def agg(metric_list):
        df = pd.DataFrame(metric_list)
        means = df.mean().to_dict()
        stds  = df.std(ddof=1).to_dict()
        return {'mean': means, 'std': stds}

    summary = {
        'targets': targets,
        'MLP': {'RMSE': agg(metrics['MLP']['RMSE']), 'R2': agg(metrics['MLP']['R2'])},
        'RF':  {'RMSE': agg(metrics['RF']['RMSE']),  'R2': agg(metrics['RF']['R2'])}
    }
    (outdir / "metrics_cv.json").write_text(json.dumps(summary, indent=2))
    return summary

def bootstrap_uncertainty(Xtr, Ytr, Xte, base_mlp, n_models=15, sample_frac=0.75, replace=True, seed=42):
    rng = np.random.RandomState(seed)
    preds = []
    scaler = StandardScaler()
    scaler.fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    n = max(1, int(np.ceil(sample_frac * len(Xtr))))
    for b in range(n_models):
        idx = rng.choice(len(Xtr), size=n, replace=replace)
        Xb = Xtr_s[idx]
        Yb = Ytr.iloc[idx]
        model = MLPRegressor(**base_mlp.get_params())
        model.random_state = seed + 100 + b
        model.max_iter = max(400, base_mlp.max_iter // 2)
        model.fit(Xb, Yb)
        preds.append(model.predict(Xte_s))

    P = np.stack(preds, axis=0)  # (B, N, T)
    mean = P.mean(axis=0)
    std  = P.std(axis=0, ddof=1)
    return mean, std

def rf_permutation_importance(X, Y, rf_model, seed=42):
    rf = RandomForestRegressor(**rf_model.get_params())
    rf.random_state = seed
    rf.fit(X, Y)
    result = permutation_importance(rf, X, Y, n_repeats=10, random_state=seed, n_jobs=-1, scoring='r2')
    imp = pd.DataFrame({'feature': X.columns, 'importance_mean': result.importances_mean, 'importance_std': result.importances_std})
    return imp.sort_values('importance_mean', ascending=False)

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.get('seed', 42))

    outdir = Path(cfg['paths']['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset(cfg['paths']['dataset_candidates'])
    df = pd.read_excel(dataset_path)
    df = clean_and_standardize_columns(df)

    targets = cfg['targets']
    eng_cfg = cfg.get('engineered_features', {})
    X = engineer_features(df, eng_cfg)
    Y = df[targets].copy()

    mlp, rf = build_models(cfg['models'], seed=cfg.get('seed', 42))

    summary = kfold_evaluate(X, Y, mlp, rf, k=cfg.get('kfolds', 5), seed=cfg.get('seed', 42), outdir=outdir)
    print(json.dumps(summary, indent=2))

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=cfg.get('seed', 42))
    mean_pred, std_pred = bootstrap_uncertainty(
        Xtr, Ytr, Xte, mlp,
        n_models=cfg['bootstrap']['n_models'],
        sample_frac=cfg['bootstrap']['sample_frac'],
        replace=cfg['bootstrap']['replace'],
        seed=cfg.get('seed', 42)
    )
    df_unc = pd.DataFrame({
        **{f'y_true_{c}': Yte[c].values for c in Y.columns},
        **{f'y_mean_{c}': mean_pred[:, j] for j, c in enumerate(Y.columns)},
        **{f'y_std_{c}': std_pred[:, j] for j, c in enumerate(Y.columns)},
    })
    df_unc.to_parquet(outdir / "bootstrap_uncertainty.parquet", index=False)

    imp = rf_permutation_importance(X, Y, rf_model=rf, seed=cfg.get('seed', 42))
    imp.to_csv(outdir / "permutation_importance.csv", index=False)

    print(f"\nWrote outputs to: {outdir.resolve()}")
    print("Artifacts: metrics_cv.json, fold_predictions.parquet, bootstrap_uncertainty.parquet, permutation_importance.csv")

if __name__ == "__main__":
    main()