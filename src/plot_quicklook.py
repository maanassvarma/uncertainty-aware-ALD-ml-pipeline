import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_cv_metrics(metrics_path: Path, save_dir: Path):
    data = json.loads(metrics_path.read_text())
    for model in ["MLP", "RF"]:
        for metric in ["RMSE", "R2"]:
            means = data[model][metric]["mean"]
            stds  = data[model][metric]["std"]
            # Bar plot: mean with text of ±std
            plt.figure()
            keys = list(means.keys())
            vals = [means[k] for k in keys]
            plt.bar(keys, vals)
            plt.title(f"{model} {metric} (CV mean)")
            plt.ylabel(metric)
            for i, k in enumerate(keys):
                s = stds.get(k, 0.0)
                plt.text(i, vals[i], f"±{s:.3f}", ha="center", va="bottom", rotation=0)
            plt.tight_layout()
            plt.savefig(save_dir / f"cv_{model}_{metric}.png")
            plt.close()

def plot_uncertainty(parquet_path: Path, save_dir: Path):
    df = pd.read_parquet(parquet_path)
    # For each target, plot y_true vs y_mean with ±1σ bands
    for col in ["Deposited Thickness", "Refractive Index", "Wet Etch Rate"]:
        y_true = df[f"y_true_{col}"]
        y_mean = df[f"y_mean_{col}"]
        y_std  = df[f"y_std_{col}"]
        plt.figure()
        plt.scatter(range(len(y_true)), y_true)
        plt.plot(range(len(y_mean)), y_mean)
        # error band via vertical lines (to avoid setting colors/styles explicitly)
        for i in range(len(y_mean)):
            lo = y_mean.iloc[i] - y_std.iloc[i]
            hi = y_mean.iloc[i] + y_std.iloc[i]
            plt.vlines(i, lo, hi)
        plt.title(f"Prediction ±1σ — {col}")
        plt.xlabel("Sample index")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(save_dir / f"uncertainty_{col.replace(' ', '_')}.png")
        plt.close()

def plot_perm_importance(csv_path: Path, save_dir: Path):
    df = pd.read_csv(csv_path)
    # Take top 10 features
    df = df.sort_values("importance_mean", ascending=False).head(10)
    plt.figure()
    plt.barh(df["feature"], df["importance_mean"])
    plt.title("RF Permutation Importance (Top 10)")
    plt.xlabel("Importance (mean)")
    plt.tight_layout()
    plt.savefig(save_dir / "perm_importance_top10.png")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, required=True, help="Path to outputs/")
    args = ap.parse_args()
    save_dir = Path(args.artifacts)
    # Expected files
    metrics = save_dir / "metrics_cv.json"
    unc = save_dir / "bootstrap_uncertainty.parquet"
    imp = save_dir / "permutation_importance.csv"
    if metrics.exists():
        plot_cv_metrics(metrics, save_dir)
    if unc.exists():
        plot_uncertainty(unc, save_dir)
    if imp.exists():
        plot_perm_importance(imp, save_dir)

if __name__ == "__main__":
    main()