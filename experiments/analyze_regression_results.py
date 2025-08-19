import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from molcollisions.datasets import Dockstring


def load_predictions(target: str = "ESR2", fp_config: str = "exact-r2", optimize_hp: bool = False):
    """Load/aggregate predictions for a specific configuration across multiple files."""

    suffix = "-opt" if optimize_hp else ""
    results_path = Path("results/regression") / target / (fp_config + suffix)

    if not os.path.exists(results_path):
        raise ValueError(f"Results path does not exist: {results_path}")

    # Find all trials
    trial_files = list(results_path.glob("trial_*.pkl"))

    if not trial_files:
        raise ValueError(f"No trial files found in {results_path}")

    # Load all results
    all_predictions = []
    for trial_file in trial_files:
        with open(trial_file, "rb") as f:
            preds = pickle.load(f)
            all_predictions.append(preds)

    return all_predictions


def compute_trial_statistics(target, predictions):
    """Compute mean/std statistics across trials."""

    dataset = Dockstring(target)
    _, _, _, y_test = dataset.load()

    # Extract metrics from each trial
    r2_values = []
    mse_values = []
    mae_values = []

    # Compute metrics
    for preds in predictions:
        mean_preds = preds["mean_preds"]
        r2_values.append(r2_score(y_test, mean_preds))
        mse_values.append(mean_squared_error(y_test, mean_preds))
        mae_values.append(mean_absolute_error(y_test, mean_preds))

    # Compute statistics
    r2_mean = float(np.mean(r2_values).round(3))
    r2_std = float(np.std(r2_values).round(3))
    mse_mean = float(np.mean(mse_values).round(3))
    mse_std = float(np.std(mse_values).round(3))
    mae_mean = float(np.mean(mae_values).round(3))
    mae_std = float(np.std(mae_values).round(3))

    results = {
        "R2 (mean, std)": (r2_mean, r2_std),
        "MSE (mean, std)": (mse_mean, mse_std),
        "MAE (mean, std)": (mae_mean, mae_std),
    }

    return results


def aggregate_all_results(config_file):
    """Aggregate results for all configurations in YAML file."""

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    results_list = []

    for target in config["targets"]:
        for fp_config in config["fingerprints"]:
            for optimize_hp in config["optimize_hp"]:
                try:
                    suffix = "-opt" if optimize_hp else ""
                    print(f"Processing results: {target}/{fp_config + suffix}...")

                    # Load all trial data
                    predictions = load_predictions(target, fp_config, optimize_hp=optimize_hp)

                    # Compute statistics across trials
                    stats = compute_trial_statistics(target, predictions)

                    # Add metadata
                    stats["target"] = target
                    stats["optimized params"] = optimize_hp
                    # stats['n_trials'] = len(predictions)

                    # Parse fingerprint type
                    fp = fp_config.split("-")[0]
                    if fp.startswith("exact"):
                        fp_type = "exact"
                        fp_size = None
                    elif fp.startswith("compressed"):
                        fp_type = "compressed"
                        fp_size = int(fp[10:])
                    elif fp.startswith("sortslice"):
                        fp_type = "sort&slice"
                        fp_size = int(fp[9:])

                    stats["fp type"] = fp_type
                    stats["fp size"] = fp_size

                    results_list.append(stats)

                except Exception as e:
                    print(f"Error processing {target}/{fp_config}: {e}")
                    continue

    # Convert to
    results_df = pd.DataFrame(results_list)

    # Define the desired column order
    column_order = [
        "target",
        "fp type",
        "fp size",
        "optimized params",
        "R2 (mean, std)",
        "MSE (mean, std)",
        "MAE (mean, std)",
    ]

    # Reorder the dataframe
    df_reordered = results_df[column_order]

    return df_reordered


def main():
    config_file = "configs/regression_experiments.yaml"

    print("Aggregating all regression results...")
    results_df = aggregate_all_results(config_file)

    csv_file = "results/regression_summary.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
