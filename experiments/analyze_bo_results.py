import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from molcollisions.bo_utils import auc_score
from molcollisions.datasets import Dockstring


def load_bo_predictions(target: str = "ESR2", acq: str = "ei", fp_config: str = "exact-r2"):
    """Load/aggregate predictions for a specific configuration across multiple trials."""

    results_path = Path("results/bo") / target / acq / fp_config

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


def compute_trial_statistics(predictions, target: str = "ESR2", acq: str = "ei"):
    """Compute statistics for AUC score across trials."""

    dataset = Dockstring(target, n_train=1000000)
    _, _, y_train, y_test = dataset.load()
    y = np.concatenate([y_train, y_test])

    best_mol = np.min(y)
    auc_scores = []

    # Compute auc_score from each trial
    for pred in predictions:
        best = pred["best"]
        auc_scores.append(auc_score(best, best_mol))

    # Compute mean/std for AUC score
    auc_mean = float(np.mean(auc_scores).round(3))
    auc_std = float(np.std(auc_scores).round(3))

    results = {
        "AUC Score (mean, std)": (auc_mean, auc_std),
    }

    return results


def aggregate_all_results(config_file):
    """Aggregate results for all configurations in YAML file."""

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    results_list = []

    for target in config["targets"]:
        for acq in config["acq_funcs"]:
            for fp_config in config["fingerprints"]:
                try:
                    print(f"Processing results: {target}/{fp_config}...")

                    # Load all trial data
                    predictions = load_bo_predictions(target=target, acq=acq, fp_config=fp_config)

                    # Compute statistics across trials
                    stats = compute_trial_statistics(
                        predictions=predictions, target=target, acq=acq
                    )

                    # Add metadata
                    stats["target"] = target

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
        "AUC Score (mean, std)",
    ]

    # Reorder the dataframe
    df_reordered = results_df[column_order]

    return df_reordered


def main():
    config_file = "configs/bo_experiments.yaml"

    print("Aggregating all BO results...")
    results_df = aggregate_all_results(config_file)

    csv_file = "results/bo_summary.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
