# TEMP: memory logging
import os

import numpy as np
import psutil
from scipy import stats
from sklearn.metrics import auc
from tanimoto_gp import FixedTanimotoGP, TanimotoGP_Params

from molcollisions import acquisition


def log_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")


def bo_loop(
    X: np.ndarray,
    y: np.ndarray,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    gp: FixedTanimotoGP,
    gp_params: TanimotoGP_Params,
    acq_func,
    epsilon: float,
    num_iters: int = 30,
):
    """
    Bayesian optimization loop.

    Args:
        X: Candidate pool (SMILES strings)
        y: True objective values for candidate pool
        X_observed: Initially observed SMILES strings
        y_observed: Initially observed objective values
        gp: GP model with cached training data
        gp_params: GP hyperparameters
        acq_func: Acquisition function (e.g., EI, UCB)
        epsilon: Acquisition function parameter
        num_iters: Number of BO iterations

    Returns:
        best: Best objective value at each iteration
        top10: Mean of top 10 objective values at each iteration
        X_observed: Final observed SMILES
        y_observed: Final observed objectives
        gp: Updated GP model
    """

    # Initialize metrics
    best = [np.max(y_observed)]
    top10 = [find_top10_avg(y_observed)]
    all_scores = np.concatenate(
        [y, y_observed]
    )  # This is used to compute percentiles in total dataset

    # Bool indicating if we are using UCB
    ucb = acq_func == acquisition.ucb

    for i in range(1, num_iters + 1):

        print(f"Iter: {i} | Current best: {np.max(best):0.3f} | Top 10: {top10[-1]:0.3f}")
        log_memory()

        # Set adaptive UCB parameter
        if ucb:
            epsilon = 10 / i

        # Get index of acquired data point
        idx = acq_func(X, gp, gp_params, epsilon)

        # Remove data point from candidate pool and add to observed sets
        X_new = X[idx]
        X = np.delete(X, idx, axis=0)
        X_observed = np.append(X_observed, X_new)

        y_new = y[idx]
        y = np.delete(y, idx, axis=0)
        y_observed = np.append(y_observed, y_new)

        percentile = stats.percentileofscore(all_scores, y_new)
        print(f"Observed function value: {y_new:0.3f} | Percentile: {percentile:0.3f}")

        # Compute metrics
        best.append(np.max(y_observed))
        top10.append(find_top10_avg(y_observed))

        # Update GP
        gp.add_observation(gp_params, idx, y_new)

    print(f"Best observed molecule: {np.max(best):0.3f} | Top 10: {top10[-1]}")
    return best, top10, X_observed, y_observed


def bo_split(X_full: np.ndarray, y_full: np.ndarray, n_init: int = 1000, trial_seed: int = 42):
    """
    Split dataset into initial / candidate pool for BO loop.

    Initial dataset is created by sampling n_init molecules from
    bottom 80% of dataset.

    Args:
        X_full: Entire candidate pool
        y_full: Docking scores corresponding to X_full
        n_init: Number of obsservations in initial dataset
        trial_seed: RNG random seed

    Returns:
        X_init: Initial observations
        X: Entire candidate pool minus X_init
        y_init: Docking scores of initial observations
        y: Docking scores corresponding to X
    """
    rng = np.random.RandomState(trial_seed)

    # Find docking score corresponding to 80th percentile
    cutoff = np.percentile(y_full, 80)

    # Sample n_init molecules from bottom 80% of dataset
    bottom_80_indices = np.where(y_full <= cutoff)[0]
    sampled_indices = rng.choice(bottom_80_indices, size=n_init, replace=False)

    # Get remaining molecules from dataset
    top_20_indices = np.where(y_full > cutoff)[0]
    bottom_80_complement = np.setdiff1d(bottom_80_indices, sampled_indices)
    full_complement = np.concatenate([bottom_80_complement, top_20_indices])

    # Create initial dataset / candidate pool
    X_init, y_init = X_full[sampled_indices], y_full[sampled_indices]
    X, y = X_full[full_complement], y_full[full_complement]

    return X_init.tolist(), X.tolist(), y_init, y


def find_top10_avg(x):
    """Given a list x, find average of top10 values"""

    if x.size < 10:
        raise ValueError("Size of array must be larger than 10")

    indices = np.argpartition(x, -10)[-10:]
    top10 = x[indices]

    return np.mean(top10)


def auc_score(best_values, max_val):
    """Compute AUC score for BO performance."""
    iterations = np.arange(len(best_values))

    # Normalize by max auc (i.e., best molecule observed first)
    max_area = -max_val * len(best_values)

    return auc(iterations, best_values) / max_area
