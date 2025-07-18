import numpy as np
from scipy import stats
from tanimoto_gp import FixedTanimotoGP, TanimotoGP_Params


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

    best = []
    top10 = []
    y_init = y.copy()

    best.append(np.max(y_observed))
    top10.append(find_top10_avg(y_observed))

    for i in range(1, num_iters + 1):

        print(f"Iter: {i} | Current best: {np.max(best):0.3f} | Top 10: {top10[-1]:0.3f}")

        # Get index of acquired data point
        idx = acq_func(X, gp, gp_params, epsilon)

        # Remove data point from candidate pool and add to observed sets
        X_new = X[idx]
        X = np.delete(X, idx, axis=0)
        X_observed = np.append(X_observed, X_new)

        y_new = y[idx]
        y = np.delete(y, idx, axis=0)
        y_observed = np.append(y_observed, y_new)

        percentile = stats.percentileofscore(y_init, y_new)
        print(f"Observed function value: {y_new:0.3f} | Percentile: {percentile:0.3f}")

        # Compute metrics
        best.append(np.max(y_observed))
        top10.append(find_top10_avg(y_observed))

        # Update GP
        gp.add_observation(gp_params, idx, y_new)

    print(f"Best observed molecule: {np.max(best):0.3f} | Top 10: {top10[-1]}")
    return best, top10, X_observed, y_observed, gp


def find_top10_avg(x):
    """Given a list x, find average of top10 values"""

    if x.size < 10:
        raise ValueError("Size of array must be larger than 10")

    indices = np.argpartition(x, -10)[-10:]
    top10 = x[indices]

    return np.mean(top10)
