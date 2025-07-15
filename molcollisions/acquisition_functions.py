import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from tanimoto_gp import FixedTanimotoGP, TanimotoGP_Params


def ei(X: jnp.ndarray, gp: FixedTanimotoGP, gp_params: TanimotoGP_Params, epsilon: float = 0.01):
    """
    Computes the expected improvement (EI) at points X
    using a fitted GP surrogate model

    Returns:
        Index of unobserved data which maximizes EI
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)
    std = jnp.sqrt(var)

    # Find incumbent by taking posterior mean at best observed point
    best_idx = jnp.argmax(gp._y_train)
    best_x = [gp.smiles_train[best_idx]]
    incumbent_mean, _ = gp.predict_y(gp_params, best_x, full_covar=False, from_train=True)
    incumbent = incumbent_mean[0]

    # Compute improvement
    improvement = mean - incumbent - epsilon

    # Compute Z-score
    z = improvement / std

    # Compute EI
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    # Handle numerical issues - if std < 1e-10, set EI to 0
    ei = ei.at[std < 1e-10].set(0)

    return jnp.argmax(ei)


def ucb(X: jnp.ndarray, gp: FixedTanimotoGP, gp_params: TanimotoGP_Params, beta: float = 0.1):
    """
    Computes the upper confidence bound (UCB) at points X
    using a fitted GP surrogate model

    Returns:
        Index of unobserved data which maximizes UCB
    """

    # Get mean and standard deviation predictions
    mean, var = gp.predict_y(gp_params, X, full_covar=False)

    # Calculate upper confidence bound
    ucb = mean + beta * np.sqrt(var)

    return np.argmax(ucb)


def uniform(X: jnp.ndarray):
    """
    Randomly chooses elements of X according to uniform distribution.
    """

    return np.random.randint(len(X))
