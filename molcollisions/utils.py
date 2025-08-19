import jax
import jax.numpy as jnp
import optax
import tanimoto_gp

from molcollisions.fingerprints import CompressedFP, ExactFP, MolecularFingerprint, SortSliceFP


def optimize_gp_params(gp, gp_params, tol=1e-3, max_iters=10000):
    """
    Optimize GP parameters until convergence or max steps reached

    Args:
        gp: Gaussian Process instance
        gp_params: Initial parameters
        tol: Tolerance for convergence (default 1e-3)
        max_steps: Maximum optimization steps (default 10000)
    """

    # Compute minimum noise value, prevents numerical instablility due to small noise
    var_y = jnp.var(gp._y_train)
    min_noise = 1e-4 * var_y
    min_raw_noise = jnp.log(jnp.exp(min_noise) - 1)

    print(f"Start MLL: {gp.marginal_log_likelihood(params=gp_params)}")

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(gp_params)

    # Perform one step of gradient descent
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
        grad_norm = jnp.linalg.norm(jnp.array(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        min_noise_reached = False

        # Gradient clipping for stability
        noise = tanimoto_gp.TRANSFORM(params.raw_noise)
        if noise < min_noise:
            params = tanimoto_gp.TanimotoGP_Params(
                raw_amplitude=params.raw_amplitude, raw_noise=min_raw_noise, mean=gp_params.mean
            )
            min_noise_reached = True

        return params, opt_state, grad_norm, loss, min_noise_reached

    # Run optimization loop
    for i in range(max_iters):

        gp_params, opt_state, grad_norm, loss, min_noise_reached = step(gp_params, opt_state)

        if min_noise_reached:
            print("Minimum noise value reached, stopping early")
            break

        if grad_norm < tol:
            print(f"Converged after {i+1} steps, gradient norm = {grad_norm}")
            break

        if i % 1000 == 0:
            print(f"Iteration {i}:")
            print(f"  Loss: {loss}")
            print(f"  Gradient norm: {grad_norm}")
            print(f"  Params: {gp_params}")
            print(f"  Natural params: {natural_params(gp_params)}")

    print(f"End MLL (after optimization): {-loss}")
    print(f"End GP parameters (after optimization): {gp_params}")

    return gp_params


def natural_params(gp_params):
    """Returns the natural parameters (after softplus transform, positive values)"""
    return [
        float(tanimoto_gp.TRANSFORM(gp_params.raw_amplitude)),
        float(tanimoto_gp.TRANSFORM(gp_params.raw_noise)),
    ]


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


def fp_from_str(fp: str) -> MolecularFingerprint:
    """
    Convert fingerprint config from string to MolecularFingerprint objects.

    Args:
        fp: Input takes the form {fp_type}{fp_size}-r{radius}, e.g.
        - "exact-r2",
        - "compressed2048-r4",
        etc.

    Returns:
        - MolecularFingerprint object
    """

    parts = fp.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid fingerprint format: {fp}")

    fp_type, radius = parts

    # Extract radius (assuming format like "r2", "r4", etc.)
    if not radius.startswith("r"):
        raise ValueError(f"Invalid radius format: {radius}")

    radius = int(radius[1:])

    # Parse fingerprint type
    if fp_type == "exact":
        return ExactFP(radius=radius)
    elif fp_type.startswith("compressed"):
        # Extract size from something like "compressed2048"
        fp_size = int(fp_type[10:])
        return CompressedFP(fp_size=fp_size, radius=radius)
    elif fp_type.startswith("sortslice"):
        fp_size = int(fp_type[9:])
        return SortSliceFP(fp_size=fp_size, radius=radius)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
