import jax.numpy as jnp

from molcollisions.fingerprints import CompressedFP, MolecularFingerprint, SparseFP


def optimize_params():
    raise NotImplementedError


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


def fp_from_str(fp: str) -> MolecularFingerprint:
    """
    Convert fingerprint config from string to MolecularFingerprint objects.

    Args:
        fp: Input takes the form {fp_type}{fp_size}-r{radius}, e.g.
        - "sparse-r2",
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
    if fp_type == "sparse":
        return SparseFP(radius=radius)
    elif fp_type.startswith("compressed"):
        # Extract size from something like "compressed2048"
        fp_size = int(fp_type[10:])
        return CompressedFP(fp_size=fp_size, radius=radius)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
