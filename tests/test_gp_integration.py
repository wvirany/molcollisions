"""
Basic integration test for fingerprint / GP interface.

Simple test to ensure our fingerprint classes work with tanimoto_gp.
"""

import jax.numpy as jnp
import numpy as np
import tanimoto_gp

from molcollisions.fingerprints import CompressedFP, SparseFP

# Simple dataset
smiles = ["CCO", "CCC", "CC"]
y = np.array([1.0, -0.5, 0.8])
smiles_test = ["CCCO"]


def test_sparse_fp_integration():
    """Test that SparseFP works with ConstantMeanTanimotoGP"""

    # Create fingerprint and GP
    fp_func = SparseFP(radius=2, count=True)
    gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles, y)

    # Basic checks for initialization
    assert gp._smiles_train == smiles
    assert gp._K_train_train.shape == (3, 3)


def test_compressed_fp_integration():
    """Test that CompressedFP works with ConstantMeanTanimotoGP"""

    # Create fingerprint and GP
    fp_func = CompressedFP(radius=2, count=True, fp_size=2048)
    gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles, y)

    # Basic checks for initialization
    assert gp._smiles_train == smiles
    assert gp._K_train_train.shape == (3, 3)


def test_gp_prediction():
    """Test that GP can make a simple prediction."""

    fp_func = SparseFP(radius=2, count=True)
    gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles, y)

    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=jnp.array(1.0), raw_noise=jnp.array(-2.0), mean=jnp.array(0.0)
    )

    # Make prediction
    mean_pred, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    # Check it's finite
    assert jnp.isfinite(mean_pred).all()


def test_fixed_gp():
    """Test that FixedTanimotoGP achieves the same results as ConstantMeanTanimotoGP"""

    fp_func = SparseFP(radius=2, count=True)

    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=jnp.array(1.0), raw_noise=jnp.array(-2.0), mean=jnp.array(0.0)
    )

    constant_mean_gp = tanimoto_gp.ConstantMeanTanimotoGP(fp_func, smiles, y)
    fixed_gp = tanimoto_gp.FixedTanimotoGP(gp_params, fp_func, smiles, y)

    const_gp_pred, _ = constant_mean_gp.predict_y(gp_params, smiles_test, full_covar=False)
    fixed_gp_pred, _ = fixed_gp.predict_y(
        gp_params, smiles_test, full_covar=False, from_train=False
    )

    assert np.allclose(const_gp_pred, fixed_gp_pred, rtol=1e-10)
