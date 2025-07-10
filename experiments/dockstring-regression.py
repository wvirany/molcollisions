import argparse
import os

import jax.numpy as jnp
import tanimoto_gp
from configs.regression_config import RegressionExperiment, RegressionResults
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tanimoto_gp import ConstantMeanTanimotoGP

from molcollisions.datasets import Dockstring
from molcollisions.utils import fp_from_str, inverse_softplus, optimize_params


def run_single_trial(experiment: RegressionExperiment) -> RegressionResults:
    """Run a single regression trial."""

    dataset = Dockstring(target=experiment.target, n_train=experiment.n_train, seed=experiment.seed)
    smiles_train, smiles_test, y_train, y_test = dataset.load()
    print(f"Data loaded - Train: {len(smiles_train)}, Test: {len(smiles_test)}")

    # Initialize GP Parameters
    amp = jnp.var(y_train)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_train)

    # Pre-softplus parameters (for numerical stability)
    gp_params = tanimoto_gp.TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    print("Building GP...")
    gp = ConstantMeanTanimotoGP(experiment.fingerprint, smiles_train, y_train)

    # Optimize GP hyperparameters wrt MLL
    if experiment.optimize_hp:
        print("Optimizing hyperparameters...")
        gp_params = optimize_params(gp, gp_params)

    # Make predictions
    print("Making predictions...")
    mean, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)

    # Compute metrics
    r2 = r2_score(y_test, mean)
    mse = mean_squared_error(y_test, mean)
    mae = mean_absolute_error(y_test, mean)

    return RegressionResults(experiment=experiment, r2=r2, mse=mse, mae=mae, gp_params=gp_params)


def main(
    target: str = "PARP1",
    fp_config: str = "sparse-r2",
    n_train: int = 10000,
    optimize_hp: bool = False,
    save_results: bool = False,
):

    # Convert string-formatted fp config to MolecularFingerprint object
    fp = fp_from_str(fp_config)

    experiment = RegressionExperiment(
        target=target, fingerprint=fp, n_train=n_train, optimize_hp=optimize_hp
    )

    # Initialize trial ID as SLURM array ID
    slurm_array_id = os.getenv("SLURM_ARRAY_TASK_ID")

    if slurm_array_id is not None:
        trial_id = int(slurm_array_id)
        experiment.trial_id = trial_id
        experiment.seed = trial_id

    results = run_single_trial(experiment)

    if save_results:
        results.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="PARP1")
    parser.add_argument("--fp_config", type=str, default="sparse-r2")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--optimize_hp", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    main(
        target=args.target,
        fp_config=args.fp_config,
        n_train=args.n_train,
        optimize_hp=args.optimize_hp,
        save_results=args.save_results,
    )
