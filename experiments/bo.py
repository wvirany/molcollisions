import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jax.numpy as jnp
import numpy as np
from tanimoto_gp import FixedTanimotoGP, TanimotoGP_Params

from molcollisions.acquisition import ei, ucb, uniform
from molcollisions.bo_utils import bo_loop, bo_split
from molcollisions.datasets import Dockstring
from molcollisions.fingerprints import MolecularFingerprint
from molcollisions.utils import fp_from_str, inverse_softplus


@dataclass
class BOExperiment:
    """Contains BO experiment configuration."""

    # Experiment details
    target: str
    fingerprint: MolecularFingerprint

    n_init: int = 1000
    budget: int = 1000
    acquisition_func: str = "ei"
    epsilon: float = 0.01

    trial_id: int = 0
    seed: int = 42

    # SLURM job parameters
    n_trials: int = 1
    time: str = "2:00:00"
    mem: str = "64G"


@dataclass
class BOResults:
    """Contains regression experiment results."""

    experiment: BOExperiment
    best: np.ndarray
    top10: np.ndarray
    X_observed: np.ndarray
    y_observed: np.ndarray
    gp_params: TanimotoGP_Params

    def save(self):
        """Save experiment results to pickle file."""

        results = {
            "best": self.best,
            "top10": self.top10,
            "X_observed": self.X_observed,
            "y_observed": self.y_observed,
            "gp_params": self.gp_params,
        }

        # Build results path
        fp_config = self.experiment.fingerprint.get_fp_type()
        results_path = Path("results") / "bo" / self.experiment.target / fp_config

        # Save results to pickle file
        trial_file = results_path / f"trial_{self.experiment.trial_id:02d}.pkl"
        os.makedirs(os.path.dirname(trial_file), exist_ok=True)
        with open(trial_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {trial_file}")


def create_experiments_from_yaml(config_file: Path) -> List[BOExperiment]:
    raise NotImplementedError("Need to implement create_experiments_from_yaml()")


def submit_slurm_jobs(experiments: List[BOExperiment], save_results: bool = False):
    raise NotImplementedError("Need to implement submit_slurm_jobs()")


def single_bo_trial(experiment: BOExperiment) -> BOResults:
    """Run a single BO trial."""

    dataset = Dockstring(target=experiment.target, seed=experiment.seed)
    smiles_train, smiles_test, y_train, y_test = dataset.load()

    X = np.concatenate([smiles_train, smiles_test])
    # Minimizing docking scores corresponds to maximizing negative docking scores
    y = -np.concatenate([y_train, y_test])

    # Create initial observed datasets and unobserved candidate pools
    X_init, X, y_init, y = bo_split(X, y, experiment.n_init, experiment.seed)

    # Initialize GP parameters
    amp = jnp.var(y_init)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_init)

    # Pre-softplus parameters (for numerical stability)
    gp_params = TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp),
        raw_noise=inverse_softplus(noise),
        mean=train_mean,
    )

    # Initialize GP
    print("Building GP...")
    gp = FixedTanimotoGP(gp_params, experiment.fingerprint, X_init, y_init)

    # Run BO procedure
    best, top10, X_observed, y_observed = bo_loop(
        X=X,
        y=y,
        X_observed=X_init,
        y_observed=y_init,
        gp=gp,
        gp_params=gp_params,
        acq_func=experiment.acquisition_func,
        epsilon=experiment.epsilon,
        num_iters=experiment.budget,
    )

    return BOResults(
        experiment=experiment,
        best=best,
        top10=top10,
        X_observed=X_observed,
        y_observed=y_observed,
        gp_params=gp_params,
    )


def main(
    config: str = None,
    submit: bool = False,
    target: str = "PARP1",
    fp_config: str = "sparse-r2",
    n_init: int = 1000,
    budget: int = 1000,
    acq_func: str = "ei",
    epsilon: float = 0.01,
    save_results: bool = False,
):

    if config:
        experiments = create_experiments_from_yaml(config)

        if submit:
            submit_slurm_jobs(experiments, save_results)
        else:
            print(f"Generated {len(experiments)} experiments (dry run)")
            for i, exp in enumerate(experiments):
                print(
                    f"  {i}: {exp.target}, {exp.fingerprint.get_fp_type()}, optimize_hp: {exp.optimize_hp}"
                )
    else:

        fp = fp_from_str(fp_config)

        experiment = BOExperiment(
            target=target, fingerprint=fp, n_init=n_init, budget=budget, epsilon=epsilon
        )

        if acq_func == "ei":
            experiment.acquisition_func = ei
        elif acq_func == "ucb":
            experiment.acusition_func = ucb
        elif acq_func == "uniform":
            experiment.acquisition_func = uniform
        else:
            print(f"Invalid acquisition function: {acq_func}")

        # Initialize trial ID as SLURM array ID
        slurm_array_id = os.getenv("SLURM_ARRAY_TASK_ID")

        if slurm_array_id is not None:
            experiment.trial_id = int(slurm_array_id)
            experiment.seed = int(slurm_array_id)

        results = single_bo_trial(experiment)

        if save_results:
            results.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Submit a YAML config file
    parser.add_argument("--config", type=str, help="YAML config for experiment generation")
    parser.add_argument("--submit", action="store_true")

    # Submit individual experiment parameters for a single trial
    parser.add_argument("--target", type=str, default="PARP1")
    parser.add_argument("--fp_config", type=str, default="sparse-r2")
    parser.add_argument("--n_init", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--acq_func", type=str, default="ei")
    parser.add_argument("--epsilon", type=float, default=0.01)

    # Option to save results
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()

    # I currently only allow one config of n_init, budget to be saved to keep 'results/' directory clean
    if args.save_results and (args.n_init != 1000 or args.budget != 1000):
        raise ValueError(
            f"Only saving results for n_init, budget = 1000. Got n_init = {args.n_init}, budget = {args.budget}"
        )

    main(
        config=args.config,
        submit=args.submit,
        target=args.target,
        fp_config=args.fp_config,
        n_init=args.n_init,
        budget=args.budget,
        acq_func=args.acq_func,
        epsilon=args.epsilon,
        save_results=args.save_results,
    )
