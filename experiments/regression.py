import argparse
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jax.numpy as jnp
import yaml
from tanimoto_gp import ConstantMeanTanimotoGP, TanimotoGP_Params

from molcollisions.datasets import Dockstring
from molcollisions.fingerprints import MolecularFingerprint
from molcollisions.utils import fp_from_str, inverse_softplus, optimize_gp_params


@dataclass
class RegressionExperiment:
    """Contains regression experiment configuration."""

    # Experiment details
    target: str
    fingerprint: MolecularFingerprint
    n_train: int = 10000
    optimize_hp: bool = False

    trial_id: int = 0
    seed: int = 42

    # SLURM job parameters
    n_trials: int = 1
    time: str = "8:00:00"
    mem: str = "16G"


@dataclass
class RegressionResults:
    """Contains regression experiment results."""

    experiment: RegressionExperiment
    mean_preds: jnp.ndarray
    cov_preds: jnp.ndarray
    gp_params: TanimotoGP_Params

    def save(self):
        """Save experiment results to pickle file."""

        results = {
            "mean_preds": self.mean_preds,
            "cov_preds": self.cov_preds,
            "gp_params": self.gp_params,
        }

        # Build results path
        fp_config = self.experiment.fingerprint.get_fp_type()
        results_path = Path("results") / "regression" / self.experiment.target / fp_config

        # Save results to pickle file
        trial_file = results_path / f"trial_{self.experiment.trial_id:02d}.pkl"
        os.makedirs(os.path.dirname(trial_file), exist_ok=True)
        with open(trial_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {trial_file}")


def create_experiments_from_yaml(config_file: Path) -> List[RegressionExperiment]:
    """Create list of experiments from YAML config file."""

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    experiments = []

    for target in config["targets"]:
        for n_train in config["n_train"]:
            for optimize_hp in config["optimize_hp"]:
                for fp_config in config["fingerprints"]:
                    exp = RegressionExperiment(
                        target=target,
                        fingerprint=fp_from_str(fp_config),
                        optimize_hp=optimize_hp,
                        n_train=n_train,
                        n_trials=config["n_trials"],
                        time=config["time"],
                        mem=config["mem"],
                    )
                    experiments.append(exp)

    return experiments


def submit_slurm_jobs(
    experiments: List[RegressionExperiment],
    save_results: bool = False,
):
    """Submit SLURM array job for each experiment configuration."""

    # Read the template
    with open("templates/regression_job.sh", "r") as f:
        template = f.read()

    for exp in experiments:
        # Create unique job name for this experiment
        fp_config = exp.fingerprint.get_fp_type()
        job_name = fp_config
        if exp.optimize_hp:
            job_name += "-opt"

        # Create log directory
        log_dir = Path("logs") / "regression" / f"{exp.target}" / job_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Fill in SLURM template
        script_content = template.format(
            job_name=job_name,
            n_trials=exp.n_trials - 1,  # Number of jobs in array is 0-indexed
            time=exp.time,
            mem=exp.mem,
            log_dir=log_dir,
            target=exp.target,
            fp_config=fp_config,
            n_train=exp.n_train,
            optimize_hp_flag=" --optimize_hp" if exp.optimize_hp else "",
            save_results_flag=" --save_results" if save_results else "",
        )

        # Write and submit temporary job script
        script_path = f"tmp/temp_{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        print(f"Submitting job: {exp.target}/{job_name}")
        subprocess.run(["sbatch", script_path], check=True)

        # Remove temporary job script
        os.remove(script_path)
        time.sleep(1)


def single_regression_trial(experiment: RegressionExperiment) -> RegressionResults:
    """Run a single regression trial."""

    dataset = Dockstring(target=experiment.target, n_train=experiment.n_train, seed=experiment.seed)
    smiles_train, smiles_test, y_train, y_test = dataset.load()
    print(f"Data loaded - Train: {len(smiles_train)}, Test: {len(smiles_test)}")

    # Initialize GP Parameters
    amp = jnp.var(y_train)
    noise = 1e-2 * amp
    train_mean = jnp.mean(y_train)

    # Pre-softplus parameters (for numerical stability)
    gp_params = TanimotoGP_Params(
        raw_amplitude=inverse_softplus(amp), raw_noise=inverse_softplus(noise), mean=train_mean
    )

    print("Building GP...")
    gp = ConstantMeanTanimotoGP(experiment.fingerprint, smiles_train, y_train)

    # Optimize GP hyperparameters wrt MLL
    if experiment.optimize_hp:
        print("Optimizing hyperparameters...")
        gp_params = optimize_gp_params(gp, gp_params)

    # Make predictions
    print("Making predictions...")
    mean, cov = gp.predict_y(gp_params, smiles_test, full_covar=False)

    return RegressionResults(
        experiment=experiment,
        mean_preds=mean,
        cov_preds=cov,
        gp_params=gp_params,
    )


def main(
    config: str = None,
    submit: bool = False,
    target: str = "PARP1",
    fp_config: str = "sparse-r2",
    n_train: int = 10000,
    optimize_hp: bool = False,
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
        # Convert string-formatted fp config to MolecularFingerprint object
        fp = fp_from_str(fp_config)

        experiment = RegressionExperiment(
            target=target, fingerprint=fp, n_train=n_train, optimize_hp=optimize_hp
        )

        # Initialize trial ID as SLURM array ID
        slurm_array_id = os.getenv("SLURM_ARRAY_TASK_ID")

        if slurm_array_id is not None:
            experiment.trial_id = int(slurm_array_id)
            experiment.seed = int(slurm_array_id)

        results = single_regression_trial(experiment)

        if save_results:
            results.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Submit a YAML config file
    parser.add_argument("--config", type=str, help="YAML config for experiment generation")
    parser.add_argument(
        "--submit", action="store_true", help="Submit SLURM jobs (use with --config)"
    )

    # Submit individual experiment parameters for a single trial
    parser.add_argument("--target", type=str, default="PARP1")
    parser.add_argument("--fp_config", type=str, default="sparse-r2")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--optimize_hp", action="store_true")

    # Option to save results
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()

    # Currently only allowing one config of n_train to be saved to keep 'results/' directory clean
    if args.save_results and args.n_train != 10000:
        raise ValueError(f"Only saving results for n_train = 10000. Got n_train = {args.n_train}")

    main(
        config=args.config,
        submit=args.submit,
        target=args.target,
        fp_config=args.fp_config,
        n_train=args.n_train,
        optimize_hp=args.optimize_hp,
        save_results=args.save_results,
    )
