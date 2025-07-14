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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tanimoto_gp import ConstantMeanTanimotoGP, TanimotoGP_Params

from molcollisions.datasets import Dockstring
from molcollisions.fingerprints import MolecularFingerprint
from molcollisions.utils import fp_from_str, inverse_softplus, optimize_params


@dataclass
class RegressionExperiment:
    """Contains experiment configuration."""

    # Core experiment details
    target: str
    fingerprint: MolecularFingerprint
    n_train: int = 10000

    # Experiment settings
    optimize_hp: bool = False
    trial_id: int = 0
    seed: int = 42


@dataclass
class RegressionResults:
    """Contains regression experiment results."""

    experiment: RegressionExperiment
    r2: float
    mse: float
    mae: float
    gp_params: TanimotoGP_Params

    def save(self):
        """Save experiment results to pickle file."""

        results = {"r2": self.r2, "mse": self.mse, "mae": self.mae, "gp_params": self.gp_params}

        # Build results path
        fp_config = self.experiment.fingerprint.get_fp_type()
        results_path = Path("results") / "regression" / self.experiment.target / fp_config

        # Save results to pickle file
        trial_file = results_path / f"trial_{self.experiment.trial_id:03d}.pkl"

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
                    )
                    experiments.append(exp)

    return experiments


def submit_slurm_jobs(
    experiments: List[RegressionExperiment],
    save_results: bool = False,
    n_trials: int = 30,
):
    """Submit SLURM array jobs for each experiment configuration."""

    for i, experiment in enumerate(experiments):
        # Create unique job name for this experiment
        job_name = f"regression-{experiment.target}-{experiment.fingerprint.get_fp_type()}"
        if experiment.optimize_hp:
            job_name += "-opt"

        # Create log directory
        log_dir = Path("logs") / "regression" / job_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Build Python command
        python_cmd = f"""python regression.py \\
            --target {experiment.target} \\
            --fp_config {experiment.fingerprint.get_fp_type()} \\
            --n_train {experiment.n_train} \\
            {"--optimize_hp" if experiment.optimize_hp else ''} \\
            {"--save_results" if save_results else ''}"""

        # Submit SLURM array job
        slurm_cmd = f"""sbatch \\
            --job-name={job_name} \\
            --array=0-{n_trials-1} \\
            --partition=amilan \\
            --qos=normal \\
            --time=8:00:00 \\
            --nodes=1 \\
            --mem=128G \\
            --output={log_dir}/%a.out \\
            --wrap="module purge && module load python && module load anaconda && \\
                    conda activate molcollisions && \\
                    {python_cmd}" """

        print(f"Submitting: {job_name}")
        subprocess.run(slurm_cmd, shell=True, check=True)
        time.sleep(1)


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
    gp_params = TanimotoGP_Params(
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

        results = run_single_trial(experiment)
        print(f"Results: r2: {results.r2:.3f} | mse: {results.mse:.3f} | mae: {results.mae:.3f}")

        if save_results:
            results.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Submit a YAML config file with
    parser.add_argument("--config", type=str, help="YAML config for experiment generation")
    parser.add_argument(
        "--submit", action="store_true", help="Submit SLURM jobs (use with --config)"
    )

    # Submit individual experiment parameters for a single trial
    parser.add_argument("--target", type=str, default="PARP1")
    parser.add_argument("--fp_config", type=str, default="sparse-r2")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--optimize_hp", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    main(
        config=args.config,
        submit=args.submit,
        target=args.target,
        fp_config=args.fp_config,
        n_train=args.n_train,
        optimize_hp=args.optimize_hp,
        save_results=args.save_results,
    )
