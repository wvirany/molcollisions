import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from tanimoto_gp import TanimotoGP_Params

from molcollisions.fingerprints import MolecularFingerprint


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
