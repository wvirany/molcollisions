from dataclasses import dataclass

from molcollisions.fingerprint import MolecularFingerprint


@dataclass
class BOExperiment:
    """Contains BO experiment configuration."""

    # Experiment details
    target: str
    fingerprint: MolecularFingerprint
    pool_size: int = 10000
    n_init: int = 1000
    budget: int = 1000
    acquisition_func: str = "ei"
    epsilon: float = 0.01

    trial_id: int = 0
    seed: int = 42

    # SLURM job parameters
    n_trials: int = 1
    time: str = "8:00:00"
    mem: str = "16G"


@dataclass
class BOResults:
    """Contains regression experiment results."""

    experiment: BOExperiment


def single_bo_trial(experiment: BOExperiment) -> BOResults:
    pass
