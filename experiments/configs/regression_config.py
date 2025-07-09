from dataclasses import dataclass


@dataclass
class RegressionExperiment:
    """Contains experiment configuration."""

    # Core experiment details
    dataset: str
    fingerprint: str
    n_train: int = 10000

    # Fingerprint configuration
    radius: int = 2
    fp_size: int = 2048
    sparse: bool = True

    # Experiment settings
    optimize_hp: bool = False
    n_trials: int = 30
    seed: int = 42


@dataclass
class RegressionResults:
    """Contains regression experiment results."""

    config: RegressionExperiment

    r2: float
    mse: float
    mae: float
