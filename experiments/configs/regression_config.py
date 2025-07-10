from dataclasses import dataclass

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
    seed: int = 42


@dataclass
class RegressionResults:
    """Contains regression experiment results."""

    experiment: RegressionExperiment

    r2: float
    mse: float
    mae: float

    gp_params: TanimotoGP_Params
