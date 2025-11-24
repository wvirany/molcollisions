# Hash Collisions in Molecular Fingerprints

Code to accompany the paper: [https://arxiv.org/abs/2511.17078](https://arxiv.org/abs/2511.17078)


# Development Setup

```bash
# Clone and install
git clone https://github.com/wvirany/molcollisions.git

cd molcollisions

pip install -e .
pre-commit install
```


## Dependencies

**Core libraries:**

```toml
dependencies = [
    "numpy",
    "pandas",
    "rdkit",
    "jax",
    "tanimoto_gp @ git+https://github.com/wvirany/tanimoto-gp.git@fixed-gp-stable",
    "kern_gp @ git+https://github.com/wvirany/kernel-only-gp.git@update-cholesky-stable",
]
```

This project uses custom implementations of existing packages:

* `tanimoto_gp` - Tanimoto kernel Gaussian processes (forked to provide caching for fast BO)
* `kern_gp` - Kernel computations with Cholesky updates (forked to provide efficient Cholesky factor updates for BO)

**Sort&Slice Fingerprint Dependency**

The `SortSliceFP` fingerprint class depends on the `ECFP-Sort-and-Slice` repository (see [here](https://github.com/MarkusFerdinandDablander/ECFP-Sort-and-Slice)). To set this up:

```bash
# Clone the ECFP-Sort-and-Slice repo
git clone https://github.com/MarkusFerdinandDablander/ECFP-Sort-and-Slice.git

# Add to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/ECFP-Sort-and-Slice
```

**Development tools:** `pytest`, `pre-commit`, `black`, `ruff`, `mypy`

# Experiments

Experiments can either be run programmatically or via the command line. The following demonstrates how to run regression experiments, but the interface is the same for BO experiments.

### Programmatic usage:

```py
from regression import RegressionExperiment, single_regression_trial
from molcollisions.fingerprints import ExactFP

# Create experiment
experiment = RegressionExperiment(
    target="PARP1",
    fingerprint=ExactFP(),
    n_train=100,
    optimize_hp=False
)

# Run single trial
results = single_regression_trial(experiment)
print(results.r2)
```

### Command line usage:

The user has the option to submit a single job or specify parameters using a configuration file:

Single experiment:

```bash
python regression.py --target PARP1 --fp_config exact-r2 --n_train 100 --save_results
```

Batch experiments from config file:

```bash
# View experiments (dry run)
python regression.py --config regression_experiments.yaml

# Submit to SLURM with 30 trials per experiment
python regression.py --config regression_experiments.yaml --submit --save_results
```

Configuration files must be in YAML format:

```yaml
name: "regression_experiments"
targets: ["PARP1", "F2"]
n_train: [10000]
optimize_hp: [true, false]

fingerprints:
  - exact-r2
  - compressed2048-r2

# SLURM job parameters
n_trials: 10
time: "8:00:00"
mem: "128G"
```

This generates experiments for each combination of parameters. SLURM jobs use the template in templates/regression_job.sh and save results to `results/regression/{target}/{fingerprint}/trial_*.pkl`. Each experiment is run `n_trials` different times with a seed corresponding to the SLRUM array ID, allowing for multiple random trials of a given experiment with different initializations:

```py
# Initialize trial ID as SLURM array ID
slurm_array_id = os.getenv("SLURM_ARRAY_TASK_ID")

if slurm_array_id is not None:
    experiment.trial_id = int(slurm_array_id)
    experiment.seed = int(slurm_array_id)
```
