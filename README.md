## Development Setup

```bash
# Clone and install
git clone https://github.com/wvirany/molcollisions.git
cd molcollisions
pip install -e .

# Set up code quality tools
pre-commit install
```


### Dependencies

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

This project uses specific forks of existing packages:

* `tanimoto_gp` - Tanimoto kernel Gaussian processes (forked to provide caching optimizations)
* `kern_gp` - Kernel computations with Cholesky updates (forked to provide efficient Cholesky factor updates)

**Development tools:** `pytest`, `pre-commit`, `black`, `ruff`, `mypy`

**Code quality:**

Code formatting and linting run automatically on commit via pre-commit hooks, type checking with `mypy` is manual:

```bash
# Manual run on all files
pre-commit run --all-files

# Type checking (manual)
mypy molcollisions/
```
