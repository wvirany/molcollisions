## Development Setup

```bash
# Clone and install
git clone https://github.com/wvirany/molcollisions.git
cd molcollisions
pip install -e ".[dev]"

# Set up code quality tools
pre-commit install

# Optional: Run type checking
mypy molcollisions/
```

**Note**: Code formatting and linting with `black` and `ruff` is run automatically on commit via pre-commit hooks, but can be run via
```
pre-commit run --all-files
```

`mypy` must be run manually to check for type errors
