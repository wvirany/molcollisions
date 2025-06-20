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

**Note**: Code formatting and linting run automatically on commit via pre-commit hooks.
