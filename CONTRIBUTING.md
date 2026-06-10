# Contributing to Optrix

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/kevin-wangs/optrix.git
cd optrix
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code Standards

- Type hints required for all public functions
- Docstrings in Google style
- 100 char line limit (ruff enforced)
- All new features require tests

## Pull Request Process

1. Fork and create a feature branch from `main`
2. Add tests for new functionality
3. Ensure `pytest` passes
4. Update docs if API changed
5. Submit PR with clear description

## Architecture

Read `ARCHITECTURE.md` before modifying core modules.
