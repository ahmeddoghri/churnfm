# Contributing

Thanks for your interest in improving this project!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Before opening a pull request

- Keep changes focused; one logical change per PR.
- Add or update tests for any behaviour you change — CI runs `pytest` on
  Python 3.9 through 3.13, plus the example and benchmark scripts.
- Keep the public API type-annotated; the package ships a `py.typed` marker,
  so type regressions are user-visible.
- Run `pytest -q` locally and make sure it passes.
- Keep the README accurate if you change public behaviour.

## Reporting bugs

Open an issue with a minimal reproduction, the expected vs. actual behaviour,
and your environment. For security issues, see [SECURITY.md](SECURITY.md)
instead of filing a public issue.
