# Contributing

Thanks for contributing to Answer Engineering.

## Development setup

Use Python 3.12.

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Required local validation

Before opening a pull request, run:

```bash
./scripts/check
```

This is the main local validation entry point and is intended to mirror CI.

It runs, in order:

1. `ruff format --check .`
2. `ruff check .`
3. `pyright`
4. `pylint --rcfile conventions/pylintrc src`
5. `pytest`
6. `python -m build`

If `./scripts/check` fails, fix the reported issues before submitting a change.

## CI

CI is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

At the time of writing, the workflow performs:

- full repository validation via `./scripts/check`
- package verification by building and installing the wheel in a clean environment

## Documentation expectations

Documentation should be:

- faithful to the implemented code under [`src/`](src/)
- explicit about what is current behavior versus future direction
- linked with working relative markdown links where practical
- parseable when showing rule-language examples

Useful entry points:

- [Documentation index](docs/README.md)
- [Current codebase reality](docs/current/codebase-reality.md)
- [Current architecture](docs/current/architecture.md)
- [Rule language reference](docs/rules/language-reference.md)

## Pull request expectations

A good pull request should:

- have a clear scope
- avoid unrelated refactors
- include tests or validation appropriate to the change
- leave the repository in a passing `./scripts/check` state

## Golden snapshots

If you intentionally change output that is protected by snapshot tests, see:

- [Golden snapshots](docs/dev/golden-snapshots.md)
