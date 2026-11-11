# `--no-editable`: see the note under [tool.uv] in pyproject.toml.
UV := uv run --no-editable

.PHONY: test test-fast lint fmt typecheck bench baselines ppo-small

test:
	$(UV) pytest

test-fast:
	$(UV) pytest -m "not slow"

lint:
	$(UV) ruff check src tests

fmt:
	$(UV) ruff format src tests && $(UV) ruff check --fix src tests

typecheck:
	$(UV) mypy src

bench:
	$(UV) catan benchmark --games 200 --workers 8

baselines:
	$(UV) catan tournament experiments/v1_examples/baselines.yaml

ppo-small:
	$(UV) catan train experiments/v1_examples/ppo_small.yaml
