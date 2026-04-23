clean_models:
	@rm *.pth

clean_runs:
	@rm -rf runs/*

clean_all: clean_models clean_runs

clean_logs:
	@rm *.log

dev_local:
	./dev_local.sh up

dev_local_down:
	./dev_local.sh down

dev_local_status:
	./dev_local.sh status

# ---------------------------------------------------------------------------
# Lint / pre-commit
# ---------------------------------------------------------------------------

# Auto-fix lint + format on the entire src/ + tests/ tree.
lint_fix:
	uvx ruff check --fix src/ tests/
	uvx ruff format src/ tests/

# Read-only lint check (matches CI exactly).
lint:
	uvx ruff check src/ tests/

# Install the pre-commit hooks so `git commit` auto-fixes lint + format.
# Run once after cloning the repo.
hooks:
	uvx pre-commit install --install-hooks
	@echo "✓ pre-commit hooks installed — lint + format auto-fix on every commit"

hooks_run:
	uvx pre-commit run --all-files
