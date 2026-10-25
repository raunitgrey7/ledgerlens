.PHONY: setup lint format test bench bench-gate serve clean docker-build

UV ?= uv
JOBS ?= 4

setup:
	$(UV) sync --all-extras
	$(UV) run ledgerlens atlas

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest -q

bench:
	$(UV) run python -m ledgerlens.bench --jobs $(JOBS)

bench-gate:
	$(UV) run python -m ledgerlens.bench --jobs $(JOBS) --gate

serve:
	$(UV) run ledgerlens serve --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ledgerlens:latest .

clean:
	rm -rf .pytest_cache .ruff_cache data/synthetic/sample
