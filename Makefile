.PHONY: setup data run report test lint format all clean

export PYTHONUNBUFFERED := 1

setup:  ## install the locked environment
	uv sync --locked

data:  ## download and verify Hillstrom (4 MB) and Criteo (311 MB)
	uv run uplift-targeting data

run:  ## full pipeline: readout, models, evaluation, Criteo, figures
	uv run uplift-targeting run

report:  ## build the static report page in site/
	uv run uplift-targeting report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

all: setup data run report

clean:  ## remove derived data (raw downloads are kept)
	rm -rf data/interim
