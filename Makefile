.PHONY: all smoke test lint format install dvc-pull download parse features train train-ability train-ball train-policies simulate-validate backtest serve serve-frontend serve-docker clean

# ── Reproduce the full pipeline ──────────────────────────────────────────────
all:
	dvc repro

# ── Smoke: run the pipeline end-to-end on fixture data ───────────────────────
smoke:
	pytest tests/ -m smoke -v

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=ml --cov=backend --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-leakage:
	pytest tests/leakage/ -v

test-simulator:
	pytest tests/simulator/ -v

# ── Lint + format ─────────────────────────────────────────────────────────────
lint:
	ruff check ml/ backend/ tests/
	mypy ml/ backend/

format:
	ruff format ml/ backend/ tests/
	ruff check --fix ml/ backend/ tests/

# ── Install ───────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	pre-commit install

# ── Data ──────────────────────────────────────────────────────────────────────
dvc-pull:
	dvc pull

download:
	python ml/ingest/download.py

parse:
	python ml/ingest/parse.py

# ── Pipeline stages ──────────────────────────────────────────────────────────
# Each stage runs for every (format, gender) combination with real data.
# Test-format trains ball/policy models for both genders same as t20/odi —
# women's Test data (24 matches total) is enough for those. The backtest
# step is the exception: it needs a date-split validation set to calibrate
# a logistic fit, and women's Test only has 3 matches in that window — below
# the number of parameters being fit. `ml.evaluate.backtest` detects this
# itself (val_matches < 2x feature count) and reports `insufficient_data`
# rather than a miscalibrated score, so it's simply not in the `backtest`
# target below.
features:
	python ml/features/snapshots.py

train-ability:
	python -m ml.models.ability.train --format t20 --gender male
	python -m ml.models.ability.train --format odi --gender male
	python -m ml.models.ability.train --format test --gender male
	python -m ml.models.ability.train --format t20 --gender female
	python -m ml.models.ability.train --format odi --gender female
	python -m ml.models.ability.train --format test --gender female

train-ball:
	python -m ml.models.ball.train --format t20 --gender male
	python -m ml.models.ball.train --format odi --gender male
	python -m ml.models.ball.train --format test --gender male
	python -m ml.models.ball.train --format t20 --gender female
	python -m ml.models.ball.train --format odi --gender female
	python -m ml.models.ball.train --format test --gender female

train-policies:
	python -m ml.models.policies.train --format t20 --gender male
	python -m ml.models.policies.train --format odi --gender male
	python -m ml.models.policies.train --format test --gender male
	python -m ml.models.policies.train --format t20 --gender female
	python -m ml.models.policies.train --format odi --gender female
	python -m ml.models.policies.train --format test --gender female

train: train-ability train-ball train-policies

simulate-validate:
	python ml/simulate/validate.py

backtest:
	python -m ml.evaluate.backtest --format t20 --gender male
	python -m ml.evaluate.backtest --format odi --gender male
	python -m ml.evaluate.backtest --format test --gender male
	python -m ml.evaluate.backtest --format t20 --gender female
	python -m ml.evaluate.backtest --format odi --gender female
	python -m ml.evaluate.backtest --format test --gender female

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

serve-frontend:
	cd frontend && npx vite --port 3000

serve-docker:
	docker compose up

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
