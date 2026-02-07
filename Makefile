.PHONY: format lint typecheck check build train train-full test-inference

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

check: format lint typecheck

build:
	./scripts/build_train.sh

train:
	./scripts/run_training.sh

train-full: build train

test-inference:
	./scripts/test_inference.sh
