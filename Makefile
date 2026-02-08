.PHONY: format lint typecheck check build train train-full train-local train-local-lora test-inference
.PHONY: build-runpod train-runpod plan-runpod apply-runpod destroy-runpod

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

# RunPod backend (set BACKEND=runpod or use -runpod targets)
build-runpod:
	BACKEND=runpod ./scripts/build_train.sh

train-runpod:
	BACKEND=runpod ./scripts/run_training.sh

plan-runpod:
	cd infra/runpod && terraform init && terraform plan

apply-runpod:
	cd infra/runpod && terraform init && terraform apply

destroy-runpod:
	cd infra/runpod && terraform destroy

# Local training (single node, no GCP). Use 2 procs for faster LoRA iteration.
train-local:
	docker compose up train

train-local-lora:
	docker compose run --rm train-lora --nproc_per_node=2 -m lora.train

test-inference:
	./scripts/test_inference.sh
