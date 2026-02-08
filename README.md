# Distributed ML Training

PyTorch DDP example: MNIST training with Vertex AI Experiments, Model Registry, and Cloud Run.

## Prerequisites

- Docker
- [uv](https://github.com/astral-sh/uv)

## Dev

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run pyright
```

## Local Training

```bash
# MNIST on CPU
docker compose up train

# LoRA on CPU
docker compose up train-lora

# LoRA on GPU (requires nvidia-docker)
docker compose --profile gpu up train-gpu
```

Or with uv from `packages/train`:

```bash
cd packages/train && cp pyproject.cpu.toml pyproject.toml && uv sync
uv run torchrun --nproc_per_node=4 -m mnist.train   # or -m lora.train
```

## Task x Device Matrix

Any task (mnist, lora) on any device (cpu, gpu):

| Task   | Device | Build                    | Run                          |
|--------|--------|--------------------------|------------------------------|
| MNIST  | CPU    | `Dockerfile.train.cpu`    | `TASK=mnist` (default)       |
| LoRA   | CPU    | `Dockerfile.train.cpu`    | `TASK=lora ./scripts/run_lora.sh` |
| MNIST  | GPU    | `Dockerfile.train.gpu`    | `TASK=mnist`                 |
| LoRA   | GPU    | `Dockerfile.train.gpu`    | `TASK=lora`                  |

Set `use_gpu = false` in terraform for CPU VMs; `use_gpu = true` for GPU VMs.

## Backends

Training supports multiple backends. Set `BACKEND=gcp` (default) or `BACKEND=runpod`.

| Backend | Build | Provision | Run |
|---------|-------|-----------|-----|
| GCP | Cloud Build → Artifact Registry | `cd infra && terraform apply` | `make train` |
| RunPod | `make build-runpod` or [GitHub Actions](.github/workflows/build-train.yaml) | `make apply-runpod` | `make train-runpod` |

See [docs/runpod.md](docs/runpod.md) for RunPod setup when GCP quota is unavailable.

## GCP (Vertex AI + Model Registry + Cloud Run)

1. **Provision infra** (VMs, GCS, Artifact Registry, Cloud Run):

```bash
./scripts/bootstrap_tfstate.sh YOUR_PROJECT_ID
cd infra
cp terraform.tfvars.example terraform.tfvars
cp backend.tfvars.example backend.tfvars
# Edit both: project_id in terraform.tfvars, bucket in backend.tfvars
terraform init -backend-config=backend.tfvars
terraform apply
```

2. **Distributed training on cloud** (2-node DDP on GCE VMs):

```bash
./scripts/run_gcp.sh
```

This builds the training image via Cloud Build, pushes to Artifact Registry, then SSHs to master and worker VMs to run `torchrun` with `--nnodes=2`. VMs use the GCE metadata service to authenticate to Artifact Registry (no gcloud on VMs). Metrics and models are logged to Vertex AI Experiments and uploaded to GCS + Model Registry.

3. **Deploy endpoint** (run before or after training; Cloud Run serves from GCS):

```bash
./scripts/deploy_endpoint.sh
cd infra && terraform apply
```

4. **Test inference**:

```bash
make test-inference
```

Or manually (requires `gcloud auth login`):

```bash
curl -X POST $(cd infra && terraform output -raw inference_url)/predict \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"image": [[...]]}'  # 784 floats for 28x28
```

## Cost

Use e2-small or e2-medium; `terraform destroy` when done.

## GPU Training

GPU is enabled by default. To use CPU instead, set `use_gpu = false` in `infra/terraform.tfvars`.

See [docs/gpu-training.md](docs/gpu-training.md) for the full GPU setup guide, cost notes, and troubleshooting.

## Docs

See [docs/](docs/README.md) for next steps, portfolio roadmap, and GPU training reference.
