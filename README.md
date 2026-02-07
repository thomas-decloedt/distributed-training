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
docker compose up
```

Or with uv:

```bash
uv sync
uv run torchrun --nproc_per_node=4 src/train.py
```

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

## Docs

See [docs/](docs/README.md) for next steps, portfolio roadmap, and GPU training reference.
