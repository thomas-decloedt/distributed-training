# GPU Training Guide

This project supports both CPU and GPU training. Use the `use_gpu` Terraform variable to switch between modes.

---

## Quick Start (GPU on GCP)

GPU is enabled by default. If needed, confirm `infra/terraform.tfvars`:

```hcl
use_gpu          = true
accelerator_type = "nvidia-tesla-t4"
```

Provision or update infrastructure:

```bash
cd infra
terraform apply
```

3. Build the GPU training image and run training:

```bash
./scripts/run_gcp.sh
```

The build script uses `Dockerfile.train.gpu` when `use_gpu` is true. The run script adds `--gpus all` to `docker run` when GPU is enabled.

---

## Configuration

### Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `use_gpu` | `true` | Enable GPU for training (T4/L4) |
| `accelerator_type` | `"nvidia-tesla-t4"` | GPU type when `use_gpu=true` |

### Supported GPU Types (us-central1)

- `nvidia-tesla-t4` — T4 (cost-effective, ~$0.35/hr per VM)
- `nvidia-l4` — L4 (newer, may have limited availability)
- `nvidia-tesla-v100` — V100 (older, check zone availability)

### Machine Type

When `use_gpu=true`, VMs use `n1-standard-4` (4 vCPUs, 15 GB RAM). The CPU `machine_type` variable is ignored in GPU mode.

### Switching to CPU

Set `use_gpu = false` in `infra/terraform.tfvars` and run `terraform apply` to provision CPU VMs (e2-medium).

---

## Step-by-Step Setup

1. **Edit `infra/terraform.tfvars`**:

```hcl
project_id        = "your-gcp-project-id"
region            = "us-central1"
zone              = "us-central1-a"
use_gpu           = true
accelerator_type  = "nvidia-tesla-t4"
cloud_build_user  = "your-email@gmail.com"
```

2. **Apply Terraform** (first time may take 10–15 min for NVIDIA driver install + reboot):

```bash
cd infra
terraform init -backend-config=backend.tfvars
terraform apply
```

3. **Build GPU image**:

```bash
./scripts/build_train.sh
```

4. **Run distributed training**:

```bash
./scripts/run_training.sh
```

---

## Local GPU Training

If you have a GPU and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed:

```bash
docker compose --profile gpu up train-gpu
```

Or with a single process:

```bash
docker build -f Dockerfile.train.gpu -t train-gpu .
docker run --gpus all -v $(pwd)/data:/app/packages/train/data -v $(pwd)/checkpoints:/app/packages/train/checkpoints train-gpu --nproc_per_node=1 -m lora.train
```

---

## Cost Notes

| Mode | VM | Approximate cost |
|------|-----|------------------|
| CPU | e2-medium | ~$0.02/hr per VM |
| GPU | n1-standard-4 + T4 | ~$0.35/hr per VM |

**Important:** Tear down GPU resources when idle to avoid charges:

```bash
cd infra
terraform destroy
```

---

## Troubleshooting

### Check GPU on VMs

SSH to a VM and verify:

```bash
gcloud compute ssh dt-master --zone=us-central1-a
nvidia-smi
```

### Driver version

NVIDIA driver 535+ is recommended for CUDA 12.1. The startup script installs `cuda-drivers` from the NVIDIA repo.

### Container toolkit

Ensure `nvidia-container-toolkit` is installed and Docker is configured:

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### First boot delay

GPU VMs reboot once after NVIDIA driver installation. Allow 10–15 minutes for the first `terraform apply` to complete. Subsequent runs are faster.

### Image build failures

If Cloud Build fails for the GPU image, ensure `packages/train/pyproject.gpu.toml` exists and that the PyTorch CUDA 12.1 index is accessible.

---

## Architecture Reference

### Backend Selection

- **CPU:** `gloo` (default when no CUDA)
- **GPU:** `nccl` (auto-selected when `torch.cuda.is_available()`)

The application code in `packages/common/distributed.py` and `packages/mnist/train.py` or `packages/lora/train.py` chooses the backend and device automatically.

### Process Layout

- **CPU:** 1 process per node, 2 nodes total (master + worker)
- **GPU:** 1 process per GPU per node; for MNIST, 1 GPU per node is sufficient
