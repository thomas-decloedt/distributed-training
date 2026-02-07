# GPU Training Reference

How GPU-based distributed training is typically done. Use this when adding GPU support to the project.

---

## 1. Managed Training (Vertex AI)

**Most common approach** — no VM/SSH management.

- Submit a custom training job with GPU type (T4, L4, A100) and instance count
- Vertex handles provisioning, networking, scaling
- Example: `gcloud ai custom-jobs create ... --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=4`

---

## 2. GCE VMs with GPUs (Current Project Pattern)

### Machine Types

- `n1-standard-4` + 1× `nvidia-tesla-t4` or `nvidia-l4`
- `a2-highgpu-1g` (A100)
- `g2-standard-8` (L4 GPUs)

### Images

- Use GPU-ready base image (e.g. `nvidia/cuda:12.1-runtime`) or
- Install NVIDIA drivers in startup script

### DDP Backend

- Use `nccl` instead of `gloo` for GPU (2–3× faster)
- Process layout: typically 1 process per GPU

### Terraform Example

```hcl
resource "google_compute_instance" "master" {
  machine_type = "n1-standard-4"
  # ...
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }
  scheduling {
    on_host_maintenance = "TERMINATE"
  }
}
```

---

## 3. Kubernetes (Kubeflow)

- GPU node pool with `nvidia.com/gpu` resource
- Kubeflow Training Operator (PyTorchJob CRD)
- Each pod requests GPUs via resource spec

---

## 4. CPU vs GPU Comparison

| Aspect | Current (CPU) | Typical GPU |
|--------|---------------|--------------|
| Backend | `gloo` | `nccl` |
| Base image | `python:3.11-slim` | `nvidia/cuda` or `pytorch/pytorch` |
| VM | `e2-medium` | `n1-standard-4` + T4 |
| Drivers | None | NVIDIA drivers + container runtime |
| Cost | ~$0.02/hr per VM | ~$0.35/hr per T4 VM |

---

## 5. Config Changes for GPU

In `src/config.py` / `src/distributed.py`:

- Backend: `nccl` when `torch.cuda.is_available()`
- Already present in `src/distributed.py`: `backend = "nccl" if torch.cuda.is_available() else "gloo"`

Docker training image:

- Use `nvidia/cuda` or `pytorch/pytorch` base
- Ensure `nvidia-container-toolkit` on host for `docker run --gpus all`
