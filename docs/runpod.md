# RunPod Backend

Run GPU training on RunPod instead of GCP when you need immediate access without quota approval.

**Same flow as GCP:** Build image once → push to registry → pods pull and run it. No copying code.

## Prerequisites

- RunPod account
- RunPod API key ([Settings → API Keys](https://www.runpod.io/console/user/settings))
- Docker Hub account (for pushing training image)
- SSH key added to RunPod ([Settings → SSH Public Keys](https://www.runpod.io/console/user/settings))
- GCP project (for model storage, Vertex AI, Artifact Registry)

## One-time setup

1. **Apply GCP core** (bucket, Artifact Registry, serve image, Cloud Run). Same platform as GCP backend, no VMs:
   ```bash
   cd infra
   terraform init -backend-config=backend.tfvars
   terraform apply -var="enable_gcp_backend=false" -var="project_id=YOUR_PROJECT_ID" -var="cloud_build_user=YOUR_EMAIL"
   ```
   (If you use remote state, keep `backend.tfvars` with your state bucket. Omit `-var=...` if those are in `terraform.tfvars`.)
   Then set RunPod tfvars from outputs: `terraform output -raw model_bucket` and `terraform output -raw project_id`.

2. **Create RunPod terraform config:**

```bash
cd infra/runpod
cp terraform.tfvars.example terraform.tfvars
```

3. **Edit `terraform.tfvars`:**

```
runpod_api_key  = "YOUR_RUNPOD_API_KEY"
docker_image    = "YOUR_DOCKERHUB_USER/distributed-training-train:latest"
gcp_project_id  = "YOUR_PROJECT_ID"   # same as infra core (terraform output project_id)
gcs_bucket      = "YOUR_BUCKET"      # same as infra core (terraform output model_bucket)
single_node     = true   # 1 pod × 2 GPUs (cheapest)
gcp_sa_key_path = "./gcp-sa-key.json"
```

3. **Create GCP service account** (for Vertex AI + model upload): IAM → Service Accounts → Create. Grant `roles/storage.objectAdmin` and `roles/aiplatform.admin`. Create JSON key, save as `infra/runpod/gcp-sa-key.json`. (Core already created the Artifact Registry and serve image in step 1.)

4. **Add SSH key to RunPod** (required for `make train-runpod`):
   - [RunPod Settings → SSH Public Keys](https://www.runpod.io/console/user/settings)
   - Paste the **full contents** of `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub` (not the fingerprint)
   - Single-node uses `POD_ID@ssh.runpod.io`; RunPod injects keys from your account

5. **For GitHub Actions builds** (optional): Repo → Settings → Secrets and variables → Actions. Add:
   - `DOCKERHUB_USERNAME` – your Docker Hub username
   - `DOCKERHUB_TOKEN` – Docker Hub access token (Account Settings → Security → Access Tokens)

## Usage

```bash
# 1. Build and push training image to Docker Hub
#    Option A: GitHub Actions (recommended) – add DOCKERHUB_USERNAME and DOCKERHUB_TOKEN to repo secrets,
#    then push to main or run the workflow manually. Builds in the cloud, no local Docker needed.
#    Option B: Local – DOCKERHUB_USER=your-username make build-runpod

# 2. Provision pod(s) with YOUR image (single_node: 1 pod × 2 GPUs; multi-node: 2 pods × 1 GPU)
make apply-runpod

# 3. Run training (SSH runs torchrun on the pod)
TASK=lora make train-runpod

# 4. Teardown when done
make destroy-runpod
```

Or with explicit `BACKEND`:

```bash
BACKEND=runpod make build
BACKEND=runpod make train
```

## SSH connection

Pods use your image and expose SSH. Single-node uses RunPod proxy: `ssh <pod_id>@ssh.runpod.io`.

**"Permission denied (publickey)"** → Add your public key to [RunPod Settings → SSH Public Keys](https://www.runpod.io/console/user/settings). Paste the full `cat ~/.ssh/id_ed25519.pub` output (not the SHA256 fingerprint). Ensure each key is on its own line.

- **Key mismatch**: The script uses `~/.ssh/id_ed25519` or `~/.ssh/id_rsa` by default. If your key is elsewhere, set `SSH_KEY_PATH`:
  ```bash
  SSH_KEY_PATH=~/.ssh/your_key make train-runpod
  ```
- **Test manually**: `ssh -i ~/.ssh/id_ed25519 POD_ID@ssh.runpod.io` (replace POD_ID from `terraform output pod_id`). If this works, the key is correct; if not, re-add the matching public key to RunPod.

**Workaround when proxy auth keeps failing** (e.g. account key injection not working): inject your key into the pod via Terraform and use direct SSH:

1. Add to `infra/runpod/terraform.tfvars`:
   ```
   ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... your@email.com"
   ```
   (full contents of `cat ~/.ssh/id_ed25519.pub`)

2. `make destroy-runpod && make apply-runpod` to recreate the pod with SSH set up.

3. In the [RunPod console](https://www.runpod.io/console/pods) → your pod → **Connect** tab → copy **"SSH over exposed TCP"** (e.g. `ssh root@X.X.X.X -p12345 -i ~/.ssh/id_ed25519`).

4. Run training with IP and port from the RunPod console (Connect → SSH over exposed TCP):
   ```bash
   RUNPOD_MASTER_IP=206.41.93.58 RUNPOD_SSH_PORT=50774 make train-runpod
   ```

## Cost

Spot/interruptible: ~$0.2–0.5/hr per GPU. Single-node (1 pod × 2 GPUs) is cheapest. Run `make destroy-runpod` when done.

## Same flow as GCP

Build and run use the same `BACKEND` interface. GCP remains the default; set `BACKEND=runpod` to use RunPod. Models still upload to GCS; experiments still use Vertex AI if configured.
