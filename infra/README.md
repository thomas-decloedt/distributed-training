# Infrastructure

Terraform layout (all applied from this root except RunPod):

- **core/** – Shared platform: GCS bucket, Artifact Registry, serve image build, Cloud Run. Always applied; used by both GCP VMs and RunPod.
- **gcp/** – GCP compute backend (optional): VMs, firewall, IAM for training. Enabled when `enable_gcp_backend = true` (default). Child module called from root; no separate state.
- **runpod/** – Separate stack; its own state and tfvars. RunPod provider and pods. Uses the same project and bucket as core (set `gcp_project_id` and `gcs_bucket` from core outputs).

## When to set `enable_gcp_backend`

- **GCP VMs for training**: keep `enable_gcp_backend = true` (default). Apply from `infra/`; you get core + VMs.
- **RunPod only**: set `enable_gcp_backend = false`. Apply from `infra/` first to create core (bucket, AR, serve image, Cloud Run). Then apply `infra/runpod/` with `gcp_project_id` and `gcs_bucket` set to `terraform output project_id` and `terraform output model_bucket`.

See `docs/runpod.md` for the full RunPod flow.
