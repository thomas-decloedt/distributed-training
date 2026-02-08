variable "runpod_api_key" {
  type        = string
  description = "RunPod API key (from Settings → API Keys)"
  sensitive   = true
}

variable "docker_image" {
  type        = string
  description = "Docker image for training (e.g. USER/distributed-training-train:latest from Docker Hub)"
}

variable "gcp_project_id" {
  type        = string
  description = "GCP project ID for model storage (GCS bucket)"
}

variable "gcs_bucket" {
  type        = string
  description = "GCS bucket for model artifacts (e.g. PROJECT-distributed-training)"
}

variable "gcp_region" {
  type        = string
  default     = ""
  description = "GCP region for Vertex AI / Artifact Registry (must match core, e.g. europe-west4). Leave empty for us-central1."
}

variable "gpu_type_ids" {
  type        = list(string)
  default     = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40", "NVIDIA L4"]
  description = "GPU types to try (in availability order). Must match RunPod API exactly (e.g. 'NVIDIA GeForce RTX 4090' not 'NVIDIA RTX 4090')."
}

variable "single_node" {
  type        = bool
  default     = true
  description = "true = 1 pod with 2 GPUs (cheapest, single-node DDP). false = 2 pods with 1 GPU each (multi-node DDP)"
}

variable "ssh_public_key" {
  type        = string
  default     = ""
  description = "Public key (contents of .pub file) for direct SSH when RunPod proxy fails. If set, pod starts SSH daemon with this key."
}

variable "gcp_sa_key_path" {
  type        = string
  default     = ""
  description = "Path to GCP service account JSON key. Enables Vertex AI + GCS upload. Create SA with roles/storage.objectAdmin + roles/aiplatform.admin"
}
