variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "Default zone for compute instances"
}

variable "master_zone" {
  type        = string
  default     = ""
  description = "Zone for master VM. If empty, uses var.zone. Use different zones to improve GPU capacity availability."
}

variable "worker_zone" {
  type        = string
  default     = ""
  description = "Zone for worker VM. If empty, uses var.zone."
}

variable "machine_type" {
  type    = string
  default = "e2-medium"
}

variable "cloud_build_user" {
  type        = string
  description = "Email of user who runs terraform/gcloud (needs Cloud Build permissions)"
}

variable "use_gpu" {
  type        = bool
  default     = true
  description = "Enable GPU for training (T4/L4)"
}

variable "accelerator_type" {
  type        = string
  default     = "nvidia-tesla-t4"
  description = "GPU type when use_gpu=true (e.g. nvidia-tesla-t4, nvidia-l4)"
}

variable "gpu_machine_type" {
  type        = string
  default     = "n1-standard-4"
  description = "Machine type when use_gpu=true. Try n2-standard-4 if n1 has no capacity."
}

variable "preemptible" {
  type        = bool
  default     = false
  description = "Use spot/preemptible VMs. Often better GPU availability, but can be interrupted."
}

variable "enable_gcp_backend" {
  type        = bool
  default     = true
  description = "Create GCP compute backend (VMs, firewall). Set false when using RunPod so only core (bucket, AR, Cloud Run) is created."
}

