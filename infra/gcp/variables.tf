variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region (e.g. for docker auth in startup script)"
}

variable "zone" {
  type        = string
  description = "Default zone for compute instances"
}

variable "master_zone" {
  type        = string
  description = "Zone for master VM"
}

variable "worker_zone" {
  type        = string
  description = "Zone for worker VM"
}

variable "use_gpu" {
  type        = bool
  description = "Enable GPU for training"
}

variable "machine_type" {
  type        = string
  description = "Machine type when use_gpu=false"
}

variable "gpu_machine_type" {
  type        = string
  description = "Machine type when use_gpu=true"
}

variable "accelerator_type" {
  type        = string
  description = "GPU type when use_gpu=true"
}

variable "preemptible" {
  type        = bool
  description = "Use preemptible VMs"
}

variable "model_bucket" {
  type        = string
  description = "GCS bucket for model uploads (from core module)"
}

variable "artifact_repository_id" {
  type        = string
  description = "Artifact Registry repository ID (from core module)"
}

variable "project_number" {
  type        = string
  description = "GCP project number (for compute SA IAM)"
}
