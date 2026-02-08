variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP region for bucket, Artifact Registry, Cloud Run"
}

variable "cloud_build_user" {
  type        = string
  description = "Email of user who runs terraform/gcloud (Cloud Build and Run invoker)"
}
