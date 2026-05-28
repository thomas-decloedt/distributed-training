terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }

  backend "gcs" {
    prefix = "distributed-training"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# Core: bucket, Artifact Registry, serve image, Cloud Run. Shared by GCP and RunPod.
# ------------------------------------------------------------------------------
module "core" {
  source = "./core"

  project_id       = var.project_id
  region           = var.region
  cloud_build_user = var.cloud_build_user
}

# ------------------------------------------------------------------------------
# GCP compute backend (optional). Omit when using RunPod.
# ------------------------------------------------------------------------------
data "google_project" "project" {
  project_id = var.project_id
}

locals {
  master_zone = var.master_zone != "" ? var.master_zone : var.zone
  worker_zone = var.worker_zone != "" ? var.worker_zone : var.zone
}

module "gcp" {
  count  = var.enable_gcp_backend ? 1 : 0
  source = "./gcp"

  project_id             = var.project_id
  region                 = var.region
  zone                   = var.zone
  master_zone            = local.master_zone
  worker_zone            = local.worker_zone
  use_gpu                = var.use_gpu
  machine_type           = var.machine_type
  gpu_machine_type       = var.gpu_machine_type
  accelerator_type       = var.accelerator_type
  preemptible            = var.preemptible
  model_bucket           = module.core.model_bucket
  artifact_repository_id = module.core.artifact_repository_id
  project_number         = data.google_project.project.number

  depends_on = [module.core]
}
