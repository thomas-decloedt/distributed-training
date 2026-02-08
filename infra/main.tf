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
locals {
  use_gpu_str    = var.use_gpu ? "true" : "false"
  machine_type   = var.use_gpu ? var.gpu_machine_type : var.machine_type
  master_zone    = var.master_zone != "" ? var.master_zone : var.zone
  worker_zone    = var.worker_zone != "" ? var.worker_zone : var.zone
  startup_script = <<-EOF
    set -e
    USE_GPU="${local.use_gpu_str}"

    if [ "$${USE_GPU}" = "true" ] && [ ! -f /var/lib/nvidia-driver-installed ]; then
      apt-get update && apt-get install -y curl
      curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
      dpkg -i /tmp/cuda-keyring.deb
      apt-get update && apt-get install -y cuda-drivers
      touch /var/lib/nvidia-driver-installed
      reboot
      exit 0
    fi

    apt-get update && apt-get install -y docker.io curl apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    apt-get update && apt-get install -y google-cloud-sdk
    systemctl enable docker && systemctl start docker

    if [ "$${USE_GPU}" = "true" ]; then
      distribution=$$(. /etc/os-release;echo $$ID$$VERSION_ID)
      curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/$${distribution}/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update && apt-get install -y nvidia-container-toolkit
      nvidia-ctk runtime configure --runtime=docker
      systemctl restart docker
    fi

    gcloud auth configure-docker ${var.region}-docker.pkg.dev --quiet
  EOF
}

resource "google_compute_firewall" "torchrun" {
  count       = var.enable_gcp_backend ? 1 : 0
  name        = "distributed-training-torchrun"
  network     = "default"
  depends_on  = [module.core]

  allow {
    protocol = "tcp"
    ports    = ["29500"]
  }

  source_ranges = ["10.128.0.0/20"]
  target_tags   = ["distributed-training"]
}

resource "google_compute_instance" "master" {
  count        = var.enable_gcp_backend ? 1 : 0
  name         = "dt-master"
  machine_type = local.machine_type
  zone         = local.master_zone
  depends_on   = [module.core]

  tags = ["distributed-training"]

  service_account {
    scopes = ["cloud-platform"]
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  dynamic "guest_accelerator" {
    for_each = var.use_gpu ? [1] : []
    content {
      type  = var.accelerator_type
      count = 1
    }
  }

  scheduling {
    automatic_restart   = !var.preemptible
    on_host_maintenance = var.use_gpu || var.preemptible ? "TERMINATE" : "MIGRATE"
    preemptible         = var.preemptible
  }

  metadata_startup_script = local.startup_script
}

resource "google_compute_instance" "worker" {
  count        = var.enable_gcp_backend ? 1 : 0
  name         = "dt-worker"
  machine_type = local.machine_type
  zone         = local.worker_zone
  depends_on   = [module.core]

  tags = ["distributed-training"]

  service_account {
    scopes = ["cloud-platform"]
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  dynamic "guest_accelerator" {
    for_each = var.use_gpu ? [1] : []
    content {
      type  = var.accelerator_type
      count = 1
    }
  }

  scheduling {
    automatic_restart   = !var.preemptible
    on_host_maintenance = var.use_gpu || var.preemptible ? "TERMINATE" : "MIGRATE"
    preemptible         = var.preemptible
  }

  metadata_startup_script = local.startup_script
}

# GCP backend: compute SA can pull serve image and upload models
data "google_project" "project" {
  project_id = var.project_id
}

resource "google_artifact_registry_repository_iam_member" "compute_pull" {
  count      = var.enable_gcp_backend ? 1 : 0
  project    = var.project_id
  location   = module.core.region
  repository = module.core.artifact_repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "compute_upload" {
  count  = var.enable_gcp_backend ? 1 : 0
  bucket = module.core.model_bucket
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_aiplatform_user" {
  count   = var.enable_gcp_backend ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}
