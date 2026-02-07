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

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
    "storage-api.googleapis.com",
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

resource "google_compute_firewall" "torchrun" {
  name       = "distributed-training-torchrun"
  network    = "default"
  depends_on = [google_project_service.apis]

  allow {
    protocol = "tcp"
    ports    = ["29500"]
  }

  source_ranges = ["10.128.0.0/20"]
  target_tags   = ["distributed-training"]
}

resource "google_compute_instance" "master" {
  name         = "dt-master"
  machine_type = var.machine_type
  zone         = var.zone
  depends_on   = [google_project_service.apis]

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

  metadata_startup_script = <<-EOF
    apt-get update && apt-get install -y docker.io curl apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    apt-get update && apt-get install -y google-cloud-sdk
    systemctl enable docker && systemctl start docker
    gcloud auth configure-docker ${var.region}-docker.pkg.dev --quiet
  EOF
}

resource "google_compute_instance" "worker" {
  name         = "dt-worker"
  machine_type = var.machine_type
  zone         = var.zone
  depends_on   = [google_project_service.apis]

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

  metadata_startup_script = <<-EOF
    apt-get update && apt-get install -y docker.io curl apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    apt-get update && apt-get install -y google-cloud-sdk
    systemctl enable docker && systemctl start docker
    gcloud auth configure-docker ${var.region}-docker.pkg.dev --quiet
  EOF
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-distributed-training"
  location                    = var.region
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "serve" {
  location      = var.region
  repository_id = "distributed-training"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

data "google_project" "project" {
  project_id = var.project_id
}

resource "google_storage_bucket_iam_member" "cloudrun" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_artifact_registry_repository_iam_member" "compute_pull" {
  project    = var.project_id
  location   = google_artifact_registry_repository.serve.location
  repository = google_artifact_registry_repository.serve.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "compute_upload" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "cloud_build_user" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.builder"
  member  = "user:${var.cloud_build_user}"
}

resource "google_project_iam_member" "cloud_build_compute_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "cloud_build_compute_artifactregistry" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "cloud_build_compute_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "null_resource" "build_serve_image" {
  triggers = {
    dockerfile = filemd5("${path.module}/../Dockerfile.serve")
    deps      = filemd5("${path.module}/../requirements-serve.txt")
  }
  provisioner "local-exec" {
    command     = "gcloud builds submit --config=${path.module}/../cloudbuild.yaml ${path.module}/.."
    working_dir = path.module
  }
  depends_on = [
    google_artifact_registry_repository.serve,
    google_project_iam_member.cloud_build_user,
    google_project_iam_member.cloud_build_compute_storage,
    google_project_iam_member.cloud_build_compute_artifactregistry,
    google_project_iam_member.cloud_build_compute_logging,
  ]
}

resource "google_cloud_run_v2_service" "serve" {
  name     = "distributed-training-serve"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.serve.repository_id}/serve:latest"
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        startup_cpu_boost = true
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 12
      }
      env {
        name  = "MODEL_PATH"
        value = "gs://${google_storage_bucket.models.name}/models/latest/model.pt"
      }
    }
  }

  depends_on = [null_resource.build_serve_image]
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  location = google_cloud_run_v2_service.serve.location
  name     = google_cloud_run_v2_service.serve.name
  role     = "roles/run.invoker"
  member   = "user:${var.cloud_build_user}"
}
