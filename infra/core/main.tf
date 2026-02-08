# Shared platform: GCS bucket, Artifact Registry, serve image, Cloud Run.
# Used by both GCP VM backend and RunPod backend (same Vertex AI, model registry, inference).

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

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-distributed-training"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
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

# Cloud Run needs to read model from bucket
resource "google_storage_bucket_iam_member" "cloudrun" {
  bucket     = google_storage_bucket.models.name
  role       = "roles/storage.objectViewer"
  member     = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
  depends_on = [google_project_service.apis]
}

# Cloud Build: permissions for default compute SA
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

# Build serve image (same for GCP and RunPod model registration)
resource "null_resource" "build_serve_image" {
  triggers = {
    dockerfile = filemd5("${path.root}/../Dockerfile.serve")
    packages   = filemd5("${path.root}/../packages/serve/pyproject.toml")
    region     = var.region
  }
  provisioner "local-exec" {
    command     = "gcloud builds submit --config=cloudbuild.yaml --substitutions=_REGION=${var.region},_REPO=distributed-training ."
    working_dir = "${path.root}/.."
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
        name  = "MODEL_CASE"
        value = "mnist"
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
