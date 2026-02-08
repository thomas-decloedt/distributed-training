output "project_id" {
  value       = var.project_id
  description = "GCP project ID"
}

output "region" {
  value       = var.region
  description = "GCP region"
}

output "model_bucket" {
  value       = google_storage_bucket.models.name
  description = "GCS bucket for model artifacts"
}

output "artifact_repository_id" {
  value       = google_artifact_registry_repository.serve.repository_id
  description = "Artifact Registry repository ID (distributed-training)"
}

output "serve_image_uri" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.serve.repository_id}/serve:latest"
  description = "Full URI of the serve image for Vertex AI model registration"
}

output "inference_url" {
  value       = google_cloud_run_v2_service.serve.uri
  description = "Cloud Run inference endpoint URL"
}
