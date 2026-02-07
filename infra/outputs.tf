output "project_id" {
  value       = var.project_id
  description = "GCP project ID"
}

output "region" {
  value       = var.region
  description = "GCP region"
}

output "zone" {
  value       = var.zone
  description = "GCP zone for compute instances"
}

output "master_ip" {
  value       = google_compute_instance.master.network_interface[0].access_config[0].nat_ip
  description = "Master node external IP"
}

output "worker_ip" {
  value       = google_compute_instance.worker.network_interface[0].access_config[0].nat_ip
  description = "Worker node external IP"
}

output "master_internal_ip" {
  value       = google_compute_instance.master.network_interface[0].network_ip
  description = "Master node internal IP for torchrun rendezvous"
}

output "model_bucket" {
  value       = google_storage_bucket.models.name
  description = "GCS bucket for model artifacts"
}

output "inference_url" {
  value       = google_cloud_run_v2_service.serve.uri
  description = "Cloud Run inference endpoint URL"
}
