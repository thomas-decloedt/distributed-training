# Core (always)
output "project_id" {
  value       = module.core.project_id
  description = "GCP project ID"
}

output "region" {
  value       = module.core.region
  description = "GCP region"
}

output "model_bucket" {
  value       = module.core.model_bucket
  description = "GCS bucket for model artifacts"
}

output "inference_url" {
  value       = module.core.inference_url
  description = "Cloud Run inference endpoint URL"
}

# GCP backend (only when enable_gcp_backend = true)
output "use_gpu" {
  value       = var.use_gpu
  description = "Whether GPU is enabled for training"
}

output "zone" {
  value       = var.zone
  description = "GCP zone for compute instances"
}

output "master_zone" {
  value       = local.master_zone
  description = "Zone where master VM runs"
}

output "worker_zone" {
  value       = local.worker_zone
  description = "Zone where worker VM runs"
}

output "master_ip" {
  value       = try(google_compute_instance.master[0].network_interface[0].access_config[0].nat_ip, null)
  description = "Master node external IP (null when using RunPod)"
}

output "worker_ip" {
  value       = try(google_compute_instance.worker[0].network_interface[0].access_config[0].nat_ip, null)
  description = "Worker node external IP (null when using RunPod)"
}

output "master_internal_ip" {
  value       = try(google_compute_instance.master[0].network_interface[0].network_ip, null)
  description = "Master node internal IP for torchrun rendezvous (null when using RunPod)"
}
