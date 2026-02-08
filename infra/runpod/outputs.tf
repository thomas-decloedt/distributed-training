output "single_node" {
  value       = var.single_node
  description = "Whether using 1 pod (2 GPUs) vs 2 pods (1 GPU each)"
}

output "pod_id" {
  value       = var.single_node ? runpod_pod.single[0].id : runpod_pod.master[0].id
  description = "Pod ID for RunPod proxy SSH (POD_ID@ssh.runpod.io)"
}

output "master_ip" {
  value       = var.single_node ? runpod_pod.single[0].public_ip : runpod_pod.master[0].public_ip
  description = "Master pod public IP for direct SSH (or single pod when single_node)"
}

output "worker_ip" {
  value       = var.single_node ? "" : runpod_pod.worker[0].public_ip
  description = "Worker pod public IP (empty when single_node)"
}

output "worker_pod_id" {
  value       = var.single_node ? "" : runpod_pod.worker[0].id
  description = "Worker pod ID for RunPod proxy SSH (empty when single_node)"
}

output "docker_image" {
  value       = var.docker_image
  description = "Training image used by pods"
}

output "gcp_project_id" {
  value       = var.gcp_project_id
  description = "GCP project for model storage"
}

output "gcs_bucket" {
  value       = var.gcs_bucket
  description = "GCS bucket for model artifacts"
}

output "gcp_credentials_path" {
  value       = var.gcp_sa_key_path != "" ? "/tmp/gcp-sa-key.json" : ""
  description = "Path to GCP credentials file in container (empty when not configured)"
}
