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

output "master_zone" {
  value       = var.master_zone
  description = "Zone where master VM runs"
}

output "worker_zone" {
  value       = var.worker_zone
  description = "Zone where worker VM runs"
}
