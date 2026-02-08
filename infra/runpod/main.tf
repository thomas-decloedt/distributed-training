terraform {
  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "~> 1.0"
    }
  }
}

provider "runpod" {
  api_key = var.runpod_api_key
}

# GCP credentials: when gcp_sa_key_path is set, inject SA key for Vertex AI + GCS upload
# Write GCP env to /root/.gcp_env so SSH sessions can source it (RunPod often doesn't inject env with custom entrypoint)
locals {
  gcp_sa_key_b64  = var.gcp_sa_key_path != "" ? base64encode(file(var.gcp_sa_key_path)) : ""
  gcp_creds_setup = "[ -n \"$GCP_SA_KEY_B64\" ] && echo \"$GCP_SA_KEY_B64\" | base64 -d > /tmp/gcp-sa-key.json && chmod 600 /tmp/gcp-sa-key.json || true"
  ssh_key_b64     = base64encode(var.ssh_public_key)
  ssh_setup       = var.ssh_public_key != "" ? "apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null && mkdir -p /root/.ssh && echo \"${local.ssh_key_b64}\" | base64 -d >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && service ssh start && " : ""
  gcp_env_line3   = var.gcp_region != "" ? " 'export GOOGLE_CLOUD_LOCATION=\"${var.gcp_region}\"'" : ""
  gcp_env_file    = "printf '%s\\n' 'export GOOGLE_CLOUD_PROJECT=\"${var.gcp_project_id}\"' 'export VERTEX_STAGING_BUCKET=\"${var.gcs_bucket}\"'${local.gcp_env_line3} > /root/.gcp_env && "
  ssh_setup_entrypoint = (var.ssh_public_key != "" || var.gcp_sa_key_path != "") ? [
    "/bin/bash", "-c",
    "${local.gcp_env_file}${local.gcp_creds_setup} && ${local.ssh_setup}exec sleep infinity"
  ] : ["sleep", "infinity"]
}

locals {
  runpod_env = merge(
    {
      GOOGLE_CLOUD_PROJECT  = var.gcp_project_id
      VERTEX_STAGING_BUCKET = var.gcs_bucket
    },
    var.gcp_region != "" ? { GOOGLE_CLOUD_LOCATION = var.gcp_region } : {},
    local.gcp_sa_key_b64 != "" ? {
      GCP_SA_KEY_B64                 = local.gcp_sa_key_b64
      GOOGLE_APPLICATION_CREDENTIALS = "/tmp/gcp-sa-key.json"
    } : {}
  )
}

# Single-node: 1 pod with 2 GPUs (cheapest, good for learning)
resource "runpod_pod" "single" {
  count              = var.single_node ? 1 : 0
  name               = "dt-train"
  image_name         = var.docker_image
  gpu_type_ids       = var.gpu_type_ids
  gpu_type_priority  = "availability"
  gpu_count          = 2
  cloud_type         = "COMMUNITY"
  support_public_ip   = true
  interruptible       = true
  container_disk_in_gb = 20
  volume_in_gb       = 20

  ports = ["22/tcp", "29500/tcp"]
  docker_entrypoint = local.ssh_setup_entrypoint

  env = local.runpod_env
}

# Multi-node: 2 pods with 1 GPU each
resource "runpod_pod" "master" {
  count               = var.single_node ? 0 : 1
  name                = "dt-master"
  image_name          = var.docker_image
  gpu_type_ids        = var.gpu_type_ids
  gpu_type_priority   = "availability"
  gpu_count           = 1
  cloud_type          = "COMMUNITY"
  support_public_ip   = true
  interruptible      = true
  container_disk_in_gb = 20
  volume_in_gb        = 20

  ports             = ["22/tcp", "29500/tcp"]
  docker_entrypoint = local.ssh_setup_entrypoint

  env = local.runpod_env
}

resource "runpod_pod" "worker" {
  count               = var.single_node ? 0 : 1
  name                = "dt-worker"
  image_name          = var.docker_image
  gpu_type_ids        = var.gpu_type_ids
  gpu_type_priority   = "availability"
  gpu_count           = 1
  cloud_type          = "COMMUNITY"
  support_public_ip   = true
  interruptible       = true
  container_disk_in_gb = 20
  volume_in_gb        = 20

  ports             = ["22/tcp", "29500/tcp"]
  docker_entrypoint = local.ssh_setup_entrypoint

  env = merge(local.runpod_env, { MASTER_ADDR = runpod_pod.master[0].public_ip })

  depends_on = [runpod_pod.master]
}
