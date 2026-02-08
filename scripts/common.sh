#!/usr/bin/env bash
# Shared config for build and run scripts. Source with: source "$(dirname "$0")/common.sh"
set -e
cd "$(dirname "$0")/.."

PROJECT_ID=$(cd infra && terraform output -raw project_id 2>/dev/null || gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
  echo "Set PROJECT_ID or run terraform apply first"
  exit 1
fi

REGION=$(cd infra && terraform output -raw region 2>/dev/null)
REGION="${REGION:-us-central1}"
REPO="distributed-training"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/train:latest"
BUCKET="${PROJECT_ID}-distributed-training"

MASTER_INTERNAL=$(cd infra && terraform output -raw master_internal_ip 2>/dev/null || true)
MASTER_EXTERNAL=$(cd infra && terraform output -raw master_ip 2>/dev/null || true)
WORKER_EXTERNAL=$(cd infra && terraform output -raw worker_ip 2>/dev/null || true)
ZONE=$(cd infra && terraform output -raw zone 2>/dev/null || echo "us-central1-a")
MASTER_ZONE=$(cd infra && terraform output -raw master_zone 2>/dev/null || echo "$ZONE")
WORKER_ZONE=$(cd infra && terraform output -raw worker_zone 2>/dev/null || echo "$ZONE")
USE_GPU=$(cd infra && terraform output -raw use_gpu 2>/dev/null || echo "true")

export PROJECT_ID REGION REPO IMAGE BUCKET MASTER_INTERNAL MASTER_EXTERNAL WORKER_EXTERNAL ZONE MASTER_ZONE WORKER_ZONE USE_GPU
