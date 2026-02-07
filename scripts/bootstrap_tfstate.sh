#!/usr/bin/env bash
set -e
if [ -z "$1" ]; then
  echo "Usage: ./scripts/bootstrap_tfstate.sh PROJECT_ID"
  echo "Creates GCS bucket for Terraform state (run once per project)"
  exit 1
fi
PROJECT_ID=$1
BUCKET="${PROJECT_ID}-tfstate"
gsutil mb -p "$PROJECT_ID" -l us-central1 "gs://${BUCKET}" 2>/dev/null || true
echo "Bucket gs://${BUCKET} ready"
echo "Create infra/backend.tfvars with: bucket = \"${BUCKET}\""
echo "Then: cd infra && terraform init -backend-config=backend.tfvars"
