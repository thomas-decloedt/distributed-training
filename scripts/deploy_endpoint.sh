#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "Building and pushing serve image..."
gcloud builds submit --config=cloudbuild.yaml .
echo "Run: cd infra && terraform apply"
echo "Or if infra already applied: terraform apply -target=google_cloud_run_v2_service.serve"
