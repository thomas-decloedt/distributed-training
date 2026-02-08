#!/usr/bin/env bash
# Create Artifact Registry repo and build serve image (needed for Vertex AI model registration).
# Run once per project when using RunPod or any flow that registers models to Vertex.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
REPO_ID="distributed-training"

if [ -z "$PROJECT_ID" ]; then
  echo "Usage: $0 [PROJECT_ID] [REGION]"
  echo "  Or set default: gcloud config set project PROJECT_ID"
  exit 1
fi

echo "Enabling APIs..."
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com --project="$PROJECT_ID"

echo "Creating repository $REPO_ID in $REGION..."
gcloud artifacts repositories create "$REPO_ID" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  2>/dev/null || echo "Repository already exists."

echo "Building and pushing serve image..."
cd "$ROOT_DIR"
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions="_REGION=$REGION,_REPO=$REPO_ID" \
  --project="$PROJECT_ID" \
  .

echo "Done. Model registration will use $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_ID/serve:latest"
