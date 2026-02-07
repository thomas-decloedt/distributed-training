#!/usr/bin/env bash
set -e
source "$(dirname "$0")/common.sh"

echo "Building and pushing training image..."
gcloud builds submit --config=cloudbuild-train.yaml .
