#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

BACKEND="${BACKEND:-gcp}"
DEVICE="${DEVICE:-gpu}"

if [ "$DEVICE" = "true" ] || [ "$DEVICE" = "gpu" ]; then
  DOCKERFILE="Dockerfile.train.gpu"
elif [ "$DEVICE" = "false" ] || [ "$DEVICE" = "cpu" ]; then
  DOCKERFILE="Dockerfile.train.cpu"
else
  DOCKERFILE="Dockerfile.train.${DEVICE}"
fi

echo "Building training image (${DOCKERFILE}, backend=${BACKEND})..."

case "$BACKEND" in
  gcp)
    source "$SCRIPT_DIR/common.sh"
    gcloud builds submit --config=cloudbuild-train.yaml \
      --substitutions=_DOCKERFILE=${DOCKERFILE},_REGION=${REGION} .
    ;;
  runpod)
    DOCKERHUB_USER="${DOCKERHUB_USER:?Set DOCKERHUB_USER for RunPod build}"
    IMAGE="${DOCKERHUB_USER}/distributed-training-train:latest"
    docker build -f "$DOCKERFILE" -t "$IMAGE" .
    docker push "$IMAGE"
    echo "Pushed $IMAGE"
    ;;
  *)
    echo "Unknown backend: $BACKEND. Supported: gcp, runpod"
    exit 1
    ;;
esac
