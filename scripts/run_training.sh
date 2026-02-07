#!/usr/bin/env bash
set -e
source "$(dirname "$0")/common.sh"

if [ -z "$MASTER_INTERNAL" ] || [ -z "$MASTER_EXTERNAL" ]; then
  echo "Run: cd infra && terraform apply"
  exit 1
fi

echo "Starting distributed training on 2 nodes..."
echo "Master: $MASTER_EXTERNAL (internal: $MASTER_INTERNAL)"
echo "Worker: $WORKER_EXTERNAL"

RDZV_ID="distributed-training-$(date +%s)"
gcloud compute ssh "dt-master" --zone="${ZONE}" --command="sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null; sudo docker pull ${IMAGE} && sudo docker run --rm --network=host -e GOOGLE_CLOUD_PROJECT=${PROJECT_ID} -e VERTEX_STAGING_BUCKET=${BUCKET} -e MASTER_ADDR=${MASTER_INTERNAL} -e MASTER_PORT=29500 ${IMAGE} --nnodes=2 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_INTERNAL}:29500 --rdzv_id=${RDZV_ID} --node_rank=0 src/train.py" &
MASTER_PID=$!
echo "Waiting 90s for master to pull image and start listening on port 29500..."
sleep 90
gcloud compute ssh "dt-worker" --zone="${ZONE}" --command="sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null; sudo docker pull ${IMAGE} && sudo docker run --rm --network=host -e GOOGLE_CLOUD_PROJECT=${PROJECT_ID} -e VERTEX_STAGING_BUCKET=${BUCKET} -e MASTER_ADDR=${MASTER_INTERNAL} -e MASTER_PORT=29500 ${IMAGE} --nnodes=2 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_INTERNAL}:29500 --rdzv_id=${RDZV_ID} --node_rank=1 src/train.py" &
WORKER_PID=$!
wait $MASTER_PID $WORKER_PID
echo "Training complete. Model uploaded to gs://${BUCKET}/models/"
