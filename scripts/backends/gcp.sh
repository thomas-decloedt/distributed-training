#!/usr/bin/env bash
# GCP backend: uses gcloud compute ssh to run training on GCE VMs.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source root common.sh for GCP terraform outputs
source "$ROOT_DIR/scripts/common.sh"

# GCP uses internal IP for rendezvous (faster, same VPC)
MASTER_ADDR="${MASTER_INTERNAL}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/train:latest"
BUCKET="${PROJECT_ID}-distributed-training"

# Source shared backend logic (TRAIN_ENV, TORCHRUN_ARGS, wait_for_master_port)
source "$SCRIPT_DIR/common.sh"

GPU_FLAG=""
if [ "$USE_GPU" = "true" ]; then
  GPU_FLAG="--gpus all "
fi

run_master_train() {
  local cmd="sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null; sudo docker pull ${IMAGE} && sudo docker run --rm ${GPU_FLAG}--network=host -e GOOGLE_CLOUD_PROJECT=${PROJECT_ID} -e VERTEX_STAGING_BUCKET=${BUCKET} -e MASTER_ADDR=${MASTER_ADDR} -e MASTER_PORT=29500 ${IMAGE} ${TORCHRUN_ARGS} --node_rank=0"
  gcloud compute ssh "dt-master" --zone="${MASTER_ZONE}" --command="$cmd"
}

run_worker_train() {
  local cmd="sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null; sudo docker pull ${IMAGE} && sudo docker run --rm ${GPU_FLAG}--network=host -e GOOGLE_CLOUD_PROJECT=${PROJECT_ID} -e VERTEX_STAGING_BUCKET=${BUCKET} -e MASTER_ADDR=${MASTER_ADDR} -e MASTER_PORT=29500 ${IMAGE} ${TORCHRUN_ARGS} --node_rank=1"
  gcloud compute ssh "dt-worker" --zone="${WORKER_ZONE}" --command="$cmd"
}

# GCP master uses internal IP; check from worker (reachable in same VPC)
wait_for_master() {
  echo "Waiting for master to pull image and start listening on port 29500..."
  echo "  (initial 2min for pull + container startup)"
  sleep 120
  for i in $(seq 1 24); do
    echo -n "  Check $i/24... "
    if gcloud compute ssh "dt-worker" --zone="${WORKER_ZONE}" --command="python3 -c \"
import socket
s = socket.socket()
s.settimeout(5)
try:
    s.connect(('${MASTER_ADDR}', 29500))
    s.close()
except: exit(1)
\" 2>/dev/null"; then
      echo "Master ready."
      return 0
    fi
    [ $i -eq 24 ] && { echo "FAILED"; echo "Timeout: master not reachable"; return 1; }
    echo "retry in 10s"
    sleep 10
  done
  return 1
}
