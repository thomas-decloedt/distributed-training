#!/usr/bin/env bash
# Shared training config and helpers. Sourced by backends after they set IMAGE, MASTER_ADDR, PROJECT_ID, BUCKET.
set -e

TASK="${TASK:-mnist}"
if [ "$TASK" = "lora" ]; then
  TRAIN_MODULE="lora.train"
else
  TRAIN_MODULE="mnist.train"
fi

RDZV_ID="distributed-training-$(date +%s)"
RDZV_OPTS="--rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:29500 --rdzv_id=${RDZV_ID} --rdzv-conf=join_timeout=1800"
TORCHRUN_ARGS="--nnodes=2 --nproc_per_node=1 ${RDZV_OPTS} -m ${TRAIN_MODULE}"

# Env vars passed to the training container/process
export TRAIN_ENV="GOOGLE_CLOUD_PROJECT=${PROJECT_ID} VERTEX_STAGING_BUCKET=${BUCKET} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=29500"

wait_for_master_port() {
  local host="$1"
  local port="${2:-29500}"
  local max_attempts="${3:-24}"
  local interval="${4:-10}"

  echo "Waiting for master to listen on ${host}:${port}..."
  for i in $(seq 1 "$max_attempts"); do
    echo -n "  Check $i/$max_attempts... "
    if python3 -c "
import socket
s = socket.socket()
s.settimeout(5)
try:
    s.connect(('${host}', ${port}))
    s.close()
    exit(0)
except Exception:
    exit(1)
" 2>/dev/null; then
      echo "Master ready."
      return 0
    fi
    [ "$i" -eq "$max_attempts" ] && { echo "FAILED"; return 1; }
    echo "retry in ${interval}s"
    sleep "$interval"
  done
  return 1
}
