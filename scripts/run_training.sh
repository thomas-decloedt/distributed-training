#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND="${BACKEND:-gcp}"
BACKEND_SCRIPT="$SCRIPT_DIR/backends/$BACKEND.sh"

if [ ! -f "$BACKEND_SCRIPT" ]; then
  echo "Unknown backend: $BACKEND. Supported: gcp, runpod"
  exit 1
fi

source "$BACKEND_SCRIPT"

if [ -z "$MASTER_ADDR" ]; then
  echo "Backend $BACKEND failed to set MASTER_ADDR. Run: cd infra && terraform apply"
  exit 1
fi

TASK="${TASK:-mnist}"
echo "Starting distributed training on 2 nodes (backend=$BACKEND, task=$TASK)..."
echo "Master: $MASTER_ADDR"

run_master_train &
MASTER_PID=$!

wait_for_master || { kill $MASTER_PID 2>/dev/null; exit 1; }

run_worker_train &
WORKER_PID=$!

wait $MASTER_PID $WORKER_PID
echo "Training complete. Model uploaded to gs://${BUCKET}/models/"
