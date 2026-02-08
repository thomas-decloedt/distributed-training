#!/usr/bin/env bash
# RunPod backend: uses SSH to run training on RunPod GPU pods.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNPOD_INFRA="$ROOT_DIR/infra/runpod"

# Load RunPod terraform outputs (suppress terraform warnings to stderr)
# Override with RUNPOD_MASTER_IP and RUNPOD_SSH_PORT when using ssh_public_key workaround
SINGLE_NODE=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw single_node 2>/dev/null || true | tr -d '[:space:]')
POD_ID=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw pod_id 2>/dev/null || true)
WORKER_POD_ID=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw worker_pod_id 2>/dev/null || true)
MASTER_IP="${RUNPOD_MASTER_IP:-$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw master_ip 2>/dev/null || true)}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-}"
WORKER_IP=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw worker_ip 2>/dev/null || true)
PROJECT_ID=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw gcp_project_id 2>/dev/null || true)
BUCKET=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw gcs_bucket 2>/dev/null || true)

# RunPod SSH: proxy (POD_ID@ssh.runpod.io) or direct (root@IP -p PORT when ssh_public_key workaround)
# Override with SSH_KEY_PATH if your key is elsewhere
# Use RUNPOD_SSH_PORT when using direct SSH (get from RunPod console Connect tab)
runpod_ssh() {
  local target_id="$1"
  local cmd="$2"
  local ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
  local key_opt=""
  if [ -n "$SSH_KEY_PATH" ] && [ -f "$SSH_KEY_PATH" ]; then
    key_opt="-i $SSH_KEY_PATH"
  elif [ -f ~/.ssh/id_ed25519 ]; then
    key_opt="-i ~/.ssh/id_ed25519"
  elif [ -f ~/.ssh/id_rsa ]; then
    key_opt="-i ~/.ssh/id_rsa"
  fi
  local port_opt=""
  [ -n "$RUNPOD_SSH_PORT" ] && port_opt="-p $RUNPOD_SSH_PORT"
  # Direct SSH (root@IP -p PORT) when we have IP+port (ssh_public_key workaround, single-node only)
  if [ -n "$MASTER_IP" ] && [ -n "$RUNPOD_SSH_PORT" ] && [ "$SINGLE_NODE" = "true" ]; then
    echo "SSH (direct): root@${MASTER_IP} -p ${RUNPOD_SSH_PORT}" >&2
    runpod_ssh_direct() { ssh $ssh_opts $key_opt $port_opt "root@${MASTER_IP}" "$cmd"; }
    for attempt in 1 2 3 4 5; do
      if runpod_ssh_direct; then return 0; fi
      [ "$attempt" -lt 5 ] && echo "Retry $attempt/5 in 5s..." >&2 && sleep 5
    done
    return 1
  elif [ -n "$target_id" ]; then
    ssh $ssh_opts $key_opt "${target_id}@ssh.runpod.io" "$cmd"
  else
    ssh $ssh_opts $key_opt $port_opt "root@${MASTER_IP}" "$cmd"
  fi
}

# Sanitize: terraform may output warnings; treat as empty
case "$MASTER_IP" in *[!0-9.]*) MASTER_IP="" ;; esac
case "$WORKER_IP" in *[!0-9.]*) WORKER_IP="" ;; esac

if [ "$SINGLE_NODE" = "true" ]; then
  if [ -n "$RUNPOD_SSH_PORT" ]; then
    [ -z "$MASTER_IP" ] && { echo "Direct SSH needs RUNPOD_MASTER_IP (or terraform output master_ip). Example: RUNPOD_MASTER_IP=206.41.93.58 RUNPOD_SSH_PORT=50774 make train-runpod"; exit 1; }
  elif [ -z "$POD_ID" ]; then
    echo "RunPod not provisioned. Run: cd infra/runpod && terraform apply"
    exit 1
  fi
else
  if [ -z "$MASTER_IP" ]; then
    echo "RunPod not provisioned. Run: cd infra/runpod && terraform apply"
    exit 1
  fi
  if [ -z "$WORKER_POD_ID" ] && [ -z "$WORKER_IP" ]; then
    echo "RunPod multi-node needs worker. Run: cd infra/runpod && terraform apply"
    exit 1
  fi
fi

# RunPod: MASTER_ADDR for rendezvous (multi-node) or display (single-node)
MASTER_ADDR="${MASTER_IP:-${POD_ID}@ssh.runpod.io}"

# Source shared backend logic (sets TORCHRUN_ARGS; we override for single-node)
source "$SCRIPT_DIR/common.sh"

# GCP credentials: use SA key on pod when available; else DISABLE_VERTEX and upload from local
GCP_CREDS_PATH=$(cd "$RUNPOD_INFRA" && terraform output -no-color -raw gcp_credentials_path 2>/dev/null || true)
if [ -n "$GCP_CREDS_PATH" ]; then
  TRAIN_ENV="${TRAIN_ENV} GOOGLE_APPLICATION_CREDENTIALS=${GCP_CREDS_PATH}"
else
  TRAIN_ENV="${TRAIN_ENV} DISABLE_VERTEX=1"
fi

# Single-node: --nnodes=1 --nproc_per_node=2 (no rendezvous)
if [ "$SINGLE_NODE" = "true" ]; then
  TORCHRUN_ARGS="--nnodes=1 --nproc_per_node=2 -m ${TRAIN_MODULE}"
fi

WORK_DIR="/app/packages/train"
# Source GCP env file written by Terraform entrypoint (RunPod may not inject env with custom entrypoint)
GCP_ENV_PREFIX="source /root/.gcp_env 2>/dev/null && "

run_master_train() {
  local cmd="${GCP_ENV_PREFIX}cd ${WORK_DIR} && export ${TRAIN_ENV} && torchrun ${TORCHRUN_ARGS}"
  [ "$SINGLE_NODE" != "true" ] && cmd="${cmd} --node_rank=0"
  runpod_ssh "$POD_ID" "$cmd"
}

run_worker_train() {
  if [ "$SINGLE_NODE" = "true" ]; then
    true
  else
    runpod_ssh "$WORKER_POD_ID" "${GCP_ENV_PREFIX}cd ${WORK_DIR} && export ${TRAIN_ENV} && torchrun ${TORCHRUN_ARGS} --node_rank=1"
  fi
}

wait_for_master() {
  if [ "$SINGLE_NODE" = "true" ]; then
    return 0
  fi
  wait_for_master_port "${MASTER_IP}" 29500
}
