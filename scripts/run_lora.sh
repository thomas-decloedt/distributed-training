#!/usr/bin/env bash
# Run LoRA training. Requires USE_GPU=true (Dockerfile.lora image).
set -e
export TASK=lora
exec "$(dirname "$0")/run_training.sh"
