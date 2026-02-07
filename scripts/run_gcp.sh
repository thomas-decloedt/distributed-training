#!/usr/bin/env bash
# Full pipeline: build image then run training. For faster iteration, use build_train.sh and run_training.sh separately.
set -e
cd "$(dirname "$0")"

./build_train.sh && ./run_training.sh
