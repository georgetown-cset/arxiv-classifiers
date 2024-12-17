#!/usr/bin/env bash
# Download trained models from GCS to the models directory

set -euo pipefail
set -x

# This is a subset of the models trained; selected from
# https://wandb.ai/cset/huggingface?workspaceuser-jamesdunham
models=(
  ai-32
  ai-full-weak-negs
  cv-32
  cyber-32
  cyber-full-weak-negs
  nlp-32
  ro-32
)

for model in "${models[@]}"; do
  gsutil -m cp -rn gs://arxiv-classifier-v2/models/"$model" .
done
