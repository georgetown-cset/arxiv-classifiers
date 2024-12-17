#!/usr/bin/env bash

set -euo pipefail
set -x

gcloud builds submit -t gcr.io/gcp-cset-projects/arxiv-classifier-v2/ai:latest .