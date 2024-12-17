#!/usr/bin/env bash
# Run this before building the template image

gcloud dataflow flex-template build gs://cset-dataflow-templates/templates/arxiv-classifier-v2/template.json \
  --image gcr.io/gcp-cset-projects/arxiv-classifier-v2/template:latest \
  --sdk-language PYTHON \
  --metadata-file "metadata.json"
