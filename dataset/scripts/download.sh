#!/usr/bin/env bash
# Download arXiv paper metadata from GCS to disk

gsutil -m cp gs://arxiv-classifier-v2/source/\*.parquet ../source/
