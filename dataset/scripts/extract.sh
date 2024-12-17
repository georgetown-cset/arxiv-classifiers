#!/usr/bin/env bash
# Extract arXiv paper metadata from BQ to GCS
# Important: if re-running this, clear out prior extract results from GCS first

set -euo
bq extract --destination_format=PARQUET \
  gcp_cset_arxiv_metadata.arxiv_metadata_latest \
  gs://arxiv-classifier-v2/source/arxiv-\*.parquet
