#!/usr/bin/env bash

set -x

AIRFLOW_BUCKET='gs://us-east1-production-cc2-202-b42a7a54-bucket'
DATASET='arxiv_classifier'
SQL_PATH=$AIRFLOW_BUCKET/dags/sql/$DATASET
SCRIPTS_PATH=$AIRFLOW_BUCKET/dags/scripts/$DATASET
SCHEMAS_PATH=$AIRFLOW_BUCKET/dags/schemas/$DATASET

gsutil cp arxiv_classifier_dag.py $AIRFLOW_BUCKET/dags/

gsutil -m rm -rf $SQL_PATH/*
gsutil -m cp -r sql/* $SQL_PATH/

gsutil -m rm -f $SCHEMAS_PATH/*
gsutil -m cp schemas/literature/*.json $SCHEMAS_PATH/literature/
gsutil -m cp schemas/openalex/*.json $SCHEMAS_PATH/openalex/
gsutil -m cp schemas/openalex/*.json gs://airflow-data-exchange/$DATASET/schemas/openalex/
# add to airflow-data-exchange as well

gsutil -m rm -f $SCRIPTS_PATH/*
