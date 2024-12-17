import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.hooks.base_hook import BaseHook
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryInsertJobOperator,
)
from airflow.providers.google.cloud.operators.dataflow import (
    DataflowStartFlexTemplateOperator,
    DataflowTemplatedJobStartOperator,
)
from airflow.providers.google.cloud.operators.gcs import GCSDeleteObjectsOperator
from airflow.providers.google.cloud.transfers.bigquery_to_bigquery import (
    BigQueryToBigQueryOperator,
)
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import (
    BigQueryToGCSOperator,
)
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import (
    GCSToBigQueryOperator,
)
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from dataloader.airflow_utils.defaults import (
    DAGS_DIR,
    DATA_BUCKET,
    PROJECT_ID,
    get_default_args,
    get_post_success,
)
from dataloader.airflow_utils.slack import task_fail_slack_alert
from dataloader.scripts.populate_documentation import update_table_descriptions

default_args = get_default_args(pocs=["James"])


class BQCheckIDsMatch(BigQueryCheckOperator):
    def __init__(self, table_a, table_b, id_a, id_b=None, dataset=None, **kw):
        if id_b is None:
            id_b = id_a
        if dataset is not None:
            table_a = f"{dataset}.{table_a}"
            table_b = f"{dataset}.{table_b}"
        sql = f"""
            select
              -- No IDs in A are null; all IDs in B were found in A
              count(*) = count(a.{id_a})
              -- No IDs in B are null; all IDs in A were found in B
                and count(*) = count(b.{id_b})
              -- IDs are distinct and non-null
                and count(*) = count(distinct a.{id_a})
            from {table_a} as a
            full outer join {table_b} as b
              on a.{id_a} = b.{id_b}
        """
        super().__init__(
            task_id=f"check_ids_match_{table_a.replace('.', '_')}_vs_"
            f"{table_b.replace('.', '_')}",
            sql=sql,
            use_legacy_sql=False,
            **kw,
        )


def make_dagname(dag_tag: str) -> str:
    """
    Generate name of dag
    :params dag_tag: Tag to insert into name of dag
    """
    return f"arxiv_classifier_{dag_tag}"


def create_dag(dag_tag: str) -> DAG:
    with DAG(
        make_dagname(dag_tag),
        default_args=default_args,
        description=f"Apply v2 arXiv AI classifiers to {dag_tag}",
        schedule_interval=None,
        catchup=False,
    ) as dag:
        is_literature = dag_tag == "literature"
        dataset = "arxiv_classifier"
        data_dir = f"{dataset}/data/{dag_tag}"
        input_data_dir = f"{data_dir}/input"
        output_data_dir = f"{data_dir}/output"
        dataflow_data_dir = f"{data_dir}/dataflow"
        schema_dir = f"{DAGS_DIR}/schemas/{dataset}/{dag_tag}"
        production_dataset = (
            "article_classification"
            if is_literature
            else "openalex_article_classification"
        )
        staging_dataset = f"staging_{production_dataset}"

        def create_query_config(filename, destination_dataset, destination_table=None):
            return {
                "query": {
                    "query": "{% include '"
                    + f"sql/{dataset}/{dag_tag}/{filename}.sql"
                    + "' %}",
                    "useLegacySql": False,
                    "destinationTable": {
                        "projectId": PROJECT_ID,
                        "datasetId": destination_dataset,
                        "tableId": destination_table if destination_table else filename,
                    },
                    "allowLargeResults": True,
                    "createDisposition": "CREATE_IF_NEEDED",
                    "writeDisposition": "WRITE_TRUNCATE",
                }
            }

        clear_data_dir = GCSDeleteObjectsOperator(
            task_id="clear_data_dir",
            bucket_name=DATA_BUCKET,
            prefix=data_dir,
        )

        create_eligible_inputs_table = BigQueryInsertJobOperator(
            task_id="create_eligible_inputs_table",
            configuration=create_query_config("eligible_inputs", staging_dataset),
        )

        if is_literature:
            clear_data_dir >> create_eligible_inputs_table
        else:
            create_openalex_pre_clean_table = BigQueryInsertJobOperator(
                task_id="create_openalex_pre_clean_table",
                configuration=create_query_config(
                    "openalex_pre_clean", staging_dataset
                ),
            )

            export_raw_papers = BigQueryToGCSOperator(
                task_id="export_raw_papers",
                source_project_dataset_table=f"{staging_dataset}.openalex_pre_clean",
                destination_cloud_storage_uris=f"gs://{DATA_BUCKET}/{dataflow_data_dir}/input/data*.jsonl",
                export_format="NEWLINE_DELIMITED_JSON",
            )

            clean_papers = DataflowTemplatedJobStartOperator(
                task_id="clean_papers",
                template="gs://cset-dataflow-templates/templates/run_clean_text",
                location="us-east1",
                dataflow_default_options={
                    "project": PROJECT_ID,
                    "tempLocation": "gs://cset-dataflow-test/example-tmps/",
                },
                parameters={
                    "input_prefix": f"gs://{DATA_BUCKET}/{dataflow_data_dir}/input/data*",
                    "output_prefix": f"gs://{DATA_BUCKET}/{dataflow_data_dir}/output/data",
                    "fields_to_clean": "title;abstract",
                },
            )

            import_cleaned_papers = GCSToBigQueryOperator(
                task_id="import_cleaned_papers",
                bucket=DATA_BUCKET,
                source_objects=[f"{dataflow_data_dir}/output/data*"],
                schema_object=f"{dataset}/schemas/openalex/openalex_papers.json",
                destination_project_dataset_table=f"{staging_dataset}.openalex_papers",
                source_format="NEWLINE_DELIMITED_JSON",
                create_disposition="CREATE_IF_NEEDED",
                write_disposition="WRITE_TRUNCATE",
            )

            create_merged_id_predictions = BigQueryInsertJobOperator(
                task_id="create_merged_id_predictions_table",
                configuration=create_query_config(
                    "merged_id_predictions", staging_dataset
                ),
            )

            (
                clear_data_dir
                >> create_openalex_pre_clean_table
                >> export_raw_papers
                >> clean_papers
                >> import_cleaned_papers
                >> create_merged_id_predictions
                >> create_eligible_inputs_table
            )

        create_inputs_without_predictions_table = BigQueryInsertJobOperator(
            task_id="create_inputs_without_predictions_table",
            configuration=create_query_config(
                "inputs_without_predictions", staging_dataset
            ),
        )

        check_input_size = BigQueryCheckOperator(
            task_id="check_input_size",
            sql=f"select "
            f"  (select count(*) from {staging_dataset}.inputs_without_predictions) "
            f"< (select count(*) from {staging_dataset}.inputs_with_predictions)",
            use_legacy_sql=False,
        )
        create_inputs_without_predictions_table >> check_input_size

        extract_inputs = BigQueryToGCSOperator(
            task_id="extract_inputs_without_predictions",
            source_project_dataset_table=f"{staging_dataset}.inputs_without_predictions",
            destination_cloud_storage_uris=[
                f"gs://{DATA_BUCKET}/{input_data_dir}/inputs-*.jsonl"
            ],
            export_format="NEWLINE_DELIMITED_JSON",
            force_rerun=True,
        )
        (
            create_eligible_inputs_table
            >> create_inputs_without_predictions_table
            >> check_input_size
            >> extract_inputs
        )

        wait_for_load = DummyOperator(task_id="wait_for_predictions")
        for model in [
            "ai",
            "cv",
            "cyber",
            "nlp",
            "ro",
        ]:
            job_name = (
                f"arxiv-classifier-{model}-dag-{datetime.now().isoformat()}".lower()
                .replace(":", "-")
                .replace(".", "-")
            )
            run_beam_inference = DataflowStartFlexTemplateOperator(
                task_id=f"{model}_inference",
                body={
                    "launchParameter": {
                        # We're running a template container based on this spec file (see images/template)
                        "containerSpecGcsPath": "gs://cset-dataflow-templates/templates/arxiv-classifier-v2/template.json",
                        "jobName": job_name,
                        # See https://cloud.google.com/dataflow/docs/reference/rest/v1b3/projects.locations.flexTemplates/launch#FlexTemplateRuntimeEnvironment
                        "environment": {
                            "maxWorkers": 200,
                            "additionalExperiments": [
                                "enable_prime",
                            ],
                            "tempLocation": f"gs://cset-dataflow-test/arxiv-classifier-tmp/{model}/",
                            "stagingLocation": f"gs://cset-dataflow-test/arxiv-classifier-staging/{model}/",
                            # Each worker file is another container
                            "sdkContainerImage": f"gcr.io/gcp-cset-projects/arxiv-classifier-v2/{model}:latest",
                        },
                        "parameters": {
                            "input_path": f"gs://{DATA_BUCKET}/{input_data_dir}/inputs-*.jsonl",
                            "output_path": f"gs://{DATA_BUCKET}/{output_data_dir}/{model}/output-",
                            "model_path": "/model",
                        },
                    }
                },
                location="us-east1",
                dag=dag,
            )
            extract_inputs >> run_beam_inference

            load_predictions = GCSToBigQueryOperator(
                task_id=f"load_{model}_predictions",
                bucket=DATA_BUCKET,
                source_objects=[f"{output_data_dir}/{model}/output-*"],
                destination_project_dataset_table=f"{staging_dataset}.{model}_predictions",
                source_format="NEWLINE_DELIMITED_JSON",
                create_disposition="CREATE_IF_NEEDED",
                write_disposition="WRITE_TRUNCATE",
            )
            check_inputs_vs_outputs = BQCheckIDsMatch(
                table_a="inputs_without_predictions",
                table_b=f"{model}_predictions",
                id_a="id",
                dataset=staging_dataset,
            )
            (
                run_beam_inference
                >> load_predictions
                >> check_inputs_vs_outputs
                >> wait_for_load
            )

        create_predictions_table = BigQueryInsertJobOperator(
            task_id="create_predictions_table",
            configuration=create_query_config("predictions", staging_dataset),
        )
        wait_for_load >> create_predictions_table

        checks = [
            BQCheckIDsMatch(
                table_a="eligible_inputs",
                table_b="predictions",
                id_a="merged_id" if is_literature else "orig_id",
                id_b="merged_id" if is_literature else "orig_id",
                dataset=staging_dataset,
            )
        ]

        copy_to_production = BigQueryToBigQueryOperator(
            task_id="copy_predictions_to_production",
            source_project_dataset_tables=[f"{staging_dataset}.predictions"],
            destination_project_dataset_table=f"{production_dataset}.predictions",
            create_disposition="CREATE_IF_NEEDED",
            write_disposition="WRITE_TRUNCATE",
        )
        for check in checks:
            create_predictions_table >> check >> copy_to_production

        snapshot_table = f"{production_dataset}.predictions_" + datetime.now().strftime(
            "%Y%m%d"
        )
        # mk the snapshot predictions table
        create_snapshot_table = BigQueryToBigQueryOperator(
            task_id="create_predictions_snapshot",
            source_project_dataset_tables=[f"{staging_dataset}.predictions"],
            destination_project_dataset_table=snapshot_table,
            create_disposition="CREATE_IF_NEEDED",
            write_disposition="WRITE_TRUNCATE",
        )

        create_inputs_with_predictions_table = BigQueryInsertJobOperator(
            task_id="create_inputs_with_predictions_table",
            configuration=create_query_config(
                "inputs_with_predictions", staging_dataset
            ),
        )

        # populate column descriptions of both the latest and the snapshot table
        pop_predictions_descriptions = PythonOperator(
            task_id="populate_column_documentation",
            op_kwargs={
                "input_schema": f"{schema_dir}/predictions.json",
                "table_name": f"{production_dataset}.predictions",
            },
            python_callable=update_table_descriptions,
        )
        pop_snapshot_descriptions = PythonOperator(
            task_id="populate_snapshot_column_documentation",
            op_kwargs={
                "input_schema": f"{schema_dir}/predictions.json",
                "table_name": snapshot_table,
            },
            python_callable=update_table_descriptions,
        )

        success_alert = get_post_success(
            f"Article classification update succeeded for {dag_tag}!", dag
        )

        copy_to_production >> (
            pop_predictions_descriptions,
            create_snapshot_table,
            create_inputs_with_predictions_table,
        )
        (create_snapshot_table >> pop_snapshot_descriptions >> success_alert)

        if is_literature:
            trigger_openalex_classification = TriggerDagRunOperator(
                task_id=f'trigger_{make_dagname("openalex")}',
                trigger_dag_id=make_dagname("openalex"),
            )
            success_alert >> trigger_openalex_classification
        else:
            trigger_ai_safety = TriggerDagRunOperator(
                task_id="trigger_ai_safety_predictions_literature",
                trigger_dag_id="ai_safety_predictions_literature",
            )
            success_alert >> trigger_ai_safety


datasets = [
    "literature",
    "openalex",
]
for dataset in datasets:
    globals()[make_dagname(dataset)] = create_dag(dataset)
