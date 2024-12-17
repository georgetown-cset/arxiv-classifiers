-- This query runs at the end of the DAG so that on the next run
-- we can identify papers that (1) still fit inclusion criteria,
-- (2) have unchanged metadata, and (3) already have predictions
SELECT
  predictions.*,
  -- Include the input text so we can check whether it's changed
  -- in future runs
  eligible_inputs.title_english,
  eligible_inputs.abstract_english
FROM staging_article_classification.predictions
INNER JOIN staging_article_classification.eligible_inputs
  ON eligible_inputs.merged_id = predictions.merged_id
