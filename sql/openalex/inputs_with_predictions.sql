-- This query runs at the end of the DAG so that on the next run
-- we can identify papers that (1) still fit inclusion criteria,
-- (2) have unchanged metadata, and (3) already have predictions
-- We'll take predictions both from the last run of the article
-- classification pipeline on `literature` and from the last run
-- of the article classification pipeline on OA
SELECT
  predictions.orig_id,
  predictions.ai,
  predictions.cyber,
  predictions.nlp,
  predictions.cv,
  predictions.robotics,
  predictions.ai_filtered,
  predictions.cyber_filtered,
  predictions.nlp_filtered,
  predictions.cv_filtered,
  predictions.robotics_filtered,
  -- Include the input text so we can check whether it's changed
  -- in future runs
  eligible_inputs.title,
  eligible_inputs.abstract,
  eligible_inputs.year
FROM staging_openalex_article_classification.predictions
INNER JOIN staging_openalex_article_classification.eligible_inputs
  ON eligible_inputs.orig_id = predictions.orig_id
