SELECT
  -- New predictions
  ai_predictions.id AS orig_id,
  year,
  -- Predictions from each model
  cast(ai_predictions.prediction AS BOOL) AS ai,
  cast(cyber_predictions.prediction AS BOOL) AS cyber,
  cast(nlp_predictions.prediction AS BOOL) AS nlp,
  cast(cv_predictions.prediction AS BOOL) AS cv,
  cast(ro_predictions.prediction AS BOOL) AS robotics,
  -- All the following is redundant but included for backwards compatibility
  cast(ai_predictions.prediction AS BOOL) AS ai_filtered,
  cast(cyber_predictions.prediction AS BOOL) AS cyber_filtered,
  cast(nlp_predictions.prediction AS BOOL) AS nlp_filtered,
  cast(cv_predictions.prediction AS BOOL) AS cv_filtered,
  cast(ro_predictions.prediction AS BOOL) AS robotics_filtered
FROM staging_openalex_article_classification.ai_predictions
INNER JOIN staging_openalex_article_classification.cyber_predictions USING (id)
INNER JOIN staging_openalex_article_classification.nlp_predictions USING (id)
INNER JOIN staging_openalex_article_classification.cv_predictions USING (id)
INNER JOIN staging_openalex_article_classification.ro_predictions USING (id)
LEFT JOIN staging_openalex_article_classification.openalex_papers USING (id)

UNION DISTINCT

SELECT
  orig_id,
  merged_id_predictions.year,
  ai,
  cyber,
  nlp,
  cv,
  robotics,
  ai_filtered,
  cyber_filtered,
  nlp_filtered,
  cv_filtered,
  robotics_filtered
FROM staging_openalex_article_classification.merged_id_predictions
INNER JOIN staging_openalex_article_classification.eligible_inputs
  USING (orig_id)

UNION DISTINCT

SELECT
  -- Existing predictions
  inputs_with_predictions.orig_id,
  eligible_inputs.year,
  ai,
  cyber,
  nlp,
  cv,
  robotics,
  ai_filtered,
  cyber_filtered,
  nlp_filtered,
  cv_filtered,
  robotics_filtered
FROM staging_openalex_article_classification.inputs_with_predictions
-- Only include existing predictions if the paper is still eligible ...
INNER JOIN staging_openalex_article_classification.eligible_inputs
  ON inputs_with_predictions.orig_id = eligible_inputs.orig_id
    AND utilities.CLASSIFIER_PREPROCESS( -- noqa: L030
      coalesce(inputs_with_predictions.title, "")
    ) = utilities.CLASSIFIER_PREPROCESS(coalesce(eligible_inputs.title, "")) -- noqa: L030
    AND utilities.CLASSIFIER_PREPROCESS( -- noqa: L030
      coalesce(inputs_with_predictions.abstract, "")
    ) = utilities.CLASSIFIER_PREPROCESS(coalesce(eligible_inputs.abstract, "")) -- noqa: L030
-- ... and their text hasn't changed.
-- Eligible papers whose text has changed will appear again in inputs_without_predictions
-- And they'll pass through the pipeline again. We omit the old prediction here
WHERE inputs_with_predictions.orig_id NOT IN (
  SELECT id
  FROM staging_openalex_article_classification.inputs_without_predictions
) AND inputs_with_predictions.orig_id NOT IN (
  SELECT orig_id
  FROM staging_openalex_article_classification.merged_id_predictions
)
