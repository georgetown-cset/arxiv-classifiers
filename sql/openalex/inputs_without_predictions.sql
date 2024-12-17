SELECT
  eligible_inputs.orig_id AS id,
  eligible_inputs.text,
  eligible_inputs.year
FROM staging_openalex_article_classification.eligible_inputs
-- Anti-join against inputs with predictions whose text
-- hasn't changed
LEFT JOIN staging_openalex_article_classification.inputs_with_predictions
  ON
    inputs_with_predictions.orig_id = eligible_inputs.orig_id --noqa: L016
    AND (utilities.CLASSIFIER_PREPROCESS(COALESCE( -- noqa: L030
      inputs_with_predictions.title, ''
        )) = utilities.CLASSIFIER_PREPROCESS(COALESCE(eligible_inputs.title, ''))) -- noqa: L030
    AND (utilities.CLASSIFIER_PREPROCESS(COALESCE( -- noqa: L030
      inputs_with_predictions.abstract, ''
        )) = utilities.CLASSIFIER_PREPROCESS(COALESCE(eligible_inputs.abstract, ''))) -- noqa: L030
LEFT JOIN staging_openalex_article_classification.merged_id_predictions
  ON
    merged_id_predictions.orig_id = eligible_inputs.orig_id
WHERE (inputs_with_predictions.orig_id IS NULL) AND (merged_id_predictions.orig_id IS NULL)
