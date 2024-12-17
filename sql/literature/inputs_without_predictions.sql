SELECT
  eligible_inputs.merged_id AS id,
  eligible_inputs.year,
  eligible_inputs.text
FROM staging_article_classification.eligible_inputs
-- Anti-join against inputs with predictions whose text
-- hasn't changed
LEFT JOIN staging_article_classification.inputs_with_predictions
  ON
    inputs_with_predictions.merged_id = eligible_inputs.merged_id --noqa: L016
    AND utilities.CLASSIFIER_PREPROCESS(COALESCE( -- noqa: L030
      inputs_with_predictions.title_english, ''
    )) = utilities.CLASSIFIER_PREPROCESS(COALESCE(eligible_inputs.title_english, '')) -- noqa: L030
    AND utilities.CLASSIFIER_PREPROCESS(COALESCE( -- noqa: L030
      inputs_with_predictions.abstract_english, ''
    )) = utilities.CLASSIFIER_PREPROCESS(COALESCE(eligible_inputs.abstract_english, '')) -- noqa: L030
WHERE inputs_with_predictions.merged_id IS NULL
