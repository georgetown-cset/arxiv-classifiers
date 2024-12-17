SELECT DISTINCT
  openalex_papers.id AS orig_id,
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
  openalex_papers.title,
  openalex_papers.abstract,
  openalex_papers.year
FROM article_classification.predictions
INNER JOIN staging_article_classification.eligible_inputs
  ON eligible_inputs.merged_id = predictions.merged_id
LEFT JOIN
  literature.sources
  ON
    sources.merged_id = predictions.merged_id
LEFT JOIN
  staging_literature.all_metadata_with_cld2_lid
  ON
    sources.orig_id = id
INNER JOIN
  staging_openalex_article_classification.openalex_papers
  ON
    (sources.orig_id = openalex_papers.id)
    AND ((
      utilities.CLASSIFIER_PREPROCESS( -- noqa: L030
        COALESCE(openalex_papers.title, "")
      ) = utilities.CLASSIFIER_PREPROCESS(COALESCE(title_english, "")) -- noqa: L030
    ) OR (title_english IS NULL AND (title_cld2_lid_first_result_short_code != "en")))
    AND ((
      utilities.CLASSIFIER_PREPROCESS( -- noqa: L030
        COALESCE(openalex_papers.abstract, "")
      ) = utilities.CLASSIFIER_PREPROCESS(COALESCE(abstract_english, "")) -- noqa: L030
    ) OR (abstract_english IS NULL AND (abstract_cld2_lid_first_result_short_code != "en")))
WHERE sources.dataset = "openalex"
