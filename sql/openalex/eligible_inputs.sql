-- Select papers that fit inclusion criteria for classification
SELECT DISTINCT
  id AS orig_id,
  openalex_papers.title,
  openalex_papers.abstract,
  utilities.CLASSIFIER_PREPROCESS(CASE -- noqa: L030
    WHEN
      (
        openalex_papers.abstract IS NULL
      ) OR (abstract_cld2_lid_first_result_short_code != "en") THEN openalex_papers.title
    WHEN
      (openalex_papers.title IS NULL) OR (title_cld2_lid_first_result_short_code != "en") THEN openalex_papers.abstract
    ELSE openalex_papers.title || '. ' || openalex_papers.abstract
    END) AS text,
  openalex_papers.year
FROM staging_openalex_article_classification.openalex_papers
LEFT JOIN
  staging_literature.all_metadata_with_cld2_lid
  USING (id)
WHERE
  (
    (openalex_papers.title IS NOT NULL AND title_cld2_lid_first_result_short_code = "en")
    OR (openalex_papers.abstract IS NOT NULL AND abstract_cld2_lid_first_result_short_code = "en")
  )
  AND openalex_papers.year >= 2010
