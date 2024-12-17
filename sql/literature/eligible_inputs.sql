-- Select papers that fit inclusion criteria for classification
SELECT
  merged_id,
  title_english,
  abstract_english,
  utilities.CLASSIFIER_PREPROCESS(CASE -- noqa: L030
    WHEN abstract_english IS NULL THEN title_english
    WHEN title_english IS NULL THEN abstract_english
    ELSE title_english || '. ' || abstract_english
    END) AS text,
  year
FROM literature.papers
WHERE
  (
    title_english IS NOT NULL
    OR abstract_english IS NOT NULL
  )
  AND year >= 2010
