SELECT
  id,
  title,
  abstract,
  EXTRACT(YEAR FROM publication_date) AS year
FROM
  openalex.works
