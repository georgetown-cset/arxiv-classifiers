-- Define the sampling frame for weakly negative examples
create or replace table jd1881_sandbox.oa_negative_sampling_frame as (
  with negative_concepts as (
    select
      id as concept_id,
    from openalex.concepts
    where level = 0
      and display_name in (
      'Art',
      'Biology',
      'Business',
      'Economics',
      'Chemistry',
      -- 'Engineering', -- arXiv has EE
      'Environmental science',
      'Geology',
      'Materials science',
      'Geography',
      'History',
      'Medicine',
      'Philosophy',
      'Political science',
      'Psychology',
      'Sociology'
    )
  )
  select distinct
    works.id,
  from openalex.works, unnest(concepts) as concept
  inner join negative_concepts on concept.id = concept_id
  where
    publication_year >= 2010
    and coalesce(is_paratext, 'false') = 'false'
    and type not in (
      'dataset',
      'grant',
      'posted-content',
      'standard',
      'reference-entry'
    )
)