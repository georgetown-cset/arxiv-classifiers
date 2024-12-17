"""
Download weak negative examples from OpenAlex for use in training.
"""

import pandas as pd


def main():
    # We have ~103M OpenAlex works in the sampling frame table, so try taking ~0.1% of these
    sample = pd.read_gbq("""
        with sample as (
          select id
          from jd1881_sandbox.oa_negative_sampling_frame 
          tablesample system (0.1 percent)
        )
        select 
          id,
          title,
          abstract,
          type
        from openalex.works
        inner join sample using(id)
    """, project_id="gcp-cset-projects", use_bqstorage_api=True)
    print(sample["type"].value_counts())
    sample.to_parquet("../arxiv/weak_negatives.parquet")


if __name__ == '__main__':
    main()
