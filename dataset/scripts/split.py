"""
Read the source data and create 70/15/15 splits stratified on papers' first category code.

We write these splits to the dataset directory as `train.parquet`, `valid.parquet`, and `test.parquet`. They have two
text columns, `title` and `text`, where `text` is the concatenation of the `title` and the paper's abstract.
"""
from sklearn.model_selection import train_test_split

from dataset.util.preprocess import preprocess

SEED = 20230216


def main():
    df = preprocess("../source", columns=["id", "created", "updated", "title", "abstract", "categories"])
    assert set(df.columns) == {"id", "created", "updated", "year", "title", "abstract", "text", "categories"}
    # In arXiv bulk data, IDs aren't necessarily unique. A paper can reappear if it's updated. But unique IDs are a
    # reasonable assumption so to avoid downstream issues we confirm here that uniqueness has been imposed upstream
    # (e.g. by taking the first or last version of each paper).
    assert df['id'].unique().all()

    train, not_train = split_stratified(df, 0.70)
    train.to_parquet(f"../arxiv/train.parquet")
    print(f"Wrote {train.shape[0]:,.0f} examples to ../arxiv/train.parquet")

    valid, test = split_stratified(not_train, 0.50)
    del not_train

    valid.to_parquet(f"../arxiv/valid.parquet")
    print(f"Wrote {valid.shape[0]:,.0f} examples to ../arxiv/valid.parquet")

    test.to_parquet(f"../arxiv/test.parquet")
    print(f"Wrote {test.shape[0]:,.0f} examples to ../arxiv/test.parquet")

    assert df.shape[0] == train.shape[0] + valid.shape[0] + test.shape[0]


def split_stratified(df_, size):
    assert isinstance(df_.iloc[0]['categories'], list)
    strata = df_['categories'].apply(lambda x: x[0])
    return train_test_split(df_, train_size=size, stratify=strata, random_state=SEED)


if __name__ == '__main__':
    main()
