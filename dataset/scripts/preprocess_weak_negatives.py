"""
Preprocess weak-negative examples from OpenAlex for consistency with arXiv examples.
"""
import pandas as pd

from dataset.util.preprocess import _clean_text

SEED = 20230315


def main():
    df = pd.read_parquet("weak_negatives.parquet", columns=["id", "title", "abstract"])
    df = df.drop_duplicates("id")
    df = df.sample(n=100_000, random_state=SEED)
    assert df.shape[0]
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["text"] = df["title"] + ". " + df["abstract"]
    df["text"] = df["text"].apply(_clean_text).str.strip()
    assert df["id"].unique().all()
    assert (df["text"].apply(len) > 0).all()
    df[["id", "text"]].to_parquet("../arxiv/weak_negatives.parquet")


if __name__ == '__main__':
    main()