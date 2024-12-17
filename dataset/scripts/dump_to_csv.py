import pandas as pd
from datasets import load_dataset


def main():
    arxiv = load_dataset(path="../arxiv", name="AI")
    for split in ["train", "validation", "test"]:
        output = arxiv[split].to_pandas()
        source = pd.read_parquet(f"../arxiv/{split}.parquet")
        output['id'] = source['id']
        output['year'] = source['year']
        output.iloc[0]
        source.iloc[0]

if __name__ == '__main__':
    main()