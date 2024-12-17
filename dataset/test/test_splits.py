from datasets import load_dataset


def test_undersample(ai, arxiv_path):
    train = ai["train"].to_pandas()
    undersample = load_dataset(str(arxiv_path), name="AI Undersample 4-to-1", split="train").to_pandas()
    original_prop = train["label"].value_counts(normalize=True)
    undersample_prop = undersample["label"].value_counts(normalize=True)
    assert original_prop[1] < 0.2
    assert undersample_prop[1] == 0.2


def test_subsample(ai, arxiv_path):
    train = ai["train"].to_pandas()
    subsample = load_dataset(str(arxiv_path), name="AI 10-pct Subsample", split="train").to_pandas()
    original_n = train.shape[0]
    subsample_n = subsample.shape[0]
    assert abs(subsample_n - original_n / 10) < 5


def test_multilabel(arxiv_path):
    train = load_dataset(str(arxiv_path), name="Multilabel with 1-pct Subsample", split="train").to_pandas()
    pass
