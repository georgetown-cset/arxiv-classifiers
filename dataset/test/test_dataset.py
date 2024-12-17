from collections import Counter

import pytest
from datasets import DatasetDict, load_dataset


def test_paths_exist(dataset_path, arxiv_path, source_path):
    assert dataset_path.exists()
    assert arxiv_path.exists()
    assert source_path.exists()


def test_load_default(arxiv_path):
    with pytest.raises(ValueError, match="Please pick one among the available configs"):
        load_dataset(str(arxiv_path))


def test_load_by_directory(ai):
    assert isinstance(ai, DatasetDict)


def test_splits_exist(ai):
    for split in ['train', 'validation', 'test']:
        assert split in ai


def test_example_fields(ai):
    for split, dataset in ai.items():
        assert set(dataset.column_names) == {"label", "text"}
        for example in dataset:
            assert example["label"] in {0, 1}
            assert isinstance(example["text"], str)


def test_weak_negatives(arxiv_path):
    ds = load_dataset(str(arxiv_path), "AI 10-pct Subsample", split="train")
    ds_with_negs = load_dataset(str(arxiv_path), "AI 10-pct Subsample with 10-pct Weak Negatives", split="train")
    counts = Counter([eg["label"] for eg in ds])
    counts_negs = Counter([eg["label"] for eg in ds_with_negs])
    assert counts[1] == counts_negs[1]
    ds_n = counts[0] + counts[1]
    ds_with_negs_n = counts_negs[0] + counts_negs[1]
    assert abs(ds_with_negs_n - ds_n * 1.1) < 10
