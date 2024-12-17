from pathlib import Path

import pandas as pd
from datasets import load_dataset
from pytest import fixture


@fixture
def dataset_path():
    return Path(__file__).parent.parent


@fixture
def arxiv_path(dataset_path):
    return dataset_path / "arxiv"


@fixture
def source_path(dataset_path):
    return dataset_path / "source"


@fixture
def ai(arxiv_path):
    return load_dataset(str(arxiv_path), "AI")


@fixture
def source(source_path):
    return pd.read_parquet(str(source_path))
