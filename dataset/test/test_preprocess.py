from pathlib import Path


def test_sources_exist(source_path):
    # This is the directory where we expect download.sh to place source data
    assert source_path.exists()
    source_paths = list(source_path.glob("../source/*.parquet"))
    assert len(source_paths), "No source files"


def test_outputs_exist(arxiv_path):
    # These are the outputs from split.py
    for split in ["train", "valid", "test"]:
        assert (arxiv_path / f"{split}.parquet").exists()


def test_text_is_nonmissing(source):
    for field in ["title", "abstract"]:
        assert not (source[field].str.strip() == "").any()
        assert not source[field].isna().any()


def test_created_year(source):
    year = source['created'].apply(lambda x: x.year).astype(int)
    assert not year.isna().any()


def test_ids_unique(source):
    assert not source["id"].duplicated().any()


def test_category_is_nonmissing(source):
    assert not source["categories"].isna().any()
    assert not (source["categories"].apply(len) == 0).any()
