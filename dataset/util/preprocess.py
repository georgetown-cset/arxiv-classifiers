"""
Functions for preprocessing the arXiv corpus.
"""
from typing import Optional

import pandas as pd

from . import normalize_unicode, normalize_whitespace, remove_accents, remove_inline_math, \
    remove_punctuation, replace_currency_symbols, replace_numbers
from .remove import remove_punctuation_keeping_periods


def preprocess(input_path, **kwargs) -> pd.DataFrame:
    """Read arXiv data from the disk, then join and preprocess the title and abstract text.

    :param input_path: Path to source data.
    :return: Dataframe including preprocessed text.
    """
    df = pd.read_parquet(input_path, **kwargs)
    assert df.shape[0]
    # combine title and abstract
    df['text'] = _join_text(df)
    for text_col in ['title', 'abstract', 'text']:
        # we might also want a preprocessed title available
        df[text_col] = df[text_col].apply(_clean_text)
    df['year'] = df['created'].apply(lambda x: x.year).astype(int)
    # we know from experience positive labels are too sparse earlier than 2010
    df = df[df['year'] >= 2010]
    df['categories'] = df['categories'].str.split(' ')
    df['categories'] = df['categories'].apply(lambda x: [cat.strip() for cat in x if cat.strip()])
    return df


def _join_text(df: pd.DataFrame) -> pd.Series:
    assert not (df['title'].str.strip() == '').any()
    assert not (df['abstract'].str.strip() == '').any()
    return df['title'] + '. ' + df['abstract']


def _clean_text(value: Optional[str], keep_periods=True) -> Optional[str]:
    if value is None or pd.isnull(value):
        return None
    value = remove_inline_math(value)
    value = normalize_unicode(value, form='NFKC')
    if keep_periods:
        # We keep periods in SciBERT training
        value = remove_punctuation_keeping_periods(value)
    else:
        value = remove_punctuation(value)
    value = replace_currency_symbols(value, '_CUR_')
    value = remove_accents(value)
    value = replace_numbers(value, '_NUMBER_')
    value = normalize_whitespace(value)
    return value
