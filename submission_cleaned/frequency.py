from __future__ import annotations
import numpy as np
import pandas as pd
from .word_features import strip_punct


def add_corpus_frequency(train: pd.DataFrame, test: pd.DataFrame):
    all_words  = pd.concat([train['word'], test['word']], axis=0).astype(str)
    normalized = all_words.map(lambda w: strip_punct(w).lower())
    counts = normalized.value_counts()
    total = counts.sum()
    log_freq = np.log((counts + 1) / (total + len(counts)))
    fallback = float(log_freq.min())

    for df in (train, test):
        norm = df['word'].astype(str).map(lambda w: strip_punct(w).lower())
        df['log_freq'] = norm.map(log_freq).fillna(fallback)
        grp = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
        df['prev_log_freq'] = grp['log_freq'].shift(1).fillna(fallback)
        df['next_log_freq'] = grp['log_freq'].shift(-1).fillna(fallback)
    return train, test


def add_wordfreq_features(train: pd.DataFrame, test: pd.DataFrame):
    try:
        from wordfreq import zipf_frequency
    except Exception as e:
        print(f'[wordfreq] unavailable: {e}')
        return train, test, False

    def _zipf(w: str) -> float:
        s = strip_punct(w).lower()
        return float(zipf_frequency(s, 'ro', wordlist='best')) if s else 0.0

    for df in (train, test):
        df['zipf_freq'] = df['word'].astype(str).map(_zipf)
        grp = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
        df['prev_zipf_freq'] = grp['zipf_freq'].shift(1).fillna(0.0)
        df['next_zipf_freq'] = grp['zipf_freq'].shift(-1).fillna(0.0)
    return train, test, True
