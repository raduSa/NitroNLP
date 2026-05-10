from __future__ import annotations
import numpy as np
import pandas as pd
from .word_features import strip_punct


def add_per_word_target_stats(train: pd.DataFrame, test: pd.DataFrame):
    stat_cols = ['tok_mean', 'tok_median', 'tok_std', 'tok_count', 'tok_skip_rate']

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby('tok_lc')['answer'].agg(
            tok_mean='mean', tok_median='median', tok_std='std', tok_count='count',
            tok_skip_rate=lambda s: float((s == 0).mean()),
        )

    train = train.copy()
    test = test.copy()
    lc = lambda w: strip_punct(w).lower()
    train['tok_lc'] = train['word'].astype(str).map(lc)
    test['tok_lc'] = test['word'].astype(str).map(lc)

    test = test.merge(_agg(train), left_on='tok_lc', right_index=True, how='left')
    for col in stat_cols:
        train[col] = np.nan
    for text in train['text'].unique():
        agg = _agg(train.loc[train['text'] != text])
        rows = train['text'] == text
        merged = train.loc[rows, ['tok_lc']].merge(agg, left_on='tok_lc', right_index=True, how='left')
        for col in stat_cols:
            train.loc[rows, col] = merged[col].values

    global_mean = float(train['answer'].mean())
    for col in ('tok_mean', 'tok_median'):
        train[col] = train[col].fillna(global_mean)
        test[col] = test[col].fillna(global_mean)
    train['tok_std'] = train['tok_std'].fillna(0.0)
    test['tok_std'] = test['tok_std'].fillna(0.0)
    train['tok_skip_rate'] = train['tok_skip_rate'].fillna(0.3)
    test['tok_skip_rate'] = test['tok_skip_rate'].fillna(0.3)
    train['tok_count'] = train['tok_count'].fillna(0).astype(float)
    test['tok_count'] = test['tok_count'].fillna(0).astype(float)
    return train.drop(columns=['tok_lc']), test.drop(columns=['tok_lc'])


def add_genre_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    train['_genre'] = train['text'].str.extract(r'^([a-z]+)_', expand=False)
    test['_genre'] = test['text'].str.extract(r'^([a-z]+)_', expand=False)

    all_genres = sorted(set(train['_genre'].dropna()) | set(test['_genre'].dropna()))
    genre_to_id = {g: i for i, g in enumerate(all_genres)}
    train['genre_id'] = train['_genre'].map(genre_to_id).fillna(-1).astype(int)
    test['genre_id'] = test['_genre'].map(genre_to_id).fillna(-1).astype(int)

    global_mean = float(train['answer'].mean())
    genre_means = train.groupby('_genre')['answer'].mean()
    test['genre_enc'] = test['_genre'].map(genre_means).fillna(global_mean)
    train['genre_enc'] = np.nan
    for text in train['text'].unique():
        genre = train.loc[train['text'] == text, '_genre'].iloc[0]
        other_mean = train.loc[(train['_genre'] == genre) & (train['text'] != text), 'answer'].mean()
        train.loc[train['text'] == text, 'genre_enc'] = other_mean if pd.notna(other_mean) else global_mean
    train['genre_enc'] = train['genre_enc'].fillna(global_mean)
    return train.drop(columns=['_genre']), test.drop(columns=['_genre'])
