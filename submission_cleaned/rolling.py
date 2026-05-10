from __future__ import annotations
import numpy as np
import pandas as pd
from .config import ROLLING_WINDOWS, ROLLING_SKIP_THRESHOLD


def compute_rolling_features(
    df: pd.DataFrame,
    pred: np.ndarray,
    windows: list = ROLLING_WINDOWS,
    skip_threshold: float = ROLLING_SKIP_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df['__pred'] = pred
    df['__pos'] = np.arange(len(df))

    df_sorted = df.sort_values(['participant_id', 'doc_num', 'page_num', 'word_idx'])
    df_sorted['session_word_idx'] = df_sorted.groupby('participant_id').cumcount()
    df_sorted['pass1_pred']       = df_sorted['__pred']
    new_cols = ['session_word_idx', 'pass1_pred']
    global_mean = float(pred.mean())

    for k in windows:
        grp = df_sorted.groupby('participant_id')['__pred']
        df_sorted[f'roll_mean_{k}'] = grp.transform(
            lambda s: s.shift(1).rolling(k, min_periods=1).mean()
        ).fillna(global_mean)
        df_sorted[f'roll_skip_{k}'] = grp.transform(
            lambda s: (s.shift(1) < skip_threshold).astype(float).rolling(k, min_periods=1).mean()
        ).fillna(0.3)
        new_cols += [f'roll_mean_{k}', f'roll_skip_{k}']

    df_sorted = df_sorted.sort_values('__pos').drop(columns=['__pred', '__pos'])
    return df_sorted, new_cols
