from __future__ import annotations
import numpy as np
import pandas as pd
from .config import SEED


def augment_participant_mixup(
    train: pd.DataFrame,
    multiplier: float = 1.0,
    add_average_reader: bool = True,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    interp_cols = [c for c in ('answer', 'participant_enc', 'doc_num', 'surprisal') if c in train.columns]
    synth_parts, avg_parts = [], []

    for _, grp in train.groupby(['text', 'page_num', 'word_idx'], sort=False):
        n = len(grp)
        if n < 2:
            continue
        vals = grp[interp_cols].to_numpy(dtype=float)
        n_pairs = round(n * multiplier)
        idx_a = rng.integers(0, n, n_pairs)
        idx_b = rng.integers(0, n, n_pairs)
        idx_b[idx_a == idx_b] = (idx_b[idx_a == idx_b] + 1) % n
        alpha = rng.uniform(0.0, 1.0, (n_pairs, 1))
        synth_vals = alpha * vals[idx_a] + (1.0 - alpha) * vals[idx_b]
        synth_rows = grp.iloc[idx_a].reset_index(drop=True).copy()
        for i, col in enumerate(interp_cols):
            synth_rows[col] = synth_vals[:, i]
        synth_parts.append(synth_rows)

        if add_average_reader:
            avg_row = grp.iloc[[0]].copy()
            for i, col in enumerate(interp_cols):
                avg_row[col] = float(vals[:, i].mean())
            avg_parts.append(avg_row)

    pieces = [train]
    if synth_parts: pieces.append(pd.concat(synth_parts, ignore_index=True))
    if avg_parts: pieces.append(pd.concat(avg_parts,   ignore_index=True))
    augmented = pd.concat(pieces, ignore_index=True)
    n_added = len(augmented) - len(train)
    avg_added = sum(len(p) for p in avg_parts)
    print(f'[aug] mixup: +{n_added:,} rows ({n_added - avg_added:,} mixup + {avg_added:,} avg-reader)')
    return augmented


def augment_genre_average_reader(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train['_genre'] = train['text'].str.extract(r'^([a-z]+)_', expand=False)
    train['_tok_lc'] = train['word'].astype(str).str.lower().str.strip()
    has_part_enc = 'participant_enc' in train.columns
    global_part_enc = float(train['participant_enc'].mean()) if has_part_enc else None
    avg_rows = []

    for (_, _), grp in train.groupby(['_genre', '_tok_lc'], sort=False):
        if len(grp) < 2:
            continue
        avg_row = grp.iloc[[0]].copy()
        avg_row['answer'] = float(grp['answer'].mean())
        if has_part_enc:
            avg_row['participant_enc'] = global_part_enc
        avg_rows.append(avg_row)

    augmented = pd.concat([train] + avg_rows, ignore_index=True).drop(columns=['_genre', '_tok_lc'])
    print(f'[aug] genre-avg-reader: +{len(augmented) - len(train):,} rows')
    return augmented
