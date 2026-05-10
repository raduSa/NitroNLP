from __future__ import annotations
import re
import pandas as pd

VOWELS = set('aeiouăâîAEIOUĂÂÎ')
PUNCT_RE = re.compile(r'^[^\w]+$', re.UNICODE)
URL_RE = re.compile(r'https?://|www\.', re.IGNORECASE)
NUM_RE = re.compile(r'\d')
WORDID_RE = re.compile(r'^(?P<text>.+)_(?P<doc>\d+)_page_(?P<page>\d+)_(?P<idx>\d+)$')


def strip_punct(token: str) -> str:
    return re.sub(r'[^\w]', '', token, flags=re.UNICODE)


def _count_syllables(token: str) -> int:
    s = token.lower()
    n, in_vowel = 0, False
    for ch in s:
        is_vowel = ch in VOWELS
        if is_vowel and not in_vowel:
            n += 1
        in_vowel = is_vowel
    return max(n, 1) if any(c.isalpha() for c in token) else 0


def parse_word_id(word_id: str) -> tuple[int, int, int]:
    m = WORDID_RE.match(word_id)
    if not m:
        return 0, 0, 0
    return int(m['doc']), int(m['page']), int(m['idx'])


def build_word_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['word'] = df['word'].astype(str)
    stripped = df['word'].map(strip_punct)

    df['word_len'] = df['word'].str.len()
    df['alpha_len'] = stripped.str.len()
    df['n_syllables'] = df['word'].map(_count_syllables)
    df['is_punct'] = df['word'].map(lambda w: bool(PUNCT_RE.match(w))).astype(int)
    df['is_url'] = df['word'].map(lambda w: bool(URL_RE.search(w))).astype(int)
    df['has_digit'] = df['word'].map(lambda w: bool(NUM_RE.search(w))).astype(int)
    df['is_upper_first'] = df['word'].map(lambda w: int(bool(w[:1].isupper())))
    df['is_all_upper'] = df['word'].map(lambda w: int(len(w) > 1 and w.isupper()))
    df['ends_with_punct'] = df['word'].map(lambda w: int(len(w) > 0 and not w[-1].isalnum()))

    parsed = df['word_id'].astype(str).map(parse_word_id).tolist()
    df['doc_num'] = [p[0] for p in parsed]
    df['page_num'] = [p[1] for p in parsed]
    df['word_idx'] = [p[2] for p in parsed]
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['text', 'participant_id', 'doc_num', 'page_num', 'word_idx']).reset_index(drop=True)
    grp = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
    for col in ['word_len', 'alpha_len', 'n_syllables', 'is_punct', 'has_digit']:
        df[f'prev_{col}'] = grp[col].shift(1).fillna(0)
        df[f'next_{col}'] = grp[col].shift(-1).fillna(0)
    df['pos_in_page'] = grp.cumcount()
    df['page_size'] = grp['word_idx'].transform('size')
    df['rel_pos_in_page'] = df['pos_in_page'] / df['page_size'].clip(lower=1)
    return df
