"""
Nitro NLP - Romanian eye-tracking TRT prediction (v2).

Targets Kaggle (P100, lightgbm + transformers preinstalled).
Local quick run without LM: USE_LM=0 USE_BERT=0 python3 solution.py

Pipeline (v2 additions marked NEW):
  1. Word features (length, syllables, position, etc.).
  2. Frequency features:
     - corpus log-frequency (own-word + prev/next).
     - NEW: wordfreq.zipf_frequency('ro') - curated Romanian unigram freq.
  3. Per-word target stats from train, leave-text-out for train rows.
  4. NEW: Romanian BERT MLM features per word
       - n_bert_subwords (paper: significant predictor on its own)
       - mlm_logp_first (paper's best single LM feature: -log p of first
         subword when ALL subwords of the word are masked).
       - mlm_logp_sum (sum across subwords, for completeness).
       Plus prev/next versions.
  5. NEW: Romanian BERT contextual embedding features
       - mean of last-4-layer hidden states, averaged across each word's
         subwords; PCA to top 24 components (fit on train, transform test).
  6. (kept) Causal-LM surprisal (RoGPT2 etc.) - different signal from MLM.
  7. Modeling - GroupKFold by `text`, seed-bagged LightGBM:
       - single-stage Tweedie regressor (kept).
       - NEW: two-stage = skip classifier x non-zero regressor.
  8. NEW: blend single-stage and two-stage with OOF-optimised convex weight.
  9. Linear post-hoc calibration on the blended OOF.
 10. Submission with subtaskID=1, datapointID, answer (sorted by datapointID),
     7 strict self-checks.
"""
from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_LM = os.environ.get("USE_LM", "1") == "1"
USE_BERT = os.environ.get("USE_BERT", "1") == "1"
USE_TWO_STAGE = os.environ.get("USE_TWO_STAGE", "1") == "1"

LM_CANDIDATES = os.environ.get(
    "LM_CANDIDATES",
    "readerbench/RoGPT2-large,readerbench/RoGPT2-medium,readerbench/RoGPT2-base,"
    "dumitrescustefan/gpt-neo-romanian-780m,dumitrescustefan/gpt-neo-romanian-125m"
).split(",")
LM_MAX_LEN = int(os.environ.get("LM_MAX_LEN", "1024"))

BERT_NAME = os.environ.get(
    "BERT_NAME", "dumitrescustefan/bert-base-romanian-cased-v1"
)
BERT_MAX_LEN = int(os.environ.get("BERT_MAX_LEN", "512"))
BERT_BATCH = int(os.environ.get("BERT_BATCH", "32"))
BERT_PCA_DIM = int(os.environ.get("BERT_PCA_DIM", "24"))

N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
SEED = 42

_KAGGLE_INPUT = Path("/kaggle/input")
if _KAGGLE_INPUT.exists():
    found = next(_KAGGLE_INPUT.rglob("train_data.csv"), None)
    if found is None:
        raise FileNotFoundError(
            f"train_data.csv not found anywhere under {_KAGGLE_INPUT}. "
            f"Subdirs: {[p.name for p in _KAGGLE_INPUT.iterdir()]}"
        )
    INPUT_DIR = found.parent
    if not (INPUT_DIR / "test_data.csv").exists():
        raise FileNotFoundError(
            f"Found train_data.csv at {found} but test_data.csv is not in the same folder. "
            f"Folder contents: {[p.name for p in INPUT_DIR.iterdir()]}"
        )
    OUTPUT_DIR = Path("/kaggle/working")
else:
    INPUT_DIR = Path(__file__).parent
    OUTPUT_DIR = Path(__file__).parent

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[cfg] INPUT_DIR={INPUT_DIR}  OUTPUT_DIR={OUTPUT_DIR}")
print(f"[cfg] USE_LM={USE_LM}  USE_BERT={USE_BERT}  USE_TWO_STAGE={USE_TWO_STAGE}  "
      f"N_SEEDS={N_SEEDS}")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def comp_metric(y_true: np.ndarray, preds: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if np.std(preds) < 1e-12:
        return 0.0
    r2 = max(0.0, r2_score(y_true, preds, force_finite=True))
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears):
        pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0


# ---------------------------------------------------------------------------
# Basic feature engineering
# ---------------------------------------------------------------------------

VOWELS = set("aeiouăâîAEIOUĂÂÎ")
PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
NUM_RE = re.compile(r"\d")
WORDID_RE = re.compile(r"^(?P<text>.+)_(?P<doc>\d+)_page_(?P<page>\d+)_(?P<idx>\d+)$")


def _count_syllables(token: str) -> int:
    s = token.lower()
    n, in_v = 0, False
    for ch in s:
        v = ch in VOWELS
        if v and not in_v:
            n += 1
        in_v = v
    return max(n, 1) if any(c.isalpha() for c in token) else 0


def _strip_punct(tok: str) -> str:
    return re.sub(r"[^\w]", "", tok, flags=re.UNICODE)


def _parse_word_id(word_id: str) -> Tuple[int, int, int]:
    m = WORDID_RE.match(word_id)
    if not m:
        return 0, 0, 0
    return int(m["doc"]), int(m["page"]), int(m["idx"])


def build_word_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word"] = df["word"].astype(str)
    stripped = df["word"].map(_strip_punct)
    df["word_len"] = df["word"].str.len()
    df["alpha_len"] = stripped.str.len()
    df["n_syllables"] = df["word"].map(_count_syllables)
    df["is_punct"] = df["word"].map(lambda w: bool(PUNCT_RE.match(w))).astype(int)
    df["is_url"] = df["word"].map(lambda w: bool(URL_RE.search(w))).astype(int)
    df["has_digit"] = df["word"].map(lambda w: bool(NUM_RE.search(w))).astype(int)
    df["is_upper_first"] = df["word"].map(lambda w: int(bool(w[:1].isupper())))
    df["is_all_upper"] = df["word"].map(lambda w: int(len(w) > 1 and w.isupper()))
    df["ends_with_punct"] = df["word"].map(
        lambda w: int(len(w) > 0 and not w[-1].isalnum())
    )
    parsed = df["word_id"].astype(str).map(_parse_word_id).tolist()
    df["doc_num"] = [p[0] for p in parsed]
    df["page_num"] = [p[1] for p in parsed]
    df["word_idx"] = [p[2] for p in parsed]
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["text", "participant_id", "doc_num", "page_num", "word_idx"]
    ).reset_index(drop=True)
    grp = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
    for col in ["word_len", "alpha_len", "n_syllables", "is_punct", "has_digit"]:
        df[f"prev_{col}"] = grp[col].shift(1).fillna(0)
        df[f"next_{col}"] = grp[col].shift(-1).fillna(0)
    df["pos_in_page"] = grp.cumcount()
    df["page_size"] = grp["word_idx"].transform("size")
    df["rel_pos_in_page"] = df["pos_in_page"] / df["page_size"].clip(lower=1)
    return df


def add_corpus_frequency(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_words = pd.concat([train["word"], test["word"]], axis=0).astype(str)
    norm = all_words.map(lambda w: _strip_punct(w).lower())
    counts = norm.value_counts()
    total = counts.sum()
    log_freq = np.log((counts + 1) / (total + len(counts)))
    fallback = float(log_freq.min())
    train_norm = train["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test_norm = test["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    train["log_freq"] = train_norm.map(log_freq).fillna(fallback)
    test["log_freq"] = test_norm.map(log_freq).fillna(fallback)
    for df in (train, test):
        grp = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
        df["prev_log_freq"] = grp["log_freq"].shift(1).fillna(fallback)
        df["next_log_freq"] = grp["log_freq"].shift(-1).fillna(fallback)
    return train, test


def add_wordfreq_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Use wordfreq.zipf_frequency for curated Romanian unigram frequency.
    Returns (train, test, ok). If wordfreq unavailable, ok=False and feature
    is not added."""
    try:
        from wordfreq import zipf_frequency
    except Exception as e:
        print(f"[wordfreq] not available ({e}); skipping zipf features")
        return train, test, False

    def _zipf(w: str) -> float:
        s = _strip_punct(w).lower()
        if not s:
            return 0.0
        return float(zipf_frequency(s, "ro", wordlist="best"))

    for df in (train, test):
        df["zipf_freq"] = df["word"].astype(str).map(_zipf)
        grp = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
        df["prev_zipf_freq"] = grp["zipf_freq"].shift(1).fillna(0.0)
        df["next_zipf_freq"] = grp["zipf_freq"].shift(-1).fillna(0.0)
    return train, test, True


def add_per_word_target_stats(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stat_cols = ["tok_mean", "tok_median", "tok_std", "tok_count", "tok_skip_rate"]

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("tok_lc")["answer"].agg(
            tok_mean="mean", tok_median="median", tok_std="std",
            tok_count="count",
            tok_skip_rate=lambda s: float((s == 0).mean()),
        )

    train = train.copy()
    test = test.copy()
    train["tok_lc"] = train["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test["tok_lc"] = test["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test = test.merge(_agg(train), left_on="tok_lc", right_index=True, how="left")
    for c in stat_cols:
        train[c] = np.nan
    for t in train["text"].unique():
        a = _agg(train.loc[train["text"] != t])
        rows = train["text"] == t
        merged = train.loc[rows, ["tok_lc"]].merge(
            a, left_on="tok_lc", right_index=True, how="left"
        )
        for c in stat_cols:
            train.loc[rows, c] = merged[c].values
    global_mean = float(train["answer"].mean())
    for c in ("tok_mean", "tok_median"):
        train[c] = train[c].fillna(global_mean)
        test[c] = test[c].fillna(global_mean)
    train["tok_std"] = train["tok_std"].fillna(0.0)
    test["tok_std"] = test["tok_std"].fillna(0.0)
    train["tok_skip_rate"] = train["tok_skip_rate"].fillna(0.3)
    test["tok_skip_rate"] = test["tok_skip_rate"].fillna(0.3)
    train["tok_count"] = train["tok_count"].fillna(0).astype(float)
    test["tok_count"] = test["tok_count"].fillna(0).astype(float)
    return train.drop(columns=["tok_lc"]), test.drop(columns=["tok_lc"])


# ---------------------------------------------------------------------------
# Causal-LM surprisal (kept from v1)
# ---------------------------------------------------------------------------

def _try_load_causal_lm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in LM_CANDIDATES:
        name = name.strip()
        if not name:
            continue
        try:
            print(f"[lm] trying {name}")
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token
            print(f"[lm] loaded {name} on {device}")
            return tok, model, name
        except Exception as e:
            print(f"[lm] {name} failed: {type(e).__name__}: {e}")
    return None, None, None


def compute_causal_surprisal(df: pd.DataFrame, tokenizer, model) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    out = np.zeros(len(df), dtype=np.float32)
    df = df.reset_index().rename(columns={"index": "_orig_idx"}).copy()
    df = df.sort_values(["text", "participant_id", "doc_num", "page_num", "word_idx"])

    @torch.no_grad()
    def score_token_ids(full_ids: List[int]) -> np.ndarray:
        L = len(full_ids)
        if L < 2:
            return np.zeros(L, dtype=np.float32)
        logp = np.zeros(L, dtype=np.float32)
        if L <= LM_MAX_LEN:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(ids).logits[0]
            lp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]
            pred = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp[1:] = (-pred).cpu().numpy()
            return logp
        stride = LM_MAX_LEN // 2
        first = True
        pos = 0
        while pos < L:
            end = min(pos + LM_MAX_LEN, L)
            chunk = full_ids[pos:end]
            ids = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = model(ids).logits[0]
            lp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]
            pred = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            fill_lo = pos + 1 if first else pos + stride
            fill_hi = end
            if fill_lo < fill_hi:
                k_lo = fill_lo - pos - 1
                k_hi = fill_hi - pos - 1
                logp[fill_lo:fill_hi] = (-pred[k_lo:k_hi]).cpu().numpy()
            first = False
            if end >= L:
                break
            pos += stride
        return logp

    use_offsets = tokenizer.is_fast
    bos = tokenizer.bos_token_id
    groups = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
    n_groups = len(groups)
    for gi, (_, block) in enumerate(groups):
        words = block["word"].astype(str).tolist()
        text_buf, char_spans, cursor = [], [], 0
        for i, w in enumerate(words):
            if i > 0:
                text_buf.append(" "); cursor += 1
            text_buf.append(w); char_spans.append((cursor, cursor + len(w)))
            cursor += len(w)
        text = "".join(text_buf)
        if use_offsets:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            sub_ids = enc["input_ids"]; offsets = enc["offset_mapping"]
        else:
            sub_ids = tokenizer.encode(text, add_special_tokens=False); offsets = None
        full_ids = ([bos] if bos is not None else []) + list(sub_ids)
        offset_shift = 1 if bos is not None else 0
        token_surps = score_token_ids(full_ids)
        sub_surps = token_surps[offset_shift:]
        if offsets is not None:
            word_surps = np.zeros(len(words), dtype=np.float32)
            wi = 0
            for k, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                while wi < len(words) and char_spans[wi][1] <= s:
                    wi += 1
                if wi < len(words) and char_spans[wi][0] <= s < char_spans[wi][1]:
                    word_surps[wi] += sub_surps[k]
        else:
            word_surps = np.zeros(len(words), dtype=np.float32)
            cursor_id = 0
            for wi, w in enumerate(words):
                piece = (" " if wi > 0 else "") + w
                ids_w = tokenizer.encode(piece, add_special_tokens=False)
                k_lo = cursor_id; k_hi = cursor_id + len(ids_w)
                if k_hi <= len(sub_surps):
                    word_surps[wi] = sub_surps[k_lo:k_hi].sum()
                cursor_id = k_hi
        out[block["_orig_idx"].to_numpy()] = word_surps
        if (gi + 1) % 100 == 0 or (gi + 1) == n_groups:
            print(f"[causal-lm] {gi+1}/{n_groups} groups scored")
    return out


# ---------------------------------------------------------------------------
# Romanian BERT MLM features + contextual embeddings
# ---------------------------------------------------------------------------

def _word_token_spans(tokenizer, words: List[str]):
    """Tokenize the full page once and map subword indices to word indices.
    Returns (sub_ids list, offsets list, per_word_subword_indices list)."""
    text_buf, char_spans, cursor = [], [], 0
    for i, w in enumerate(words):
        if i > 0:
            text_buf.append(" "); cursor += 1
        text_buf.append(w); char_spans.append((cursor, cursor + len(w)))
        cursor += len(w)
    text = "".join(text_buf)
    enc = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
        return_attention_mask=False
    )
    sub_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    per_word = [[] for _ in range(len(words))]
    wi = 0
    for k, (s, e) in enumerate(offsets):
        if s == e:
            continue
        while wi < len(words) and char_spans[wi][1] <= s:
            wi += 1
        if wi < len(words) and char_spans[wi][0] <= s < char_spans[wi][1]:
            per_word[wi].append(k)
    return sub_ids, per_word


def compute_bert_features(df: pd.DataFrame, tokenizer, model, mask_token_id: int):
    """Compute MLM-based features and contextual embedding features per row.

    Returns dict with arrays of length len(df):
      - n_bert_subwords
      - mlm_logp_first  (-log p of first subword when ALL subwords masked)
      - mlm_logp_sum    (sum -log p across all subwords)
      - bert_emb        (len(df), hidden_size) raw mean-of-subwords embedding
                        (caller PCA-reduces this)
    """
    import torch
    device = next(model.parameters()).device
    hidden = model.config.hidden_size

    n_subw_arr = np.zeros(len(df), dtype=np.float32)
    mlm_first_arr = np.zeros(len(df), dtype=np.float32)
    mlm_sum_arr = np.zeros(len(df), dtype=np.float32)
    emb_arr = np.zeros((len(df), hidden), dtype=np.float32)

    df = df.reset_index().rename(columns={"index": "_orig_idx"}).copy()
    df = df.sort_values(["text", "participant_id", "doc_num", "page_num", "word_idx"])
    groups = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
    n_groups = len(groups)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    @torch.no_grad()
    def run_chunk(chunk_sub_ids: List[int], chunk_per_word: List[List[int]]):
        """Run BERT in two passes and return:
           - sub_embs (numpy, len(chunk_sub_ids), H): mean of last-4 hidden states.
           - mlm_first[w], mlm_sum[w] for each word w in chunk_per_word.
        Extracts scalars inline so we never store the full logits tensor."""
        n_sub = len(chunk_sub_ids)
        n_w = len(chunk_per_word)
        mlm_first = [0.0] * n_w
        mlm_sum = [0.0] * n_w
        if n_sub == 0:
            return np.zeros((0, hidden), dtype=np.float32), mlm_first, mlm_sum

        # 1) embeddings on unmasked chunk
        ids = torch.tensor([[cls_id] + chunk_sub_ids + [sep_id]], dtype=torch.long, device=device)
        out_emb = model(ids, output_hidden_states=True)
        last4 = torch.stack(out_emb.hidden_states[-4:], dim=0).mean(dim=0)[0]     # (L, H)
        sub_embs = last4[1:-1].cpu().numpy()                                      # (n_sub, H)

        # 2) masked predictions, batched. Extract per-variant scalars inline.
        variants = []
        variant_word = []
        for local_w, sub_pos_list in enumerate(chunk_per_word):
            if not sub_pos_list:
                continue
            v = list(chunk_sub_ids)
            for p in sub_pos_list:
                v[p] = mask_token_id
            variants.append(v)
            variant_word.append(local_w)
        if not variants:
            return sub_embs, mlm_first, mlm_sum

        for batch_start in range(0, len(variants), BERT_BATCH):
            batch = variants[batch_start:batch_start + BERT_BATCH]
            wids = variant_word[batch_start:batch_start + BERT_BATCH]
            ids_batch = [[cls_id] + v + [sep_id] for v in batch]
            ids_t = torch.tensor(ids_batch, dtype=torch.long, device=device)
            logits = model(ids_t).logits                                    # (B, L, V)
            lp = torch.log_softmax(logits, dim=-1)
            for bi, lw in enumerate(wids):
                sub_pos_list = chunk_per_word[lw]
                # positions inside the model are offset by +1 for [CLS]
                first_pos = sub_pos_list[0] + 1
                first_orig = chunk_sub_ids[sub_pos_list[0]]
                mlm_first[lw] = float(-lp[bi, first_pos, first_orig].item())
                total = 0.0
                for p in sub_pos_list:
                    total += float(-lp[bi, p + 1, chunk_sub_ids[p]].item())
                mlm_sum[lw] = total
        return sub_embs, mlm_first, mlm_sum

    for gi, (_, block) in enumerate(groups):
        words = block["word"].astype(str).tolist()
        try:
            sub_ids, per_word_idx = _word_token_spans(tokenizer, words)
        except Exception as e:
            print(f"[bert] tokenization failed on group {gi}: {e}; skipping")
            continue
        n_words = len(words)

        # Process in non-overlapping chunks of (BERT_MAX_LEN - 2) subwords.
        capacity = BERT_MAX_LEN - 2
        # Build chunks of words s.t. their cumulative subword count <= capacity.
        chunks = []                             # list of (word_lo, word_hi, sub_ids_chunk)
        wi = 0
        while wi < n_words:
            sub_count = 0
            wj = wi
            while wj < n_words:
                w_subs = len(per_word_idx[wj])
                if sub_count + w_subs > capacity:
                    break
                sub_count += w_subs
                wj += 1
            if wj == wi:                        # single word longer than capacity: keep it (truncate)
                wj = wi + 1
            chunk_sub_ids = []
            for k in range(wi, wj):
                chunk_sub_ids.extend(sub_ids[idx] for idx in per_word_idx[k])
            chunks.append((wi, wj, chunk_sub_ids))
            wi = wj

        for (wlo, whi, chunk_sub_ids) in chunks:
            if not chunk_sub_ids:
                continue
            chunk_per_word = []
            cursor = 0
            for k in range(wlo, whi):
                w_subs = len(per_word_idx[k])
                chunk_per_word.append(list(range(cursor, cursor + w_subs)))
                cursor += w_subs

            sub_embs, mlm_first, mlm_sum = run_chunk(chunk_sub_ids, chunk_per_word)

            for local_w, sub_pos_list in enumerate(chunk_per_word):
                global_w = wlo + local_w
                row_idx = block["_orig_idx"].iloc[global_w]
                if sub_pos_list:
                    emb_arr[row_idx] = sub_embs[sub_pos_list].mean(axis=0)
                    n_subw_arr[row_idx] = len(sub_pos_list)
                    mlm_first_arr[row_idx] = mlm_first[local_w]
                    mlm_sum_arr[row_idx] = mlm_sum[local_w]

        if (gi + 1) % 50 == 0 or (gi + 1) == n_groups:
            print(f"[bert] {gi+1}/{n_groups} groups scored")

    return {
        "n_bert_subwords": n_subw_arr,
        "mlm_logp_first": mlm_first_arr,
        "mlm_logp_sum": mlm_sum_arr,
        "bert_emb": emb_arr,
    }


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def lgb_params_reg(seed: int) -> dict:
    return dict(
        objective="tweedie", tweedie_variance_power=1.4, metric="rmse",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbosity=-1, seed=seed,
    )


def lgb_params_clf(seed: int) -> dict:
    return dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbosity=-1, seed=seed,
    )


def lgb_params_pos(seed: int) -> dict:
    return dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbosity=-1, seed=seed,
    )


def fit_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if np.std(y_pred) < 1e-9:
        return 1.0, 0.0
    a = np.cov(y_pred, y_true, ddof=0)[0, 1] / np.var(y_pred)
    b = float(y_true.mean() - a * y_pred.mean())
    return float(a), b


def optimize_blend(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Find w in [0,1] maximising comp_metric(y, w*p1 + (1-w)*p2)."""
    best_w, best_s = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 51):
        s = comp_metric(y_true, w * p1 + (1 - w) * p2)
        if s > best_s:
            best_s, best_w = s, float(w)
    return best_w


def main() -> None:
    train = pd.read_csv(INPUT_DIR / "train_data.csv")
    test = pd.read_csv(INPUT_DIR / "test_data.csv")
    print(f"[data] train={train.shape}  test={test.shape}")
    n_test_orig = len(test)
    test_ids_orig = set(test["datapointID"].astype(int).tolist())
    assert len(test_ids_orig) == n_test_orig

    train = build_word_features(train)
    test = build_word_features(test)
    train = add_context_features(train)
    test = add_context_features(test)
    train, test = add_corpus_frequency(train, test)
    train, test, has_zipf = add_wordfreq_features(train, test)
    train, test = add_per_word_target_stats(train, test)

    feature_cols = [
        "word_len", "alpha_len", "n_syllables",
        "is_punct", "is_url", "has_digit",
        "is_upper_first", "is_all_upper", "ends_with_punct",
        "doc_num", "page_num", "word_idx", "pos_in_page",
        "rel_pos_in_page", "page_size",
        "log_freq", "prev_log_freq", "next_log_freq",
        "tok_mean", "tok_median", "tok_std", "tok_count", "tok_skip_rate",
        "prev_word_len", "prev_alpha_len", "prev_n_syllables",
        "prev_is_punct", "prev_has_digit",
        "next_word_len", "next_alpha_len", "next_n_syllables",
        "next_is_punct", "next_has_digit",
    ]
    if has_zipf:
        feature_cols += ["zipf_freq", "prev_zipf_freq", "next_zipf_freq"]

    # Causal-LM surprisal
    if USE_LM:
        try:
            tokenizer, lm_model, lm_name = _try_load_causal_lm()
            if lm_model is not None:
                print(f"[causal-lm] computing surprisal for train ({lm_name})")
                train["surprisal"] = compute_causal_surprisal(train, tokenizer, lm_model)
                print(f"[causal-lm] computing surprisal for test ({lm_name})")
                test["surprisal"] = compute_causal_surprisal(test, tokenizer, lm_model)
                # spillover surprisal
                for df in (train, test):
                    grp = df.groupby(
                        ["text", "participant_id", "doc_num", "page_num"], sort=False
                    )
                    df["prev_surprisal"] = grp["surprisal"].shift(1).fillna(0.0)
                    df["next_surprisal"] = grp["surprisal"].shift(-1).fillna(0.0)
                feature_cols += ["surprisal", "prev_surprisal", "next_surprisal"]
                del lm_model, tokenizer
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                print("[causal-lm] no model loaded; skipping surprisal")
        except Exception as e:
            print(f"[causal-lm] failed: {e}; continuing without surprisal")
    else:
        print("[causal-lm] disabled (USE_LM=0)")

    # Romanian BERT MLM + embeddings.
    # Optimisation: MLM features depend only on (text, doc, page, word_idx) -
    # not on participant. Compute once on the unique word_ids, then merge back.
    if USE_BERT:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[bert] loading {BERT_NAME} on {device}")
            btok = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)
            bmodel = AutoModelForMaskedLM.from_pretrained(BERT_NAME).to(device).eval()
            mask_id = btok.mask_token_id

            # Build a deduplicated (per-word_id) mini-dataframe that still carries
            # the columns compute_bert_features requires (text, participant_id,
            # doc_num, page_num, word_idx, word). participant_id is set to a
            # constant placeholder so each unique (text, doc, page) becomes one group.
            cols_needed = ["word_id", "word", "text", "doc_num", "page_num", "word_idx"]
            uniq = (pd.concat([train[cols_needed], test[cols_needed]], axis=0)
                    .drop_duplicates(subset=["word_id"]).reset_index(drop=True))
            uniq["participant_id"] = 0
            print(f"[bert] dedup: full rows={len(train)+len(test)}  unique word_ids={len(uniq)}")
            print("[bert] computing features on unique word_ids")
            uniq_bert = compute_bert_features(uniq, btok, bmodel, mask_id)
            uniq_feat = pd.DataFrame({
                "word_id": uniq["word_id"].values,
                "n_bert_subwords": uniq_bert["n_bert_subwords"],
                "mlm_logp_first":  uniq_bert["mlm_logp_first"],
                "mlm_logp_sum":    uniq_bert["mlm_logp_sum"],
            })
            uniq_emb_arr = uniq_bert["bert_emb"]      # (len(uniq), hidden)
            id_to_pos = {wid: i for i, wid in enumerate(uniq["word_id"].values)}
            train = train.merge(uniq_feat, on="word_id", how="left")
            test = test.merge(uniq_feat, on="word_id", how="left")
            tr_emb = uniq_emb_arr[train["word_id"].map(id_to_pos).values]
            te_emb = uniq_emb_arr[test["word_id"].map(id_to_pos).values]
            # Spillover MLM features
            for df in (train, test):
                grp = df.groupby(
                    ["text", "participant_id", "doc_num", "page_num"], sort=False
                )
                df["prev_mlm_logp_first"] = grp["mlm_logp_first"].shift(1).fillna(0.0)
                df["next_mlm_logp_first"] = grp["mlm_logp_first"].shift(-1).fillna(0.0)
            feature_cols += [
                "n_bert_subwords", "mlm_logp_first", "mlm_logp_sum",
                "prev_mlm_logp_first", "next_mlm_logp_first",
            ]
            # PCA-reduced contextual embeddings (already broadcasted from unique).
            tr_norms = np.linalg.norm(tr_emb, axis=1)
            valid = tr_norms > 1e-6
            n_comp = min(BERT_PCA_DIM, tr_emb.shape[1], int(valid.sum()) - 1)
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=SEED)
                pca.fit(tr_emb[valid])
                tr_emb_p = pca.transform(tr_emb)
                te_emb_p = pca.transform(te_emb)
                emb_cols = [f"bert_pc_{i}" for i in range(n_comp)]
                for i, c in enumerate(emb_cols):
                    train[c] = tr_emb_p[:, i]
                    test[c] = te_emb_p[:, i]
                feature_cols += emb_cols
                print(f"[bert] PCA: explained_var_ratio sum={pca.explained_variance_ratio_.sum():.3f}")
            del bmodel, btok
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[bert] failed: {e}; continuing without BERT features")
    else:
        print("[bert] disabled (USE_BERT=0)")

    for c in feature_cols:
        assert c in train.columns and c in test.columns, f"missing feature: {c}"
    n_nan_tr = int(train[feature_cols].isna().sum().sum())
    n_nan_te = int(test[feature_cols].isna().sum().sum())
    if n_nan_tr or n_nan_te:
        print(f"[feat] WARNING NaNs train={n_nan_tr} test={n_nan_te}; filling with 0")
        train[feature_cols] = train[feature_cols].fillna(0.0)
        test[feature_cols] = test[feature_cols].fillna(0.0)
    print(f"[feat] using {len(feature_cols)} features")

    train = train.sort_values(
        ["text", "participant_id", "doc_num", "page_num", "word_idx"]
    ).reset_index(drop=True)

    X = train[feature_cols].values
    y = train["answer"].values.astype(float)
    groups = train["text"].values
    Xt = test[feature_cols].values

    n_splits = min(N_FOLDS, train["text"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))

    # ---------- Single-stage Tweedie ----------
    oof_a = np.zeros(len(train))
    test_a = np.zeros(len(test))
    fold_scores_a = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        seed_oof = np.zeros(len(va_idx)); seed_test = np.zeros(len(test))
        for si in range(N_SEEDS):
            seed = SEED + 1000 * si + fi
            dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
            dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
            booster = lgb.train(
                lgb_params_reg(seed), dtr, num_boost_round=4000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            seed_oof += booster.predict(X[va_idx], num_iteration=booster.best_iteration)
            seed_test += booster.predict(Xt, num_iteration=booster.best_iteration)
        seed_oof /= N_SEEDS; seed_test /= N_SEEDS
        oof_a[va_idx] = seed_oof
        test_a += seed_test / n_splits
        s = comp_metric(y[va_idx], seed_oof)
        fold_scores_a.append(s)
        print(f"[stage1 fold {fi}] held-out={sorted(set(groups[va_idx]))}  score={s:.3f}")
    print(f"[stage1] folds mean={np.mean(fold_scores_a):.3f} std={np.std(fold_scores_a):.3f}")
    print(f"[stage1] global OOF (raw)= {comp_metric(y, oof_a):.3f}")

    # ---------- Two-stage skip x non-zero regressor ----------
    if USE_TWO_STAGE:
        oof_b = np.zeros(len(train))
        test_b = np.zeros(len(test))
        fold_scores_b = []
        y_skip = (y == 0).astype(float)
        for fi, (tr_idx, va_idx) in enumerate(splits):
            sk_oof = np.zeros(len(va_idx)); sk_test = np.zeros(len(test))
            rg_oof = np.zeros(len(va_idx)); rg_test = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED + 1000 * si + fi
                # Skip classifier
                dtr_c = lgb.Dataset(X[tr_idx], y_skip[tr_idx])
                dva_c = lgb.Dataset(X[va_idx], y_skip[va_idx], reference=dtr_c)
                clf = lgb.train(
                    lgb_params_clf(seed), dtr_c, num_boost_round=2000,
                    valid_sets=[dva_c],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
                sk_oof += clf.predict(X[va_idx], num_iteration=clf.best_iteration)
                sk_test += clf.predict(Xt, num_iteration=clf.best_iteration)
                # Non-zero regressor
                pos_tr = tr_idx[y[tr_idx] > 0]
                # validate on positives within va_idx, fallback to full va_idx
                pos_va = va_idx[y[va_idx] > 0]
                if len(pos_va) < 10:
                    pos_va = va_idx
                dtr_r = lgb.Dataset(X[pos_tr], np.log1p(y[pos_tr]))
                dva_r = lgb.Dataset(X[pos_va], np.log1p(y[pos_va]), reference=dtr_r)
                rg = lgb.train(
                    lgb_params_pos(seed), dtr_r, num_boost_round=4000,
                    valid_sets=[dva_r],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
                rg_oof += np.expm1(rg.predict(X[va_idx], num_iteration=rg.best_iteration))
                rg_test += np.expm1(rg.predict(Xt, num_iteration=rg.best_iteration))
            sk_oof /= N_SEEDS; sk_test /= N_SEEDS
            rg_oof /= N_SEEDS; rg_test /= N_SEEDS
            sk_oof = np.clip(sk_oof, 0, 1); sk_test = np.clip(sk_test, 0, 1)
            rg_oof = np.clip(rg_oof, 0, None); rg_test = np.clip(rg_test, 0, None)
            oof_b[va_idx] = (1 - sk_oof) * rg_oof
            test_b += ((1 - sk_test) * rg_test) / n_splits
            s = comp_metric(y[va_idx], oof_b[va_idx])
            fold_scores_b.append(s)
            print(f"[stage2 fold {fi}] held-out={sorted(set(groups[va_idx]))}  score={s:.3f}")
        print(f"[stage2] folds mean={np.mean(fold_scores_b):.3f} std={np.std(fold_scores_b):.3f}")
        print(f"[stage2] global OOF (raw)= {comp_metric(y, oof_b):.3f}")
    else:
        oof_b = oof_a.copy()
        test_b = test_a.copy()

    # ---------- Blend ----------
    w = optimize_blend(y, oof_a, oof_b) if USE_TWO_STAGE else 1.0
    print(f"[blend] optimal w (single-stage weight) = {w:.3f}")
    oof_blend = w * oof_a + (1 - w) * oof_b
    test_blend = w * test_a + (1 - w) * test_b
    print(f"[blend] global OOF (raw)= {comp_metric(y, oof_blend):.3f}")

    a, b = fit_calibration(y, oof_blend)
    print(f"[calib] a={a:.4f} b={b:.4f}")
    oof_cal = np.clip(a * oof_blend + b, 0, None)
    test_cal = np.clip(a * test_blend + b, 0, None)
    print(f"[final] global OOF (calibrated)= {comp_metric(y, oof_cal):.3f}")

    print(f"[pred] test stats: min={test_cal.min():.1f} mean={test_cal.mean():.1f} "
          f"max={test_cal.max():.1f} std={test_cal.std():.1f} "
          f"#zeros={(test_cal < 1).sum()}")

    sub = pd.DataFrame({
        "subtaskID": np.ones(len(test), dtype=int),
        "datapointID": test["datapointID"].astype(int).values,
        "answer": np.round(test_cal).astype(int),
    }).sort_values("datapointID").reset_index(drop=True)

    assert list(sub.columns) == ["subtaskID", "datapointID", "answer"]
    assert len(sub) == n_test_orig
    assert set(sub["datapointID"].tolist()) == test_ids_orig
    assert sub["datapointID"].is_monotonic_increasing
    assert sub["subtaskID"].eq(1).all()
    assert sub["answer"].notna().all()
    assert (sub["answer"] >= 0).all()
    assert sub["answer"].nunique() > 1

    sub_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[out] wrote {sub_path}  rows={len(sub)}")
    print(sub.head(6).to_string(index=False))
    print("[out] all self-checks passed.")


if __name__ == "__main__":
    main()
