"""
Pipeline:
  1. Word features (length, syllables, position, train-corpus log-freq, etc.).
  2. Per-word target stats (train mean/median/skip, leave-text-out for train rows).
  3. Optional: causal-LM surprisal from a Romanian GPT-2 (one forward pass per page).
  4. Seed-bagged LightGBM regressor with GroupKFold by `text`
     (mirrors the unseen-text test split).
  5. Linear post-hoc calibration on the OOF predictions to maximise R^2.
  6. Submission with subtaskID=1, datapointID, answer (sorted by datapointID).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_LM = os.environ.get("USE_LM", "1") == "1"
# Try these causal LMs in order; first one that loads wins.
LM_CANDIDATES = os.environ.get(
    "LM_CANDIDATES",
    "readerbench/RoGPT2-large,readerbench/RoGPT2-medium,readerbench/RoGPT2-base,"
    "dumitrescustefan/gpt-neo-romanian-780m,dumitrescustefan/gpt-neo-romanian-125m"
).split(",")
LM_MAX_LEN = int(os.environ.get("LM_MAX_LEN", "1024"))
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
print(f"[cfg] INPUT_DIR={INPUT_DIR}  OUTPUT_DIR={OUTPUT_DIR}  "
      f"USE_LM={USE_LM}  N_SEEDS={N_SEEDS}")


# ---------------------------------------------------------------------------
# Metric (mirrors the competition's eval_metric)
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
# Feature engineering
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
    """Spillover features. Sorts in-place into reading order and resets index."""
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
    # Spillover frequency from prev word (frequency-spillover is a known eye-tracking effect).
    for df in (train, test):
        grp = df.groupby(["text", "participant_id", "doc_num", "page_num"], sort=False)
        df["prev_log_freq"] = grp["log_freq"].shift(1).fillna(fallback)
        df["next_log_freq"] = grp["log_freq"].shift(-1).fillna(fallback)
    return train, test


def add_per_word_target_stats(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test gets train-wide stats. Train rows get leave-text-out stats (no leakage).
    Uses explicit .loc assignment so original row order is preserved."""
    stat_cols = ["tok_mean", "tok_median", "tok_std", "tok_count", "tok_skip_rate"]

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        a = df.groupby("tok_lc")["answer"].agg(
            tok_mean="mean",
            tok_median="median",
            tok_std="std",
            tok_count="count",
            tok_skip_rate=lambda s: float((s == 0).mean()),
        )
        return a

    train = train.copy()
    test = test.copy()
    train["tok_lc"] = train["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test["tok_lc"] = test["word"].astype(str).map(lambda w: _strip_punct(w).lower())

    # Test: aggregate over all train rows.
    a_full = _agg(train)
    test = test.merge(a_full, left_on="tok_lc", right_index=True, how="left")

    # Train: leave-text-out aggregation, written back into row positions explicitly.
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
# Surprisal from a Romanian causal LM
# ---------------------------------------------------------------------------

def _try_load_lm():
    """Try LM_CANDIDATES in order. Returns (tokenizer, model, name) or (None, None, None)."""
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


def compute_surprisal(df: pd.DataFrame, tokenizer, model) -> np.ndarray:
    """Returns -log p(word | prefix) per row (in the input row order).
    One forward pass per (text, participant, doc, page). Sliding window for long pages."""
    import torch

    device = next(model.parameters()).device
    out = np.zeros(len(df), dtype=np.float32)

    df = df.reset_index().rename(columns={"index": "_orig_idx"}).copy()
    df = df.sort_values(["text", "participant_id", "doc_num", "page_num", "word_idx"])

    @torch.no_grad()
    def score_token_ids(full_ids: List[int]) -> np.ndarray:
        """Returns -log p(token_i | full_ids[:i]) for each i. Position 0 is 0.0."""
        L = len(full_ids)
        if L < 2:
            return np.zeros(L, dtype=np.float32)
        logp = np.zeros(L, dtype=np.float32)
        if L <= LM_MAX_LEN:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(ids).logits[0]              # (L, V)
            lp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]                            # (L-1,)
            pred = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp[1:] = (-pred).cpu().numpy()
            return logp
        # Sliding window: each window size LM_MAX_LEN, stride LM_MAX_LEN//2.
        # First window scores positions 1..LM_MAX_LEN-1. Subsequent windows score
        # only their second half (positions pos+stride..pos+LM_MAX_LEN-1) so each
        # token is predicted with maximal context.
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
            pred = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # length len(chunk)-1
            # pred[k] -> abs position pos+k+1
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

    # Tokenize per group with offset_mapping for accurate word→subword alignment.
    use_offsets = tokenizer.is_fast
    bos = tokenizer.bos_token_id
    groups = df.groupby(
        ["text", "participant_id", "doc_num", "page_num"], sort=False
    )
    n_groups = len(groups)
    for gi, (_, block) in enumerate(groups):
        words = block["word"].astype(str).tolist()
        # Build text + per-word char spans.
        text_buf = []
        char_spans = []
        cursor = 0
        for i, w in enumerate(words):
            if i > 0:
                text_buf.append(" ")
                cursor += 1
            text_buf.append(w)
            char_spans.append((cursor, cursor + len(w)))
            cursor += len(w)
        text = "".join(text_buf)

        if use_offsets:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            sub_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
        else:
            sub_ids = tokenizer.encode(text, add_special_tokens=False)
            offsets = None

        full_ids = ([bos] if bos is not None else []) + list(sub_ids)
        offset_shift = 1 if bos is not None else 0
        token_surps = score_token_ids(full_ids)        # surprisal, len = len(full_ids)
        # Drop the BOS slot: align with sub_ids.
        sub_surps = token_surps[offset_shift:]

        # Map subword surprisals to words via character offsets.
        if offsets is not None:
            word_surps = np.zeros(len(words), dtype=np.float32)
            wi = 0
            for k, (s, e) in enumerate(offsets):
                if s == e:                 # special / empty
                    continue
                # Advance word index until this subword's start fits inside word span.
                while wi < len(words) and char_spans[wi][1] <= s:
                    wi += 1
                if wi < len(words) and char_spans[wi][0] <= s < char_spans[wi][1]:
                    word_surps[wi] += sub_surps[k]
        else:
            # Fallback: re-tokenize each word in isolation. Less accurate (different BPE)
            # but still gives the rough surprisal magnitude.
            word_surps = np.zeros(len(words), dtype=np.float32)
            cursor_id = 0
            for wi, w in enumerate(words):
                piece = (" " if wi > 0 else "") + w
                ids_w = tokenizer.encode(piece, add_special_tokens=False)
                k_lo = cursor_id
                k_hi = cursor_id + len(ids_w)
                if k_hi <= len(sub_surps):
                    word_surps[wi] = sub_surps[k_lo:k_hi].sum()
                cursor_id = k_hi

        out[block["_orig_idx"].to_numpy()] = word_surps
        if (gi + 1) % 50 == 0 or (gi + 1) == n_groups:
            print(f"[lm] {gi+1}/{n_groups} groups scored")
    return out


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "word_len", "alpha_len", "n_syllables",
    "is_punct", "is_url", "has_digit",
    "is_upper_first", "is_all_upper", "ends_with_punct",
    "doc_num", "page_num", "word_idx", "pos_in_page", "rel_pos_in_page", "page_size",
    "log_freq", "prev_log_freq", "next_log_freq",
    "tok_mean", "tok_median", "tok_std", "tok_count", "tok_skip_rate",
    "prev_word_len", "prev_alpha_len", "prev_n_syllables",
    "prev_is_punct", "prev_has_digit",
    "next_word_len", "next_alpha_len", "next_n_syllables",
    "next_is_punct", "next_has_digit",
]


def fit_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if np.std(y_pred) < 1e-9:
        return 1.0, 0.0
    a = np.cov(y_pred, y_true, ddof=0)[0, 1] / np.var(y_pred)
    b = float(y_true.mean() - a * y_pred.mean())
    return float(a), b


def lgb_params(seed: int) -> dict:
    return dict(
        objective="tweedie",
        tweedie_variance_power=1.4,
        metric="rmse",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbosity=-1,
        seed=seed,
    )


def main() -> None:
    train = pd.read_csv(INPUT_DIR / "train_data.csv")
    test = pd.read_csv(INPUT_DIR / "test_data.csv")
    print(f"[data] train={train.shape}  test={test.shape}")
    n_test_orig = len(test)
    test_ids_orig = set(test["datapointID"].astype(int).tolist())
    assert len(test_ids_orig) == n_test_orig, "duplicate datapointID in test_data.csv"

    train = build_word_features(train)
    test = build_word_features(test)
    train = add_context_features(train)
    test = add_context_features(test)
    train, test = add_corpus_frequency(train, test)
    train, test = add_per_word_target_stats(train, test)

    feature_cols = list(FEATURE_COLS)
    if USE_LM:
        tokenizer, model, lm_name = _try_load_lm()
        if model is None:
            print("[lm] no LM loaded; continuing without surprisal")
        else:
            print(f"[lm] computing surprisal for train ({lm_name})")
            train["surprisal"] = compute_surprisal(train, tokenizer, model)
            print(f"[lm] computing surprisal for test ({lm_name})")
            test["surprisal"] = compute_surprisal(test, tokenizer, model)
            feature_cols.append("surprisal")
            del model, tokenizer
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        print("[lm] skipped (USE_LM=0)")

    # Sanity: feature columns must all be present and non-NaN.
    for c in feature_cols:
        assert c in train.columns, f"missing feature in train: {c}"
        assert c in test.columns, f"missing feature in test: {c}"
    n_nan_train = int(train[feature_cols].isna().sum().sum())
    n_nan_test = int(test[feature_cols].isna().sum().sum())
    assert n_nan_train == 0, f"NaN in train features: {n_nan_train}"
    assert n_nan_test == 0, f"NaN in test features: {n_nan_test}"
    print(f"[feat] {len(feature_cols)} features, NaNs train/test = 0/0")

    train = train.sort_values(
        ["text", "participant_id", "doc_num", "page_num", "word_idx"]
    ).reset_index(drop=True)

    X = train[feature_cols].values
    y = train["answer"].values.astype(float)
    groups = train["text"].values
    Xt = test[feature_cols].values

    n_splits = min(N_FOLDS, train["text"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)
    fold_scores = []

    splits = list(gkf.split(X, y, groups))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        seed_oof = np.zeros(len(va_idx), dtype=float)
        seed_test = np.zeros(len(test), dtype=float)
        for si in range(N_SEEDS):
            seed = SEED + 1000 * si + fi
            dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
            dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
            booster = lgb.train(
                lgb_params(seed), dtr, num_boost_round=4000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            seed_oof += booster.predict(X[va_idx], num_iteration=booster.best_iteration)
            seed_test += booster.predict(Xt, num_iteration=booster.best_iteration)
        seed_oof /= N_SEEDS
        seed_test /= N_SEEDS
        oof[va_idx] = seed_oof
        test_preds += seed_test / n_splits
        score = comp_metric(y[va_idx], seed_oof)
        fold_scores.append(score)
        print(f"[fold {fi}] held-out={sorted(set(groups[va_idx]))}  score={score:.3f}")

    print(f"[cv] folds: mean={np.mean(fold_scores):.3f}  std={np.std(fold_scores):.3f}")
    print(f"[cv] global OOF score (raw)        = {comp_metric(y, oof):.3f}")

    a, b = fit_calibration(y, oof)
    print(f"[calib] a={a:.4f}  b={b:.4f}")
    oof_cal = np.clip(a * oof + b, 0, None)
    test_preds_cal = np.clip(a * test_preds + b, 0, None)
    print(f"[cv] global OOF score (calibrated) = {comp_metric(y, oof_cal):.3f}")

    print(f"[pred] test pred stats: "
          f"min={test_preds_cal.min():.1f}  mean={test_preds_cal.mean():.1f}  "
          f"max={test_preds_cal.max():.1f}  std={test_preds_cal.std():.1f}  "
          f"#zeros={(test_preds_cal < 1).sum()}")

    # Build submission, sorted by datapointID, with strict self-checks.
    sub = pd.DataFrame({
        "subtaskID": np.ones(len(test), dtype=int),
        "datapointID": test["datapointID"].astype(int).values,
        "answer": np.round(test_preds_cal).astype(int),
    }).sort_values("datapointID").reset_index(drop=True)

    # Self-checks: must hold for a 100%-format submission.
    assert list(sub.columns) == ["subtaskID", "datapointID", "answer"], sub.columns
    assert len(sub) == n_test_orig, f"row count {len(sub)} != test rows {n_test_orig}"
    assert set(sub["datapointID"].tolist()) == test_ids_orig, \
        "submission datapointIDs do not match test_data.csv"
    assert sub["datapointID"].is_monotonic_increasing, "datapointID not sorted ascending"
    assert sub["subtaskID"].eq(1).all(), "subtaskID must be 1 for all rows"
    assert sub["answer"].notna().all(), "NaN in answer column"
    assert (sub["answer"] >= 0).all(), "negative values in answer column"
    assert sub["answer"].nunique() > 1, "answer is constant — would score 0"

    sub_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[out] wrote {sub_path}  rows={len(sub)}")
    print("[out] first 6 rows:")
    print(sub.head(6).to_string(index=False))
    print("[out] last 3 rows:")
    print(sub.tail(3).to_string(index=False))
    print("[out] all self-checks passed.")


if __name__ == "__main__":
    main()
