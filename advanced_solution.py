"""
Nitro NLP — Romanian eye-tracking reading-time prediction.
"Winning" pipeline: handcrafted features + Romanian-LM surprisal + XLM-R
embeddings + LightGBM/Ridge stack, all trained on participant-averaged
targets (Bayes-optimal for unseen-reader test split).

Designed to run on a Kaggle T4/P100 notebook (Internet ON), or locally
without GPU when USE_LM=0 USE_EMB=0.

Score on the official metric:
    100 * (max(0, R^2) + |Pearson|) / 2
"""

from __future__ import annotations
import glob
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config (env-overridable)
# ----------------------------------------------------------------------
USE_LM = int(os.environ.get("USE_LM", "1"))    # surprisal from Romanian GPT-2
USE_EMB = int(os.environ.get("USE_EMB", "1"))  # XLM-R subword count + PCA embeddings
USE_POS = int(os.environ.get("USE_POS", "1"))  # POS tags + lemma freq via Stanza
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
SEEDS = [42, 1337, 2024, 7]
PCA_DIMS = 8
SMOOTH_K = 20  # Bayesian smoothing for token target encoding
# Aggregation method for the per-word target (mean of 30 readers' TRTs).
# Some readers may be outliers; median / trimmed-mean can be more robust.
AGG_METHOD = os.environ.get("AGG_METHOD", "mean").lower()  # mean | median | trimmed
TRIM_PCT = float(os.environ.get("TRIM_PCT", "0.10"))       # only used when trimmed
LM_CANDIDATES = os.environ.get(
    "LM_CANDIDATES",
    "readerbench/RoGPT2-large,readerbench/RoGPT2-medium,"
    "readerbench/RoGPT2-base,dumitrescustefan/gpt-neo-romanian-125m"
).split(",")
EMB_MODEL = os.environ.get("EMB_MODEL", "FacebookAI/xlm-roberta-base")

# Paths: prefer Kaggle's /kaggle/input layout, fall back to script/notebook cwd.
# Works whether the file is run as a script (__file__ defined) or pasted into a
# Jupyter/Kaggle notebook cell (__file__ undefined).
def _find_inputs():
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append(os.getcwd())
    for here in candidates:
        if os.path.exists(os.path.join(here, "train_data.csv")):
            return here, here
    for p in glob.glob("/kaggle/input/**/train_data.csv", recursive=True):
        return os.path.dirname(p), "/kaggle/working"
    raise FileNotFoundError(
        "train_data.csv not found in script dir, cwd, or /kaggle/input/**"
    )

INPUT_DIR, OUTPUT_DIR = _find_inputs()
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_CSV = os.path.join(INPUT_DIR, "train_data.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test_data.csv")
OUT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")

print(f"[cfg] INPUT_DIR={INPUT_DIR}  OUTPUT_DIR={OUTPUT_DIR}")
print(f"[cfg] USE_LM={USE_LM}  USE_EMB={USE_EMB}  N_FOLDS={N_FOLDS}  SEEDS={SEEDS}")


# ----------------------------------------------------------------------
# 1. Load + parse word_id
# ----------------------------------------------------------------------
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print(f"[data] train={train.shape}  test={test.shape}")

# ----------------------------------------------------------------------
# Diagnostic: what's the upper bound of the per-row competition score
# achievable by ANY model that predicts E[TRT|word] and broadcasts to
# unseen readers?  R^2_max = sigma_word^2 / sigma_total^2,  Pearson_max
# = sqrt of that.  Tells us how much of the metric is *attainable* and
# how much is irreducible reader noise.
# ----------------------------------------------------------------------
_y = train["answer"].astype(float).values
_word_mean = (train.groupby(["word_id"])["answer"]
                   .transform("mean").astype(float).values)
_var_total = float(np.var(_y))
_var_word = float(np.var(_word_mean))
_var_reader = max(0.0, _var_total - _var_word)
_p_max = float(np.sqrt(_var_word / _var_total)) if _var_total > 0 else 0.0
_r2_max = _var_word / _var_total if _var_total > 0 else 0.0
_score_max = 100.0 * (_r2_max + _p_max) / 2.0
print(f"[ceiling] var(per-row TRT)        = {_var_total:.0f}")
print(f"[ceiling] var(per-word mean TRT)  = {_var_word:.0f}")
print(f"[ceiling] var(within-word/reader) = {_var_reader:.0f}")
print(f"[ceiling] per-row Pearson_max     = {_p_max:.4f}")
print(f"[ceiling] per-row R^2_max         = {_r2_max:.4f}")
print(f"[ceiling] per-row SCORE_max       = {_score_max:.3f}  "
      f"<-- we cannot go above this with the broadcast strategy")

WID_RE = re.compile(r"page_(\d+)_(\d+)$")

def parse_wid(wid):
    m = WID_RE.search(wid)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

for df in (train, test):
    pp = df["word_id"].apply(parse_wid)
    df["page"] = pp.str[0]
    df["pos"] = pp.str[1]


# ----------------------------------------------------------------------
# 2. Handcrafted word features
# ----------------------------------------------------------------------
ROM_VOWELS = set("aeiouăâîAEIOUĂÂÎ")
PUNCT_CHARS = set(".,;:!?…—–-()[]{}\"'`«»“”‘’/\\")

def word_feats(w):
    w = "" if w is None else str(w)
    n = len(w)
    n_alpha = sum(c.isalpha() for c in w)
    n_vowels = sum(c in ROM_VOWELS for c in w)
    n_digit = sum(c.isdigit() for c in w)
    is_punct = int(n > 0 and all(c in PUNCT_CHARS for c in w))
    cap_first = int(n > 0 and w[0].isupper())
    is_url = int(("http" in w) or ("www." in w) or ("://" in w))
    has_diacritic = int(any(c in "ăâîșțĂÂÎȘȚ" for c in w))
    return n, n_alpha, n_vowels, n_digit, is_punct, cap_first, is_url, has_diacritic

WORD_COLS = ["len", "n_alpha", "n_vowels", "n_digit",
             "is_punct", "cap_first", "is_url", "has_diac"]
for df in (train, test):
    feats = df["word"].apply(lambda w: pd.Series(word_feats(w)))
    feats.columns = WORD_COLS
    df[WORD_COLS] = feats


# ----------------------------------------------------------------------
# 3. Frequency: log corpus freq (train+test) + neighbor freqs
# ----------------------------------------------------------------------
for df in (train, test):
    df["lw"] = df["word"].astype(str).str.lower()

corpus = pd.concat([train[["lw"]], test[["lw"]]], ignore_index=True)
freq = corpus.groupby("lw").size()
total = freq.sum()
log_freq = np.log(freq / total)

for df in (train, test):
    df["log_freq"] = df["lw"].map(log_freq).astype(float)

def add_neighbor_freq(df):
    uw = (df[["text", "page", "pos", "log_freq"]]
          .drop_duplicates(["text", "page", "pos"])
          .sort_values(["text", "page", "pos"])
          .reset_index(drop=True))
    g = uw.groupby(["text", "page"])["log_freq"]
    uw["prev_lf"] = g.shift(1)
    uw["next_lf"] = g.shift(-1)
    uw["prev2_lf"] = g.shift(2)
    return df.merge(
        uw[["text", "page", "pos", "prev_lf", "next_lf", "prev2_lf"]],
        on=["text", "page", "pos"], how="left")

train = add_neighbor_freq(train)
test = add_neighbor_freq(test)


# ----------------------------------------------------------------------
# 4. Position-in-page (relative)
# ----------------------------------------------------------------------
for df in (train, test):
    page_max = (df.groupby(["text", "page"])["pos"]
                  .transform("max").replace(0, 1))
    df["rel_pos"] = df["pos"] / page_max


# ----------------------------------------------------------------------
# 4b. Sentence-position features.  Eye-tracking literature: sentence-final
# words receive longer fixations (wrap-up effect); sentence-initial words
# also have distinct patterns; distance from the last punctuation captures
# locality / clause-boundary effects.
# ----------------------------------------------------------------------
SENT_END = {".", "!", "?"}
PUNCT_BREAK = {".", ",", ";", ":", "!", "?"}

def add_sentence_features(df):
    uw = (df.drop_duplicates(["text", "page", "pos"])
            .sort_values(["text", "page", "pos"])
            .reset_index(drop=True))
    sent_idx_l, pos_in_sent_l, dist_punct_l = [], [], []
    last_tp = None
    cur_sent = cur_pos = 0
    last_punct_pos = -1
    for w, txt, pg in zip(uw["word"].astype(str), uw["text"], uw["page"]):
        tp = (txt, pg)
        if tp != last_tp:
            cur_sent = cur_pos = 0
            last_punct_pos = -1
            last_tp = tp
        sent_idx_l.append(cur_sent)
        pos_in_sent_l.append(cur_pos)
        dist_punct_l.append(cur_pos - last_punct_pos
                            if last_punct_pos >= 0 else cur_pos)
        last_char = w[-1] if w else ""
        # Handle quoted sentence endings like ." or ?'
        is_sent_term = (last_char in SENT_END or
                        (len(w) >= 2 and w[-1] in {'"', "'", "”", "’"}
                         and w[-2] in SENT_END))
        if is_sent_term:
            cur_sent += 1
            cur_pos = 0
            last_punct_pos = -1
        else:
            if last_char in PUNCT_BREAK or w in PUNCT_BREAK:
                last_punct_pos = cur_pos
            cur_pos += 1

    uw["sent_idx"] = sent_idx_l
    uw["pos_in_sent"] = pos_in_sent_l
    uw["dist_from_punct"] = dist_punct_l
    sent_size = (uw.groupby(["text", "page", "sent_idx"])["pos_in_sent"]
                   .transform("max") + 1)
    uw["sent_len"] = sent_size
    uw["rel_pos_in_sent"] = uw["pos_in_sent"] / uw["sent_len"].clip(lower=1)
    uw["is_sent_final"] = (uw["pos_in_sent"] == uw["sent_len"] - 1).astype(int)
    uw["is_sent_initial"] = (uw["pos_in_sent"] == 0).astype(int)
    cols = ["sent_idx", "pos_in_sent", "sent_len", "rel_pos_in_sent",
            "is_sent_final", "is_sent_initial", "dist_from_punct"]
    return df.merge(uw[["text", "page", "pos"] + cols],
                    on=["text", "page", "pos"], how="left")

train = add_sentence_features(train)
test = add_sentence_features(test)


# ----------------------------------------------------------------------
# 5. (GPU) Surprisal from a Romanian GPT-2 (page-level forward)
#     Adds:  surprisal, n_subtok
# ----------------------------------------------------------------------
SURPRISAL_DEFAULT = float("nan")  # filled in median-impute below if skipped

def _build_pages(df):
    """Return DataFrame with one row per (text, page), tokens = list of words."""
    uw = (df[["text", "page", "pos", "word"]]
          .drop_duplicates(["text", "page", "pos"])
          .sort_values(["text", "page", "pos"]))
    pages = (uw.groupby(["text", "page"])
               .agg(words=("word", list), positions=("pos", list))
               .reset_index())
    return pages

def compute_lm_features(train_df, test_df):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lm] device={device}")
    tok = mdl = name = None
    for cand in LM_CANDIDATES:
        try:
            t = AutoTokenizer.from_pretrained(cand.strip())
            m = AutoModelForCausalLM.from_pretrained(cand.strip()).to(device).eval()
            tok, mdl, name = t, m, cand.strip()
            print(f"[lm] loaded {name}")
            break
        except Exception as e:
            print(f"[lm] failed {cand}: {type(e).__name__}: {str(e)[:80]}")
    if mdl is None:
        print("[lm] no LM loaded — skipping surprisal")
        return None, None

    bos_id = (tok.bos_token_id if tok.bos_token_id is not None
              else tok.eos_token_id)

    @torch.no_grad()
    def page_surprisal(words):
        # Tokenize each word separately, keep alignment
        ids_per_word, n_sub = [], []
        for i, w in enumerate(words):
            prefix = "" if i == 0 else " "
            ids = tok.encode(prefix + str(w), add_special_tokens=False)
            if not ids:
                ids = tok.encode(" ?", add_special_tokens=False)  # fallback
            ids_per_word.append(ids)
            n_sub.append(len(ids))
        words_flat = [i for ids in ids_per_word for i in ids]
        # Prepend a BOS so the FIRST real subword has a valid predecessor.
        # Without this, sentence-initial words get NaN surprisal and we lose
        # the strongest signal at start-of-page positions.
        if bos_id is not None:
            flat = [bos_id] + words_flat
            offset = 1
        else:
            flat = words_flat
            offset = 0
        # Sliding window if too long
        max_len = 1024
        log_probs_full = np.full(len(flat), np.nan)
        if len(flat) <= max_len:
            x = torch.tensor([flat], device=device)
            logits = mdl(x).logits[0]  # (T, V)
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = x[0, 1:]
            tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy()
            log_probs_full[1:] = tok_lp
        else:
            stride = max_len // 2
            for s in range(0, len(flat), stride):
                e = min(s + max_len, len(flat))
                seg = flat[s:e]
                x = torch.tensor([seg], device=device)
                logits = mdl(x).logits[0]
                lp = torch.log_softmax(logits[:-1], dim=-1)
                tgt = x[0, 1:]
                tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy()
                fill_start = s + 1 if s == 0 else s + stride // 2 + 1
                for j in range(fill_start, e):
                    log_probs_full[j] = tok_lp[j - s - 1]
                if e == len(flat):
                    break
        # Drop the BOS slot so log_probs[k] aligns with words_flat[k].
        log_probs = log_probs_full[offset:]
        # Aggregate to word level: sum log prob over its subword tokens
        out, idx = [], 0
        for ids in ids_per_word:
            seg = log_probs[idx: idx + len(ids)]
            idx += len(ids)
            with np.errstate(invalid="ignore"):
                out.append(float(-np.nansum(seg)) if np.isfinite(seg).any() else np.nan)
        return out, n_sub

    pages = _build_pages(pd.concat([train_df, test_df], ignore_index=True))
    surp_rows, n_sub_rows = [], []
    for i, r in pages.iterrows():
        sp, ns = page_surprisal(r["words"])
        for w_pos, s, n in zip(r["positions"], sp, ns):
            surp_rows.append((r["text"], r["page"], w_pos, s))
            n_sub_rows.append((r["text"], r["page"], w_pos, n))
        if (i + 1) % 20 == 0 or i == len(pages) - 1:
            print(f"[lm] page {i+1}/{len(pages)}")
    surp = pd.DataFrame(surp_rows, columns=["text", "page", "pos", "surprisal"])
    nsub = pd.DataFrame(n_sub_rows, columns=["text", "page", "pos", "n_subtok"])
    return surp, nsub

if USE_LM:
    try:
        surp_df, nsub_df = compute_lm_features(train, test)
    except Exception as e:
        print(f"[lm] error: {e}")
        surp_df, nsub_df = None, None
else:
    surp_df, nsub_df = None, None

if surp_df is not None:
    train = train.merge(surp_df, on=["text", "page", "pos"], how="left")
    test = test.merge(surp_df, on=["text", "page", "pos"], how="left")
if nsub_df is not None:
    train = train.merge(nsub_df, on=["text", "page", "pos"], how="left")
    test = test.merge(nsub_df, on=["text", "page", "pos"], how="left")


# ----------------------------------------------------------------------
# 6. (GPU) XLM-RoBERTa subword count + word embedding PCA
#     Adds:  n_subtok_xlmr, emb_pca_0..7
# ----------------------------------------------------------------------
def compute_xlmr_features(train_df, test_df):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(EMB_MODEL, add_prefix_space=True)
        mdl = AutoModel.from_pretrained(EMB_MODEL).to(device).eval()
        print(f"[emb] loaded {EMB_MODEL} on {device}")
    except Exception as e:
        print(f"[emb] failed: {e}")
        return None, None

    @torch.no_grad()
    def page_embed(words):
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=512)
        word_ids = enc.word_ids(0)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = mdl(input_ids=ids, attention_mask=mask).last_hidden_state[0].cpu().numpy()
        # Average subword vectors per word index
        n_words = len(words)
        D = out.shape[1]
        sums = np.zeros((n_words, D), dtype=np.float32)
        cnt = np.zeros(n_words, dtype=np.int32)
        for tok_i, w_i in enumerate(word_ids):
            if w_i is None:
                continue
            sums[w_i] += out[tok_i]
            cnt[w_i] += 1
        cnt_safe = np.clip(cnt, 1, None)
        avg = sums / cnt_safe[:, None]
        return avg, cnt  # cnt = subword count per word

    pages = _build_pages(pd.concat([train_df, test_df], ignore_index=True))
    rows = []
    for i, r in pages.iterrows():
        words = r["words"]
        avg, cnt = page_embed(words)
        for j, w_pos in enumerate(r["positions"]):
            rows.append((r["text"], r["page"], w_pos, int(cnt[j]), avg[j]))
        if (i + 1) % 20 == 0 or i == len(pages) - 1:
            print(f"[emb] page {i+1}/{len(pages)}")
    df = pd.DataFrame(rows, columns=["text", "page", "pos", "n_subtok_xlmr", "emb"])
    # PCA
    E = np.stack(df["emb"].values)
    scaler = StandardScaler().fit(E)
    pca = PCA(n_components=PCA_DIMS, random_state=0).fit(scaler.transform(E))
    Z = pca.transform(scaler.transform(E)).astype(np.float32)
    pca_cols = [f"emb_pca_{i}" for i in range(PCA_DIMS)]
    df[pca_cols] = Z
    return df.drop(columns=["emb"]), pca_cols

if USE_EMB:
    try:
        emb_df, pca_cols = compute_xlmr_features(train, test)
    except Exception as e:
        print(f"[emb] error: {e}")
        emb_df, pca_cols = None, []
else:
    emb_df, pca_cols = None, []

if emb_df is not None:
    train = train.merge(emb_df, on=["text", "page", "pos"], how="left")
    test = test.merge(emb_df, on=["text", "page", "pos"], how="left")


# ----------------------------------------------------------------------
# 6b. POS tags + lemma frequency via Stanza (Romanian).
#     Adds:  upos_id (categorical id), is_content, is_function, log_lemma_freq
#     Cached to OUTPUT_DIR/pos_cache.csv so re-runs skip the tagging step.
# ----------------------------------------------------------------------
UPOS_LIST = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
             "AUX", "CCONJ", "SCONJ", "NUM", "PART", "INTJ", "SYM", "PUNCT", "X"]
UPOS_MAP = {u: i for i, u in enumerate(UPOS_LIST)}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PRON", "PART"}

def compute_pos_features(train_df, test_df):
    try:
        import stanza
    except ImportError:
        print("[pos] stanza not installed — skipping (pip install stanza)")
        return None
    try:
        stanza.download("ro", verbose=False)
    except Exception as e:
        print(f"[pos] model download failed: {e} — skipping")
        return None
    try:
        import torch as _torch
        gpu = _torch.cuda.is_available()
    except Exception:
        gpu = False
    print(f"[pos] loading Stanza Romanian pipeline (gpu={gpu})...")
    try:
        nlp = stanza.Pipeline(
            "ro", processors="tokenize,pos,lemma",
            tokenize_pretokenized=True, use_gpu=gpu, verbose=False)
    except Exception as e:
        print(f"[pos] pipeline init failed: {e} — skipping")
        return None

    uw = (pd.concat([train_df[["text", "page", "pos", "word"]],
                     test_df[["text", "page", "pos", "word"]]],
                    ignore_index=True)
            .drop_duplicates(["text", "page", "pos"])
            .sort_values(["text", "page", "pos"]))

    rows = []
    n_pages = uw[["text", "page"]].drop_duplicates().shape[0]
    for i, ((txt, pg), g) in enumerate(uw.groupby(["text", "page"]), 1):
        words = g["word"].astype(str).tolist()
        positions = [int(x) for x in g["pos"].tolist()]
        # Split page into sentences using SENT_END (already defined above).
        sents, sent_pos = [[]], [[]]
        for w, p in zip(words, positions):
            sents[-1].append(w)
            sent_pos[-1].append(p)
            last = w[-1] if w else ""
            is_end = (last in SENT_END or
                      (len(w) >= 2 and w[-1] in {'"', "'", "”", "’"}
                       and w[-2] in SENT_END))
            if is_end:
                sents.append([]); sent_pos.append([])
        sents = [s for s in sents if s]
        sent_pos = [s for s in sent_pos if s]
        if not sents:
            continue
        try:
            doc = nlp(sents)
        except Exception as e:
            print(f"[pos] page ({txt}, {pg}) failed: {e}")
            for ps in sent_pos:
                for p in ps:
                    rows.append((txt, int(pg), p, "X", ""))
            continue
        for sent, ps in zip(doc.sentences, sent_pos):
            tokens = sent.tokens
            if len(tokens) == len(ps):
                for tok, p in zip(tokens, ps):
                    w0 = tok.words[0]
                    rows.append((txt, int(pg), p,
                                 (w0.upos or "X"),
                                 (w0.lemma or tok.text).lower()))
            else:
                # Misalignment from MWT/clitic expansion — fallback
                for p in ps:
                    rows.append((txt, int(pg), p, "X", ""))
        if i % 20 == 0 or i == n_pages:
            print(f"[pos] tagged page {i}/{n_pages}")
    return pd.DataFrame(rows, columns=["text", "page", "pos", "upos", "lemma"])


pos_df = None
POS_CACHE = os.path.join(OUTPUT_DIR, "pos_cache.csv")
if USE_POS:
    if os.path.exists(POS_CACHE):
        try:
            pos_df = pd.read_csv(POS_CACHE)
            print(f"[pos] loaded cache {POS_CACHE} ({len(pos_df)} rows)")
        except Exception as e:
            print(f"[pos] cache read failed: {e} — recomputing")
            pos_df = None
    if pos_df is None:
        pos_df = compute_pos_features(train, test)
        if pos_df is not None:
            try:
                pos_df.to_csv(POS_CACHE, index=False)
                print(f"[pos] wrote cache {POS_CACHE} ({len(pos_df)} rows)")
            except Exception as e:
                print(f"[pos] cache write failed: {e}")

if pos_df is not None:
    train = train.merge(pos_df, on=["text", "page", "pos"], how="left")
    test = test.merge(pos_df, on=["text", "page", "pos"], how="left")
    for df in (train, test):
        df["upos"] = df["upos"].fillna("X")
        df["lemma"] = df["lemma"].fillna("").astype(str).str.lower()
        df["upos_id"] = df["upos"].map(UPOS_MAP).fillna(UPOS_MAP["X"]).astype(int)
        df["is_content"] = df["upos"].isin(CONTENT_POS).astype(int)
        df["is_function"] = df["upos"].isin(FUNCTION_POS).astype(int)
    # Lemma frequency over train+test corpus
    lemma_corpus = pd.concat([train["lemma"], test["lemma"]], ignore_index=True)
    lemma_freq = lemma_corpus[lemma_corpus.ne("")].value_counts()
    total_lemma = float(lemma_freq.sum()) or 1.0
    log_lemma_freq_map = np.log(lemma_freq / total_lemma)
    fallback_llf = float(np.log(0.5 / total_lemma))
    for df in (train, test):
        df["log_lemma_freq"] = (df["lemma"].map(log_lemma_freq_map)
                                  .fillna(fallback_llf).astype(float))


# ----------------------------------------------------------------------
# 7. Page / text level aggregate features
# ----------------------------------------------------------------------
def add_page_aggs(df):
    uw = df.drop_duplicates(["text", "page", "pos"])
    page_stats = (uw.groupby(["text", "page"])
                    .agg(page_len=("len", "mean"),
                         page_lf=("log_freq", "mean"),
                         page_size=("pos", "max"))
                    .reset_index())
    return df.merge(page_stats, on=["text", "page"], how="left")

train = add_page_aggs(train)
test = add_page_aggs(test)


# ----------------------------------------------------------------------
# 8. Leave-text-out per-token target encoding (Bayesian smoothed)
# ----------------------------------------------------------------------
PRIOR = float(train["answer"].mean())
tok_sum_all = train.groupby("lw")["answer"].sum()
tok_cnt_all = train.groupby("lw").size()

tx = (train.groupby(["text", "lw"])["answer"]
        .agg(["sum", "count"]).reset_index()
        .rename(columns={"sum": "tx_sum", "count": "tx_cnt"}))
tx["tot_sum"] = tx["lw"].map(tok_sum_all)
tx["tot_cnt"] = tx["lw"].map(tok_cnt_all)
tx["lto_sum"] = tx["tot_sum"] - tx["tx_sum"]
tx["lto_cnt"] = tx["tot_cnt"] - tx["tx_cnt"]
tx["tok_mean_lto"] = (tx["lto_sum"] + PRIOR * SMOOTH_K) / (tx["lto_cnt"] + SMOOTH_K)

train = train.merge(tx[["text", "lw", "tok_mean_lto", "lto_cnt"]],
                    on=["text", "lw"], how="left")
train["tok_mean_lto"] = train["tok_mean_lto"].fillna(PRIOR)
train["lto_cnt"] = train["lto_cnt"].fillna(0)

global_smoothed = (tok_sum_all + PRIOR * SMOOTH_K) / (tok_cnt_all + SMOOTH_K)
test["tok_mean_lto"] = test["lw"].map(global_smoothed).fillna(PRIOR)
test["lto_cnt"] = test["lw"].map(tok_cnt_all).fillna(0)


# ----------------------------------------------------------------------
# 9. Final feature list  (+ NaN imputation)
# ----------------------------------------------------------------------
FEATURES = WORD_COLS + [
    "log_freq", "prev_lf", "next_lf", "prev2_lf",
    "tok_mean_lto", "lto_cnt",
    "page", "pos", "rel_pos",
    "page_len", "page_lf", "page_size",
    # sentence-position features (wrap-up effect, clause boundaries)
    "sent_idx", "pos_in_sent", "sent_len", "rel_pos_in_sent",
    "is_sent_final", "is_sent_initial", "dist_from_punct",
]
if surp_df is not None:
    FEATURES += ["surprisal", "n_subtok"]
if emb_df is not None:
    FEATURES += ["n_subtok_xlmr"] + pca_cols
if pos_df is not None:
    FEATURES += ["upos_id", "is_content", "is_function", "log_lemma_freq"]

# Median-impute remaining NaNs from training
for c in FEATURES:
    if train[c].isna().any() or test[c].isna().any():
        m = train[c].median()
        train[c] = train[c].fillna(m)
        test[c] = test[c].fillna(m)

print(f"[feat] {len(FEATURES)} features")


# ----------------------------------------------------------------------
# 10. Aggregate training to per-(text, page, pos) — average over readers
#     This is the key Bayes-optimal trick for the unseen-reader test.
# ----------------------------------------------------------------------
agg_keys = ["text", "page", "pos"]
_agg_feat_cols = [c for c in FEATURES if c not in agg_keys]

def _agg_target(s):
    if AGG_METHOD == "median":
        return float(s.median())
    if AGG_METHOD == "trimmed":
        v = np.sort(s.to_numpy(dtype=float))
        n = max(1, int(TRIM_PCT * len(v)))
        v = v[n:-n] if len(v) > 2 * n else v
        return float(v.mean())
    return float(s.mean())

agg = (train.groupby(agg_keys)
         .agg(answer=("answer", _agg_target),
              **{c: (c, "first") for c in _agg_feat_cols})
         .reset_index())
print(f"[agg] AGG_METHOD={AGG_METHOD}  rows {len(train)} -> {len(agg)}")


# ----------------------------------------------------------------------
# 11. Models — multi-seed LightGBM (Tweedie + log1p) + Ridge
#      Stacked with NNLS on OOF
# ----------------------------------------------------------------------
def custom_score(y_true, y_pred):
    r2 = max(0.0, r2_score(y_true, y_pred))
    p = pearsonr(y_true, y_pred)[0]
    p = 0.0 if np.isnan(p) else abs(p)
    return 100.0 * (r2 + p) / 2.0

X_agg = agg[FEATURES].values
y_agg = agg["answer"].values
g_agg = agg["text"].values
X_test = test[FEATURES].values

n_folds = min(N_FOLDS, agg["text"].nunique())
gkf = GroupKFold(n_splits=n_folds)

oof_lgb_tw = np.zeros(len(agg))
oof_lgb_lg = np.zeros(len(agg))
oof_ridge = np.zeros(len(agg))
test_lgb_tw = np.zeros(len(test))
test_lgb_lg = np.zeros(len(test))
test_ridge = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_agg, y_agg, groups=g_agg)):
    held = sorted(set(g_agg[va_idx]))

    # ---- LightGBM with Tweedie loss ----
    p_va = np.zeros(len(va_idx)); p_te = np.zeros(len(test))
    for s in SEEDS:
        m = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=4000, learning_rate=0.025,
            num_leaves=63, min_data_in_leaf=20,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            lambda_l2=1.0, random_state=s, n_jobs=-1, verbose=-1)
        m.fit(X_agg[tr_idx], y_agg[tr_idx],
              eval_set=[(X_agg[va_idx], y_agg[va_idx])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        p_va += m.predict(X_agg[va_idx]) / len(SEEDS)
        p_te += m.predict(X_test) / len(SEEDS)
    oof_lgb_tw[va_idx] = p_va
    test_lgb_tw += p_te / n_folds

    # ---- LightGBM with regression on log1p(target) ----
    y_log = np.log1p(y_agg[tr_idx])
    p_va = np.zeros(len(va_idx)); p_te = np.zeros(len(test))
    for s in SEEDS:
        m = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=4000, learning_rate=0.025,
            num_leaves=63, min_data_in_leaf=20,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            lambda_l2=1.0, random_state=s, n_jobs=-1, verbose=-1)
        m.fit(X_agg[tr_idx], y_log,
              eval_set=[(X_agg[va_idx], np.log1p(y_agg[va_idx]))],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        p_va += np.expm1(m.predict(X_agg[va_idx])) / len(SEEDS)
        p_te += np.expm1(m.predict(X_test)) / len(SEEDS)
    oof_lgb_lg[va_idx] = p_va
    test_lgb_lg += p_te / n_folds

    # ---- Ridge on standardized features ----
    sc = StandardScaler().fit(X_agg[tr_idx])
    rx_tr = sc.transform(X_agg[tr_idx])
    rx_va = sc.transform(X_agg[va_idx])
    rx_te = sc.transform(X_test)
    rg = Ridge(alpha=5.0).fit(rx_tr, y_agg[tr_idx])
    oof_ridge[va_idx] = rg.predict(rx_va)
    test_ridge += rg.predict(rx_te) / n_folds

    s_tw = custom_score(y_agg[va_idx], oof_lgb_tw[va_idx])
    s_lg = custom_score(y_agg[va_idx], oof_lgb_lg[va_idx])
    s_rg = custom_score(y_agg[va_idx], oof_ridge[va_idx])
    print(f"[fold {fold}] held={held}  lgb_tw={s_tw:.2f}  lgb_log={s_lg:.2f}  ridge={s_rg:.2f}")


# ----------------------------------------------------------------------
# 12. Stack on aggregated OOF with NNLS, then expand to per-row for the
#     "real" CV score (mirrors how the metric will be computed on test).
# ----------------------------------------------------------------------
M_oof = np.column_stack([oof_lgb_tw, oof_lgb_lg, np.clip(oof_ridge, 0, None)])
M_test = np.column_stack([test_lgb_tw, test_lgb_lg, np.clip(test_ridge, 0, None)])

# Add a bias column for NNLS (acts like an intercept >= 0)
A = np.column_stack([M_oof, np.ones(len(M_oof))])
w, _ = nnls(A, y_agg)
print(f"[stack] NNLS weights (lgb_tw, lgb_log, ridge, bias) = "
      f"{[round(float(x), 4) for x in w]}")
oof_stack = A @ w
test_stack = np.column_stack([M_test, np.ones(len(M_test))]) @ w

# Linear calibration on aggregated OOF
calib = LinearRegression().fit(oof_stack.reshape(-1, 1), y_agg)
a_, b_ = float(calib.coef_[0]), float(calib.intercept_)
oof_cal = a_ * oof_stack + b_
test_cal = np.clip(a_ * test_stack + b_, 0.0, None)
print(f"[calib] a={a_:.4f}  b={b_:.4f}")

# --- Real CV score: expand aggregated OOF preds back to per-row training data ---
oof_map = dict(zip(map(tuple, agg[agg_keys].values), oof_cal))
train_keys = list(map(tuple, train[agg_keys].values))
oof_per_row = np.array([oof_map[k] for k in train_keys])
y_per_row = train["answer"].values

print()
print(f"[cv] aggregated OOF Pearson = {pearsonr(y_agg, oof_cal)[0]:.4f}")
print(f"[cv] aggregated OOF R^2     = {r2_score(y_agg, oof_cal):.4f}")
print(f"[cv] aggregated OOF score   = {custom_score(y_agg, oof_cal):.3f}")
print(f"[cv] PER-ROW   OOF Pearson  = {pearsonr(y_per_row, oof_per_row)[0]:.4f}")
print(f"[cv] PER-ROW   OOF R^2      = {r2_score(y_per_row, oof_per_row):.4f}")
print(f"[cv] PER-ROW   OOF score    = {custom_score(y_per_row, oof_per_row):.3f}  <-- expected leaderboard")


# ----------------------------------------------------------------------
# 13. Submission (broadcast same prediction to all 5 test participants)
# ----------------------------------------------------------------------
test_pred_map = dict(zip(map(tuple, test[agg_keys].values), test_cal))
# `test` is already at per-(word, participant) granularity; just look up by key
preds = np.array([test_pred_map[k] for k in map(tuple, test[agg_keys].values)])

sub = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": test["datapointID"].values,
    "answer": preds,
}).sort_values("datapointID").reset_index(drop=True)

# Paranoid checks
assert list(sub.columns) == ["subtaskID", "datapointID", "answer"]
assert len(sub) == len(test)
assert sub["datapointID"].is_unique
assert sub["datapointID"].is_monotonic_increasing
assert (sub["subtaskID"] == 1).all()
assert sub["answer"].notna().all()
assert (sub["answer"] >= 0).all()
assert sub["answer"].nunique() > 1, "constant predictor scores 0"

sub.to_csv(OUT_CSV, index=False)
print(f"[out] wrote {OUT_CSV}  rows={len(sub)}")
print(f"[pred] test stats: min={preds.min():.1f}  mean={preds.mean():.1f}  "
      f"max={preds.max():.1f}  std={preds.std():.1f}")



