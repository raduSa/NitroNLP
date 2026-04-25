from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import lightgbm as lgb

USE_LM    = os.environ.get("USE_LM",    "1") == "1"
USE_BERT  = os.environ.get("USE_BERT",  "1") == "1"
USE_TWO_STAGE = os.environ.get("USE_TWO_STAGE", "1") == "1"
USE_ROLLING   = os.environ.get("USE_ROLLING",   "1") == "1"

LM_CANDIDATES = os.environ.get(
    "LM_CANDIDATES",
    "readerbench/RoGPT2-large,readerbench/RoGPT2-medium,readerbench/RoGPT2-base,"
    "dumitrescustefan/gpt-neo-romanian-780m,dumitrescustefan/gpt-neo-romanian-125m"
).split(",")
LM_MAX_LEN = int(os.environ.get("LM_MAX_LEN", "1024"))
BERT_NAME  = os.environ.get("BERT_NAME", "dumitrescustefan/bert-base-romanian-cased-v1")
BERT_MAX_LEN = int(os.environ.get("BERT_MAX_LEN", "512"))
BERT_BATCH   = int(os.environ.get("BERT_BATCH",   "32"))
BERT_PCA_DIM = int(os.environ.get("BERT_PCA_DIM", "24"))
N_FOLDS  = int(os.environ.get("N_FOLDS",  "5"))
N_SEEDS  = int(os.environ.get("N_SEEDS",  "3"))
SEED = 42
ROLLING_WINDOWS       = [int(x) for x in os.environ.get("ROLLING_WINDOWS", "10,20,50").split(",")]
ROLLING_SKIP_THRESHOLD = float(os.environ.get("ROLLING_SKIP_THRESHOLD", "50.0"))

_KAGGLE_INPUT = Path("/kaggle/input")
if _KAGGLE_INPUT.exists():
    found = next(_KAGGLE_INPUT.rglob("train_data.csv"), None)
    if found is None:
        raise FileNotFoundError(f"train_data.csv not found under {_KAGGLE_INPUT}")
    INPUT_DIR = found.parent
    if not (INPUT_DIR / "test_data.csv").exists():
        raise FileNotFoundError(f"test_data.csv not found in {INPUT_DIR}")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    INPUT_DIR = OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[cfg] INPUT_DIR={INPUT_DIR}  USE_LM={USE_LM}  USE_BERT={USE_BERT}  "
      f"USE_TWO_STAGE={USE_TWO_STAGE}  USE_ROLLING={USE_ROLLING}  N_SEEDS={N_SEEDS}")


def comp_metric(y_true, preds):
    y_true = np.asarray(y_true, dtype=float); preds = np.asarray(preds, dtype=float)
    if np.std(preds) < 1e-12: return 0.0
    r2 = max(0.0, r2_score(y_true, preds, force_finite=True))
    pears = pearsonr(y_true, preds)[0]
    return 100.0 * (abs(pears if not np.isnan(pears) else 0.0) + r2) / 2.0


VOWELS   = set("aeiouăâîAEIOUĂÂÎ")
PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)
URL_RE   = re.compile(r"https?://|www\.", re.IGNORECASE)
NUM_RE   = re.compile(r"\d")
WORDID_RE = re.compile(r"^(?P<text>.+)_(?P<doc>\d+)_page_(?P<page>\d+)_(?P<idx>\d+)$")

def _count_syllables(token):
    s = token.lower(); n, in_v = 0, False
    for ch in s:
        v = ch in VOWELS
        if v and not in_v: n += 1
        in_v = v
    return max(n, 1) if any(c.isalpha() for c in token) else 0

def _strip_punct(tok):
    return re.sub(r"[^\w]", "", tok, flags=re.UNICODE)

def _parse_word_id(word_id):
    m = WORDID_RE.match(word_id)
    if not m: return 0, 0, 0
    return int(m["doc"]), int(m["page"]), int(m["idx"])

def build_word_features(df):
    df = df.copy(); df["word"] = df["word"].astype(str)
    stripped = df["word"].map(_strip_punct)
    df["word_len"]      = df["word"].str.len()
    df["alpha_len"]     = stripped.str.len()
    df["n_syllables"]   = df["word"].map(_count_syllables)
    df["is_punct"]      = df["word"].map(lambda w: bool(PUNCT_RE.match(w))).astype(int)
    df["is_url"]        = df["word"].map(lambda w: bool(URL_RE.search(w))).astype(int)
    df["has_digit"]     = df["word"].map(lambda w: bool(NUM_RE.search(w))).astype(int)
    df["is_upper_first"]= df["word"].map(lambda w: int(bool(w[:1].isupper())))
    df["is_all_upper"]  = df["word"].map(lambda w: int(len(w) > 1 and w.isupper()))
    df["ends_with_punct"]= df["word"].map(lambda w: int(len(w) > 0 and not w[-1].isalnum()))
    parsed = df["word_id"].astype(str).map(_parse_word_id).tolist()
    df["doc_num"]  = [p[0] for p in parsed]
    df["page_num"] = [p[1] for p in parsed]
    df["word_idx"] = [p[2] for p in parsed]
    return df

def add_context_features(df):
    df = df.sort_values(["text","participant_id","doc_num","page_num","word_idx"]).reset_index(drop=True)
    grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
    for col in ["word_len","alpha_len","n_syllables","is_punct","has_digit"]:
        df[f"prev_{col}"] = grp[col].shift(1).fillna(0)
        df[f"next_{col}"] = grp[col].shift(-1).fillna(0)
    df["pos_in_page"]     = grp.cumcount()
    df["page_size"]       = grp["word_idx"].transform("size")
    df["rel_pos_in_page"] = df["pos_in_page"] / df["page_size"].clip(lower=1)
    return df

def add_corpus_frequency(train, test):
    all_words = pd.concat([train["word"], test["word"]], axis=0).astype(str)
    norm = all_words.map(lambda w: _strip_punct(w).lower())
    counts = norm.value_counts(); total = counts.sum()
    log_freq = np.log((counts + 1) / (total + len(counts)))
    fallback = float(log_freq.min())
    for df, col in [(train, train["word"].astype(str)), (test, test["word"].astype(str))]:
        nrm = col.map(lambda w: _strip_punct(w).lower())
        df["log_freq"] = nrm.map(log_freq).fillna(fallback)
        grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
        df["prev_log_freq"] = grp["log_freq"].shift(1).fillna(fallback)
        df["next_log_freq"] = grp["log_freq"].shift(-1).fillna(fallback)
    return train, test

def add_wordfreq_features(train, test):
    try:
        from wordfreq import zipf_frequency
    except Exception as e:
        print(f"[wordfreq] unavailable: {e}"); return train, test, False
    def _zipf(w):
        s = _strip_punct(w).lower()
        return float(zipf_frequency(s, "ro", wordlist="best")) if s else 0.0
    for df in (train, test):
        df["zipf_freq"] = df["word"].astype(str).map(_zipf)
        grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
        df["prev_zipf_freq"] = grp["zipf_freq"].shift(1).fillna(0.0)
        df["next_zipf_freq"] = grp["zipf_freq"].shift(-1).fillna(0.0)
    return train, test, True

def add_per_word_target_stats(train, test):
    stat_cols = ["tok_mean","tok_median","tok_std","tok_count","tok_skip_rate"]
    def _agg(df):
        return df.groupby("tok_lc")["answer"].agg(
            tok_mean="mean", tok_median="median", tok_std="std", tok_count="count",
            tok_skip_rate=lambda s: float((s == 0).mean()))
    train = train.copy(); test = test.copy()
    train["tok_lc"] = train["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test["tok_lc"]  = test["word"].astype(str).map(lambda w: _strip_punct(w).lower())
    test = test.merge(_agg(train), left_on="tok_lc", right_index=True, how="left")
    for c in stat_cols: train[c] = np.nan
    for t in train["text"].unique():
        a = _agg(train.loc[train["text"] != t]); rows = train["text"] == t
        merged = train.loc[rows, ["tok_lc"]].merge(a, left_on="tok_lc", right_index=True, how="left")
        for c in stat_cols: train.loc[rows, c] = merged[c].values
    gm = float(train["answer"].mean())
    for c in ("tok_mean","tok_median"): train[c] = train[c].fillna(gm); test[c] = test[c].fillna(gm)
    train["tok_std"] = train["tok_std"].fillna(0.0);  test["tok_std"] = test["tok_std"].fillna(0.0)
    train["tok_skip_rate"] = train["tok_skip_rate"].fillna(0.3)
    test["tok_skip_rate"]  = test["tok_skip_rate"].fillna(0.3)
    train["tok_count"] = train["tok_count"].fillna(0).astype(float)
    test["tok_count"]  = test["tok_count"].fillna(0).astype(float)
    return train.drop(columns=["tok_lc"]), test.drop(columns=["tok_lc"])

def add_genre_features(train, test):
    train = train.copy(); test = test.copy()
    train["_genre"] = train["text"].str.extract(r"^([a-z]+)_", expand=False)
    test["_genre"]  = test["text"].str.extract(r"^([a-z]+)_", expand=False)
    all_genres = sorted(set(train["_genre"].dropna()) | set(test["_genre"].dropna()))
    g2id = {g: i for i, g in enumerate(all_genres)}
    train["genre_id"] = train["_genre"].map(g2id).fillna(-1).astype(int)
    test["genre_id"]  = test["_genre"].map(g2id).fillna(-1).astype(int)
    gm = float(train["answer"].mean())
    genre_means = train.groupby("_genre")["answer"].mean()
    test["genre_enc"] = test["_genre"].map(genre_means).fillna(gm)
    train["genre_enc"] = np.nan
    for text in train["text"].unique():
        genre = train.loc[train["text"] == text, "_genre"].iloc[0]
        other = train.loc[(train["_genre"] == genre) & (train["text"] != text), "answer"].mean()
        train.loc[train["text"] == text, "genre_enc"] = other if pd.notna(other) else gm
    train["genre_enc"] = train["genre_enc"].fillna(gm)
    return train.drop(columns=["_genre"]), test.drop(columns=["_genre"])

def augment_participant_mixup(train, multiplier=1.0, add_average_reader=True, seed=SEED):
    rng = np.random.default_rng(seed)
    interp_cols = [c for c in ("answer","participant_enc","doc_num","surprisal") if c in train.columns]
    synth_parts, avg_parts = [], []
    for _, grp in train.groupby(["text","page_num","word_idx"], sort=False):
        n = len(grp)
        if n < 2: continue
        vals = grp[interp_cols].to_numpy(dtype=float)
        n_pairs = round(n * multiplier)
        idx_a = rng.integers(0, n, n_pairs); idx_b = rng.integers(0, n, n_pairs)
        idx_b[idx_a == idx_b] = (idx_b[idx_a == idx_b] + 1) % n
        alpha = rng.uniform(0.0, 1.0, (n_pairs, 1))
        synth_vals = alpha * vals[idx_a] + (1.0 - alpha) * vals[idx_b]
        synth_rows = grp.iloc[idx_a].reset_index(drop=True).copy()
        for i, col in enumerate(interp_cols): synth_rows[col] = synth_vals[:, i]
        synth_parts.append(synth_rows)
        if add_average_reader:
            avg_row = grp.iloc[[0]].copy()
            for i, col in enumerate(interp_cols): avg_row[col] = float(vals[:, i].mean())
            avg_parts.append(avg_row)
    pieces = [train]
    if synth_parts: pieces.append(pd.concat(synth_parts, ignore_index=True))
    if avg_parts:   pieces.append(pd.concat(avg_parts,   ignore_index=True))
    augmented = pd.concat(pieces, ignore_index=True)
    n_added = len(augmented) - len(train); avg_added = sum(len(p) for p in avg_parts)
    print(f"[aug] mixup: +{n_added:,} rows ({n_added-avg_added:,} mixup + {avg_added:,} avg-reader)")
    return augmented

def augment_genre_average_reader(train):
    train = train.copy()
    train["_genre"]  = train["text"].str.extract(r"^([a-z]+)_", expand=False)
    train["_tok_lc"] = train["word"].astype(str).str.lower().str.strip()
    has_pe = "participant_enc" in train.columns
    gpe = float(train["participant_enc"].mean()) if has_pe else None
    avg_rows = []
    for (genre, tok), grp in train.groupby(["_genre","_tok_lc"], sort=False):
        if len(grp) < 2: continue
        avg_row = grp.iloc[[0]].copy()
        avg_row["answer"] = float(grp["answer"].mean())
        if has_pe: avg_row["participant_enc"] = gpe
        avg_rows.append(avg_row)
    augmented = pd.concat([train] + avg_rows, ignore_index=True).drop(columns=["_genre","_tok_lc"])
    print(f"[aug] genre-avg-reader: +{len(augmented)-len(train):,} rows")
    return augmented

def compute_rolling_features(df, pred, windows=ROLLING_WINDOWS, skip_threshold=ROLLING_SKIP_THRESHOLD):
    df = df.copy(); df["__pred"] = pred; df["__pos"] = np.arange(len(df))
    df_s = df.sort_values(["participant_id","doc_num","page_num","word_idx"])
    df_s["session_word_idx"] = df_s.groupby("participant_id").cumcount()
    df_s["pass1_pred"] = df_s["__pred"]
    new_cols = ["session_word_idx","pass1_pred"]
    gm = float(pred.mean())
    for k in windows:
        grp = df_s.groupby("participant_id")["__pred"]
        df_s[f"roll_mean_{k}"] = grp.transform(
            lambda s: s.shift(1).rolling(k, min_periods=1).mean()).fillna(gm)
        df_s[f"roll_skip_{k}"] = grp.transform(
            lambda s: (s.shift(1) < skip_threshold).astype(float).rolling(k, min_periods=1).mean()
        ).fillna(0.3)
        new_cols += [f"roll_mean_{k}", f"roll_skip_{k}"]
    df_s = df_s.sort_values("__pos").drop(columns=["__pred","__pos"])
    return df_s, new_cols

def _try_load_causal_lm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in LM_CANDIDATES:
        name = name.strip()
        if not name: continue
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
            if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token
            print(f"[lm] loaded {name} on {device}"); return tok, model, name
        except Exception as e:
            print(f"[lm] {name} failed: {e}")
    return None, None, None

def compute_causal_surprisal(df, tokenizer, model):
    import torch
    device = next(model.parameters()).device
    out = np.zeros(len(df), dtype=np.float32)
    df = df.reset_index().rename(columns={"index":"_orig_idx"}).copy()
    df = df.sort_values(["text","participant_id","doc_num","page_num","word_idx"])
    @torch.no_grad()
    def score_ids(full_ids):
        L = len(full_ids)
        if L < 2: return np.zeros(L, dtype=np.float32)
        logp = np.zeros(L, dtype=np.float32)
        if L <= LM_MAX_LEN:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            lp = torch.log_softmax(model(ids).logits[0], dim=-1)
            logp[1:] = (-lp[:-1].gather(-1, ids[0,1:].unsqueeze(-1)).squeeze(-1)).cpu().numpy()
            return logp
        stride = LM_MAX_LEN // 2; first = True; pos = 0
        while pos < L:
            end = min(pos + LM_MAX_LEN, L)
            ids = torch.tensor([full_ids[pos:end]], dtype=torch.long, device=device)
            lp = torch.log_softmax(model(ids).logits[0], dim=-1)
            tgt = ids[0, 1:]
            pred = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            fill_lo = pos+1 if first else pos+stride; fill_hi = end
            if fill_lo < fill_hi:
                logp[fill_lo:fill_hi] = (-pred[fill_lo-pos-1:fill_hi-pos-1]).cpu().numpy()
            first = False
            if end >= L: break
            pos += stride
        return logp
    use_offsets = tokenizer.is_fast; bos = tokenizer.bos_token_id
    groups = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
    n_groups = len(groups)
    for gi, (_, block) in enumerate(groups):
        words = block["word"].astype(str).tolist()
        buf, spans, cur = [], [], 0
        for i, w in enumerate(words):
            if i > 0: buf.append(" "); cur += 1
            buf.append(w); spans.append((cur, cur + len(w))); cur += len(w)
        text = "".join(buf)
        if use_offsets:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            sub_ids = enc["input_ids"]; offsets = enc["offset_mapping"]
        else:
            sub_ids = tokenizer.encode(text, add_special_tokens=False); offsets = None
        full_ids = ([bos] if bos is not None else []) + list(sub_ids)
        shift = 1 if bos is not None else 0
        sub_surps = score_ids(full_ids)[shift:]
        word_surps = np.zeros(len(words), dtype=np.float32)
        if offsets is not None:
            wi = 0
            for k, (s, e) in enumerate(offsets):
                if s == e: continue
                while wi < len(words) and spans[wi][1] <= s: wi += 1
                if wi < len(words) and spans[wi][0] <= s < spans[wi][1]: word_surps[wi] += sub_surps[k]
        else:
            cur_id = 0
            for wi, w in enumerate(words):
                ids_w = tokenizer.encode((" " if wi > 0 else "") + w, add_special_tokens=False)
                if cur_id + len(ids_w) <= len(sub_surps): word_surps[wi] = sub_surps[cur_id:cur_id+len(ids_w)].sum()
                cur_id += len(ids_w)
        out[block["_orig_idx"].to_numpy()] = word_surps
        if (gi+1) % 100 == 0 or (gi+1) == n_groups: print(f"[causal-lm] {gi+1}/{n_groups}")
    return out

def _word_token_spans(tokenizer, words):
    buf, spans, cur = [], [], 0
    for i, w in enumerate(words):
        if i > 0: buf.append(" "); cur += 1
        buf.append(w); spans.append((cur, cur+len(w))); cur += len(w)
    enc = tokenizer("".join(buf), add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
    sub_ids = enc["input_ids"]; offsets = enc["offset_mapping"]
    per_word = [[] for _ in range(len(words))]; wi = 0
    for k, (s, e) in enumerate(offsets):
        if s == e: continue
        while wi < len(words) and spans[wi][1] <= s: wi += 1
        if wi < len(words) and spans[wi][0] <= s < spans[wi][1]: per_word[wi].append(k)
    return sub_ids, per_word

def compute_bert_features(df, tokenizer, model, mask_token_id):
    import torch
    device = next(model.parameters()).device; hidden = model.config.hidden_size
    n_subw_arr = np.zeros(len(df), dtype=np.float32)
    mlm_first_arr = np.zeros(len(df), dtype=np.float32)
    mlm_sum_arr   = np.zeros(len(df), dtype=np.float32)
    emb_arr = np.zeros((len(df), hidden), dtype=np.float32)
    df = df.reset_index().rename(columns={"index":"_orig_idx"}).copy()
    df = df.sort_values(["text","participant_id","doc_num","page_num","word_idx"])
    groups = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
    n_groups = len(groups)
    cls_id = tokenizer.cls_token_id; sep_id = tokenizer.sep_token_id
    @torch.no_grad()
    def run_chunk(chunk_sub_ids, chunk_per_word):
        n_sub = len(chunk_sub_ids); n_w = len(chunk_per_word)
        mlm_first = [0.0]*n_w; mlm_sum = [0.0]*n_w
        if n_sub == 0: return np.zeros((0, hidden), dtype=np.float32), mlm_first, mlm_sum
        ids = torch.tensor([[cls_id]+chunk_sub_ids+[sep_id]], dtype=torch.long, device=device)
        out_emb = model(ids, output_hidden_states=True)
        sub_embs = torch.stack(out_emb.hidden_states[-4:], dim=0).mean(0)[0][1:-1].cpu().numpy()
        variants, variant_word = [], []
        for lw, spl in enumerate(chunk_per_word):
            if not spl: continue
            v = list(chunk_sub_ids)
            for p in spl: v[p] = mask_token_id
            variants.append(v); variant_word.append(lw)
        if not variants: return sub_embs, mlm_first, mlm_sum
        for bs in range(0, len(variants), BERT_BATCH):
            batch = variants[bs:bs+BERT_BATCH]; wids = variant_word[bs:bs+BERT_BATCH]
            ids_t = torch.tensor([[cls_id]+v+[sep_id] for v in batch], dtype=torch.long, device=device)
            lp = torch.log_softmax(model(ids_t).logits, dim=-1)
            for bi, lw in enumerate(wids):
                spl = chunk_per_word[lw]
                mlm_first[lw] = float(-lp[bi, spl[0]+1, chunk_sub_ids[spl[0]]].item())
                mlm_sum[lw]   = sum(float(-lp[bi, p+1, chunk_sub_ids[p]].item()) for p in spl)
        return sub_embs, mlm_first, mlm_sum
    for gi, (_, block) in enumerate(groups):
        words = block["word"].astype(str).tolist()
        try: sub_ids, per_word_idx = _word_token_spans(tokenizer, words)
        except Exception as e: print(f"[bert] group {gi} failed: {e}"); continue
        n_words = len(words); capacity = BERT_MAX_LEN - 2
        chunks = []; wi = 0
        while wi < n_words:
            sc = 0; wj = wi
            while wj < n_words:
                ws = len(per_word_idx[wj])
                if sc + ws > capacity: break
                sc += ws; wj += 1
            if wj == wi: wj = wi + 1
            chunk_sub_ids = [sub_ids[idx] for k in range(wi, wj) for idx in per_word_idx[k]]
            chunks.append((wi, wj, chunk_sub_ids)); wi = wj
        for (wlo, whi, csi) in chunks:
            if not csi: continue
            cpw = []; cur = 0
            for k in range(wlo, whi):
                ws = len(per_word_idx[k]); cpw.append(list(range(cur, cur+ws))); cur += ws
            sub_embs, mlm_first, mlm_sum = run_chunk(csi, cpw)
            for lw, spl in enumerate(cpw):
                ri = block["_orig_idx"].iloc[wlo+lw]
                if spl:
                    emb_arr[ri] = sub_embs[spl].mean(axis=0)
                    n_subw_arr[ri] = len(spl); mlm_first_arr[ri] = mlm_first[lw]; mlm_sum_arr[ri] = mlm_sum[lw]
        if (gi+1) % 50 == 0 or (gi+1) == n_groups: print(f"[bert] {gi+1}/{n_groups}")
    return {"n_bert_subwords": n_subw_arr, "mlm_logp_first": mlm_first_arr,
            "mlm_logp_sum": mlm_sum_arr, "bert_emb": emb_arr}

_LGB_BASE = dict(learning_rate=0.05, num_leaves=127, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbosity=-1)
def lgb_params_reg(seed): return {**_LGB_BASE, "objective":"tweedie","tweedie_variance_power":1.4,"metric":"rmse","min_data_in_leaf":200,"seed":seed}
def lgb_params_clf(seed): return {**_LGB_BASE, "objective":"binary","metric":"binary_logloss","min_data_in_leaf":200,"seed":seed}
def lgb_params_pos(seed): return {**_LGB_BASE, "objective":"regression","metric":"rmse","min_data_in_leaf":100,"seed":seed}

def fit_calibration(y_true, y_pred):
    if np.std(y_pred) < 1e-9: return 1.0, 0.0
    a = np.cov(y_pred, y_true, ddof=0)[0,1] / np.var(y_pred)
    return float(a), float(y_true.mean() - a * y_pred.mean())

def optimize_blend(y_true, p1, p2):
    best_w, best_s = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 51):
        s = comp_metric(y_true, w*p1 + (1-w)*p2)
        if s > best_s: best_s, best_w = s, float(w)
    return best_w


def main():
    train = pd.read_csv(INPUT_DIR / "train_data.csv")
    test  = pd.read_csv(INPUT_DIR / "test_data.csv")
    print(f"[data] train={train.shape}  test={test.shape}")
    n_test_orig   = len(test)
    test_ids_orig = set(test["datapointID"].astype(int).tolist())
    assert len(test_ids_orig) == n_test_orig

    train = build_word_features(train);  test = build_word_features(test)
    train = add_context_features(train); test = add_context_features(test)
    train, test = add_corpus_frequency(train, test)
    train, test, has_zipf = add_wordfreq_features(train, test)
    train, test = add_per_word_target_stats(train, test)
    train, test = add_genre_features(train, test)

    feature_cols = [
        "word_len","alpha_len","n_syllables","is_punct","is_url","has_digit",
        "is_upper_first","is_all_upper","ends_with_punct",
        "doc_num","page_num","word_idx","pos_in_page","rel_pos_in_page","page_size",
        "log_freq","prev_log_freq","next_log_freq",
        "tok_mean","tok_median","tok_std","tok_count","tok_skip_rate",
        "prev_word_len","prev_alpha_len","prev_n_syllables","prev_is_punct","prev_has_digit",
        "next_word_len","next_alpha_len","next_n_syllables","next_is_punct","next_has_digit",
        "genre_id","genre_enc",
    ]
    if has_zipf: feature_cols += ["zipf_freq","prev_zipf_freq","next_zipf_freq"]

    if USE_LM:
        try:
            tokenizer, lm_model, lm_name = _try_load_causal_lm()
            if lm_model is not None:
                print(f"[causal-lm] scoring train+test ({lm_name})")
                train["surprisal"] = compute_causal_surprisal(train, tokenizer, lm_model)
                test["surprisal"]  = compute_causal_surprisal(test,  tokenizer, lm_model)
                for df in (train, test):
                    grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
                    df["prev_surprisal"] = grp["surprisal"].shift(1).fillna(0.0)
                    df["next_surprisal"] = grp["surprisal"].shift(-1).fillna(0.0)
                feature_cols += ["surprisal","prev_surprisal","next_surprisal"]
                del lm_model, tokenizer
            else: print("[causal-lm] no model loaded")
        except Exception as e: print(f"[causal-lm] failed: {e}")
    else: print("[causal-lm] disabled")

    if USE_BERT:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[bert] loading {BERT_NAME} on {device}")
            btok   = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)
            bmodel = AutoModelForMaskedLM.from_pretrained(BERT_NAME).to(device).eval()
            mask_id = btok.mask_token_id
            cols_needed = ["word_id","word","text","doc_num","page_num","word_idx"]
            uniq = (pd.concat([train[cols_needed], test[cols_needed]], axis=0)
                    .drop_duplicates(subset=["word_id"]).reset_index(drop=True))
            uniq["participant_id"] = 0
            print(f"[bert] {len(uniq)} unique word_ids")
            ubf = compute_bert_features(uniq, btok, bmodel, mask_id)
            uniq_feat = pd.DataFrame({"word_id": uniq["word_id"].values,
                "n_bert_subwords": ubf["n_bert_subwords"],
                "mlm_logp_first":  ubf["mlm_logp_first"],
                "mlm_logp_sum":    ubf["mlm_logp_sum"]})
            id2pos = {wid: i for i, wid in enumerate(uniq["word_id"].values)}
            train = train.merge(uniq_feat, on="word_id", how="left")
            test  = test.merge(uniq_feat,  on="word_id", how="left")
            tr_emb = ubf["bert_emb"][train["word_id"].map(id2pos).values]
            te_emb = ubf["bert_emb"][test["word_id"].map(id2pos).values]
            for df in (train, test):
                grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
                df["prev_mlm_logp_first"] = grp["mlm_logp_first"].shift(1).fillna(0.0)
                df["next_mlm_logp_first"] = grp["mlm_logp_first"].shift(-1).fillna(0.0)
            feature_cols += ["n_bert_subwords","mlm_logp_first","mlm_logp_sum",
                             "prev_mlm_logp_first","next_mlm_logp_first"]
            valid = np.linalg.norm(tr_emb, axis=1) > 1e-6
            n_comp = min(BERT_PCA_DIM, tr_emb.shape[1], int(valid.sum())-1)
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=SEED)
                pca.fit(tr_emb[valid])
                tr_p = pca.transform(tr_emb); te_p = pca.transform(te_emb)
                emb_cols = [f"bert_pc_{i}" for i in range(n_comp)]
                for i, c in enumerate(emb_cols): train[c] = tr_p[:,i]; test[c] = te_p[:,i]
                feature_cols += emb_cols
                print(f"[bert] PCA var={pca.explained_variance_ratio_.sum():.3f}")
            del bmodel, btok
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[bert] failed: {e}")
    else: print("[bert] disabled")

    train["_is_orig"] = True
    train = augment_participant_mixup(train)
    train = augment_genre_average_reader(train)
    train["_is_orig"] = train["_is_orig"].fillna(False)

    for c in feature_cols:
        assert c in train.columns and c in test.columns, f"missing: {c}"
    nan_tr = int(train[feature_cols].isna().sum().sum())
    nan_te = int(test[feature_cols].isna().sum().sum())
    if nan_tr or nan_te:
        print(f"[feat] WARNING NaNs train={nan_tr} test={nan_te}; filling 0")
        train[feature_cols] = train[feature_cols].fillna(0.0)
        test[feature_cols]  = test[feature_cols].fillna(0.0)
    print(f"[feat] {len(feature_cols)} features")

    train = train.sort_values(["text","participant_id","doc_num","page_num","word_idx"]).reset_index(drop=True)
    orig_mask = train["_is_orig"].values.astype(bool)
    X = train[feature_cols].values; y = train["answer"].values.astype(float)
    groups = train["text"].values;  Xt = test[feature_cols].values

    n_splits = min(N_FOLDS, train["text"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))

    oof_a = np.zeros(len(train)); test_a = np.zeros(len(test)); fold_scores_a = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        s_oof = np.zeros(len(va_idx)); s_test = np.zeros(len(test))
        for si in range(N_SEEDS):
            seed = SEED + 1000*si + fi
            dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
            dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
            b = lgb.train(lgb_params_reg(seed), dtr, num_boost_round=4000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            s_oof += b.predict(X[va_idx], num_iteration=b.best_iteration)
            s_test += b.predict(Xt, num_iteration=b.best_iteration)
        s_oof /= N_SEEDS; s_test /= N_SEEDS
        oof_a[va_idx] = s_oof; test_a += s_test / n_splits
        sc = comp_metric(y[va_idx], s_oof); fold_scores_a.append(sc)
        print(f"[stage1 fold {fi}] {sorted(set(groups[va_idx]))}  score={sc:.3f}")
    print(f"[stage1] mean={np.mean(fold_scores_a):.3f}  OOF={comp_metric(y, oof_a):.3f}")

    if USE_TWO_STAGE:
        oof_b = np.zeros(len(train)); test_b = np.zeros(len(test)); fold_scores_b = []
        y_skip = (y == 0).astype(float)
        for fi, (tr_idx, va_idx) in enumerate(splits):
            sk_oof = np.zeros(len(va_idx)); sk_test = np.zeros(len(test))
            rg_oof = np.zeros(len(va_idx)); rg_test = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED + 1000*si + fi
                dtr_c = lgb.Dataset(X[tr_idx], y_skip[tr_idx])
                dva_c = lgb.Dataset(X[va_idx], y_skip[va_idx], reference=dtr_c)
                clf = lgb.train(lgb_params_clf(seed), dtr_c, num_boost_round=2000, valid_sets=[dva_c],
                                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                sk_oof += clf.predict(X[va_idx], num_iteration=clf.best_iteration)
                sk_test += clf.predict(Xt, num_iteration=clf.best_iteration)
                pos_tr = tr_idx[y[tr_idx] > 0]
                pos_va = va_idx[y[va_idx] > 0]
                if len(pos_va) < 10: pos_va = va_idx
                dtr_r = lgb.Dataset(X[pos_tr], np.log1p(y[pos_tr]))
                dva_r = lgb.Dataset(X[pos_va], np.log1p(y[pos_va]), reference=dtr_r)
                rg = lgb.train(lgb_params_pos(seed), dtr_r, num_boost_round=4000, valid_sets=[dva_r],
                               callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                rg_oof += np.expm1(rg.predict(X[va_idx], num_iteration=rg.best_iteration))
                rg_test += np.expm1(rg.predict(Xt, num_iteration=rg.best_iteration))
            sk_oof /= N_SEEDS; sk_test /= N_SEEDS; rg_oof /= N_SEEDS; rg_test /= N_SEEDS
            sk_oof = np.clip(sk_oof, 0, 1); sk_test = np.clip(sk_test, 0, 1)
            rg_oof = np.clip(rg_oof, 0, None); rg_test = np.clip(rg_test, 0, None)
            oof_b[va_idx] = (1-sk_oof)*rg_oof; test_b += ((1-sk_test)*rg_test)/n_splits
            sc = comp_metric(y[va_idx], oof_b[va_idx]); fold_scores_b.append(sc)
            print(f"[stage2 fold {fi}] {sorted(set(groups[va_idx]))}  score={sc:.3f}")
        print(f"[stage2] mean={np.mean(fold_scores_b):.3f}  OOF={comp_metric(y, oof_b):.3f}")
    else:
        oof_b = oof_a.copy(); test_b = test_a.copy()

    w = optimize_blend(y, oof_a, oof_b) if USE_TWO_STAGE else 1.0
    print(f"[blend] w_stage1={w:.3f}")
    oof_blend  = w*oof_a  + (1-w)*oof_b
    test_blend = w*test_a + (1-w)*test_b
    print(f"[blend] OOF={comp_metric(y, oof_blend):.3f}")

    if USE_ROLLING:
        train_orig = train[orig_mask].copy().reset_index(drop=True)
        oof_orig   = oof_blend[orig_mask]
        train_orig, roll_cols = compute_rolling_features(train_orig, oof_orig)
        test, _               = compute_rolling_features(test, test_blend)
        p2_fcols = feature_cols + roll_cols
        X2 = train_orig[p2_fcols].values; y2 = train_orig["answer"].values.astype(float)
        g2 = train_orig["text"].values;   Xt2 = test[p2_fcols].values
        n_splits2 = min(N_FOLDS, len(np.unique(g2)))
        gkf2 = GroupKFold(n_splits=n_splits2)
        oof_c = np.zeros(len(train_orig)); test_c = np.zeros(len(test)); fold_scores_c = []
        for fi, (tr2, va2) in enumerate(gkf2.split(X2, y2, g2)):
            s_oof = np.zeros(len(va2)); s_test = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED + 3000 + 1000*si + fi
                dtr2 = lgb.Dataset(X2[tr2], y2[tr2])
                dva2 = lgb.Dataset(X2[va2], y2[va2], reference=dtr2)
                b2 = lgb.train(lgb_params_reg(seed), dtr2, num_boost_round=4000, valid_sets=[dva2],
                               callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                s_oof  += b2.predict(X2[va2], num_iteration=b2.best_iteration)
                s_test += b2.predict(Xt2,     num_iteration=b2.best_iteration)
            s_oof /= N_SEEDS; s_test /= N_SEEDS
            oof_c[va2] = s_oof; test_c += s_test / n_splits2
            sc = comp_metric(y2[va2], s_oof); fold_scores_c.append(sc)
            print(f"[pass2 fold {fi}] {sorted(set(g2[va2]))}  score={sc:.3f}")
        print(f"[pass2] mean={np.mean(fold_scores_c):.3f}  OOF={comp_metric(y2, oof_c):.3f}")
        w2 = optimize_blend(y2, oof_orig, oof_c)
        print(f"[pass2 blend] w_pass1={w2:.3f}  w_pass2={1-w2:.3f}")
        oof_final  = w2*oof_orig   + (1-w2)*oof_c
        test_blend = w2*test_blend + (1-w2)*test_c
        print(f"[pass2 blend] OOF={comp_metric(y2, oof_final):.3f}")
        a, b = fit_calibration(y2, oof_final)
    else:
        a, b = fit_calibration(y, oof_blend)

    print(f"[calib] a={a:.4f}  b={b:.4f}")
    test_cal = np.clip(a*test_blend + b, 0, None)
    print(f"[pred] min={test_cal.min():.1f} mean={test_cal.mean():.1f} "
          f"max={test_cal.max():.1f} #zeros={(test_cal<1).sum()}")

    sub = pd.DataFrame({
        "subtaskID":   np.ones(len(test), dtype=int),
        "datapointID": test["datapointID"].astype(int).values,
        "answer":      np.round(test_cal).astype(int),
    }).sort_values("datapointID").reset_index(drop=True)

    assert list(sub.columns) == ["subtaskID","datapointID","answer"]
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
    print("[out] all checks passed.")


if __name__ == "__main__":
    main()
