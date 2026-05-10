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
LM_CANDIDATES = os.environ.get("LM_CANDIDATES",
    "readerbench/RoGPT2-large,readerbench/RoGPT2-medium,readerbench/RoGPT2-base,"
    "dumitrescustefan/gpt-neo-romanian-780m,dumitrescustefan/gpt-neo-romanian-125m").split(",")
LM_MAX_LEN = int(os.environ.get("LM_MAX_LEN", "1024"))
BERT_NAME  = os.environ.get("BERT_NAME", "dumitrescustefan/bert-base-romanian-cased-v1")
BERT_MAX_LEN = int(os.environ.get("BERT_MAX_LEN", "512"))
BERT_BATCH   = int(os.environ.get("BERT_BATCH",   "32"))
BERT_PCA_DIM = int(os.environ.get("BERT_PCA_DIM", "24"))
N_FOLDS  = int(os.environ.get("N_FOLDS",  "5"))
N_SEEDS  = int(os.environ.get("N_SEEDS",  "3"))
SEED = 42
ROLLING_WINDOWS        = [int(x) for x in os.environ.get("ROLLING_WINDOWS", "10,20,50").split(",")]
ROLLING_SKIP_THRESHOLD = float(os.environ.get("ROLLING_SKIP_THRESHOLD", "50.0"))

_KAGGLE = Path("/kaggle/input")
if _KAGGLE.exists():
    found = next(_KAGGLE.rglob("train_data.csv"), None)
    if not found: raise FileNotFoundError(f"train_data.csv not found under {_KAGGLE}")
    INPUT_DIR = found.parent
    if not (INPUT_DIR / "test_data.csv").exists(): raise FileNotFoundError("test_data.csv missing")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    INPUT_DIR = OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[cfg] USE_LM={USE_LM} USE_BERT={USE_BERT} USE_TWO_STAGE={USE_TWO_STAGE} "
      f"USE_ROLLING={USE_ROLLING} N_SEEDS={N_SEEDS}")


def comp_metric(y_true, preds):
    y_true = np.asarray(y_true, dtype=float); preds = np.asarray(preds, dtype=float)
    if np.std(preds) < 1e-12: return 0.0
    r2 = max(0.0, r2_score(y_true, preds, force_finite=True))
    p = pearsonr(y_true, preds)[0]
    return 100.0 * (abs(p if not np.isnan(p) else 0.0) + r2) / 2.0


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

def _strip_punct(tok): return re.sub(r"[^\w]", "", tok, flags=re.UNICODE)

def _parse_word_id(wid):
    m = WORDID_RE.match(wid)
    return (int(m["doc"]), int(m["page"]), int(m["idx"])) if m else (0, 0, 0)

def build_word_features(df):
    df = df.copy(); df["word"] = df["word"].astype(str)
    sp = df["word"].map(_strip_punct)
    df["word_len"]       = df["word"].str.len()
    df["alpha_len"]      = sp.str.len()
    df["n_syllables"]    = df["word"].map(_count_syllables)
    df["is_punct"]       = df["word"].map(lambda w: bool(PUNCT_RE.match(w))).astype(int)
    df["is_url"]         = df["word"].map(lambda w: bool(URL_RE.search(w))).astype(int)
    df["has_digit"]      = df["word"].map(lambda w: bool(NUM_RE.search(w))).astype(int)
    df["is_upper_first"] = df["word"].map(lambda w: int(bool(w[:1].isupper())))
    df["is_all_upper"]   = df["word"].map(lambda w: int(len(w) > 1 and w.isupper()))
    df["ends_with_punct"]= df["word"].map(lambda w: int(len(w) > 0 and not w[-1].isalnum()))
    p = df["word_id"].astype(str).map(_parse_word_id).tolist()
    df["doc_num"] = [x[0] for x in p]; df["page_num"] = [x[1] for x in p]; df["word_idx"] = [x[2] for x in p]
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
    norm = pd.concat([train["word"], test["word"]], axis=0).astype(str).map(lambda w: _strip_punct(w).lower())
    counts = norm.value_counts(); total = counts.sum()
    lf = np.log((counts + 1) / (total + len(counts))); fb = float(lf.min())
    for df in (train, test):
        df["log_freq"] = df["word"].astype(str).map(lambda w: _strip_punct(w).lower()).map(lf).fillna(fb)
        grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
        df["prev_log_freq"] = grp["log_freq"].shift(1).fillna(fb)
        df["next_log_freq"] = grp["log_freq"].shift(-1).fillna(fb)
    return train, test

def add_wordfreq_features(train, test):
    try: from wordfreq import zipf_frequency
    except Exception as e: print(f"[wordfreq] {e}"); return train, test, False
    def _z(w):
        s = _strip_punct(w).lower()
        return float(zipf_frequency(s, "ro", wordlist="best")) if s else 0.0
    for df in (train, test):
        df["zipf_freq"] = df["word"].astype(str).map(_z)
        grp = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
        df["prev_zipf_freq"] = grp["zipf_freq"].shift(1).fillna(0.0)
        df["next_zipf_freq"] = grp["zipf_freq"].shift(-1).fillna(0.0)
    return train, test, True

def add_per_word_target_stats(train, test):
    sc = ["tok_mean","tok_median","tok_std","tok_count","tok_skip_rate"]
    def _agg(df):
        return df.groupby("tok_lc")["answer"].agg(
            tok_mean="mean", tok_median="median", tok_std="std", tok_count="count",
            tok_skip_rate=lambda s: float((s == 0).mean()))
    train = train.copy(); test = test.copy()
    lc = lambda w: _strip_punct(w).lower()
    train["tok_lc"] = train["word"].astype(str).map(lc)
    test["tok_lc"]  = test["word"].astype(str).map(lc)
    test = test.merge(_agg(train), left_on="tok_lc", right_index=True, how="left")
    for c in sc: train[c] = np.nan
    for t in train["text"].unique():
        a = _agg(train.loc[train["text"] != t]); rows = train["text"] == t
        mg = train.loc[rows, ["tok_lc"]].merge(a, left_on="tok_lc", right_index=True, how="left")
        for c in sc: train.loc[rows, c] = mg[c].values
    gm = float(train["answer"].mean())
    for c in ("tok_mean","tok_median"): train[c] = train[c].fillna(gm); test[c] = test[c].fillna(gm)
    train["tok_std"] = train["tok_std"].fillna(0.0);       test["tok_std"] = test["tok_std"].fillna(0.0)
    train["tok_skip_rate"] = train["tok_skip_rate"].fillna(0.3)
    test["tok_skip_rate"]  = test["tok_skip_rate"].fillna(0.3)
    train["tok_count"] = train["tok_count"].fillna(0).astype(float)
    test["tok_count"]  = test["tok_count"].fillna(0).astype(float)
    return train.drop(columns=["tok_lc"]), test.drop(columns=["tok_lc"])

def add_genre_features(train, test):
    train = train.copy(); test = test.copy()
    train["_g"] = train["text"].str.extract(r"^([a-z]+)_", expand=False)
    test["_g"]  = test["text"].str.extract(r"^([a-z]+)_", expand=False)
    g2id = {g: i for i, g in enumerate(sorted(set(train["_g"].dropna()) | set(test["_g"].dropna())))}
    train["genre_id"] = train["_g"].map(g2id).fillna(-1).astype(int)
    test["genre_id"]  = test["_g"].map(g2id).fillna(-1).astype(int)
    gm = float(train["answer"].mean()); gmu = train.groupby("_g")["answer"].mean()
    test["genre_enc"] = test["_g"].map(gmu).fillna(gm)
    train["genre_enc"] = np.nan
    for t in train["text"].unique():
        g = train.loc[train["text"] == t, "_g"].iloc[0]
        oth = train.loc[(train["_g"] == g) & (train["text"] != t), "answer"].mean()
        train.loc[train["text"] == t, "genre_enc"] = oth if pd.notna(oth) else gm
    train["genre_enc"] = train["genre_enc"].fillna(gm)
    return train.drop(columns=["_g"]), test.drop(columns=["_g"])

def augment_participant_mixup(train, multiplier=1.0, add_average_reader=True, seed=SEED):
    rng = np.random.default_rng(seed)
    ic = [c for c in ("answer","participant_enc","doc_num","surprisal") if c in train.columns]
    sp, ap = [], []
    for _, grp in train.groupby(["text","page_num","word_idx"], sort=False):
        n = len(grp)
        if n < 2: continue
        vals = grp[ic].to_numpy(dtype=float); np_ = round(n * multiplier)
        ia = rng.integers(0, n, np_); ib = rng.integers(0, n, np_)
        ib[ia == ib] = (ib[ia == ib] + 1) % n
        sv = rng.uniform(0.0, 1.0, (np_, 1)) * vals[ia] + (1 - rng.uniform(0.0, 1.0, (np_, 1))) * vals[ib]
        sr = grp.iloc[ia].reset_index(drop=True).copy()
        for i, c in enumerate(ic): sr[c] = sv[:, i]
        sp.append(sr)
        if add_average_reader:
            ar = grp.iloc[[0]].copy()
            for i, c in enumerate(ic): ar[c] = float(vals[:, i].mean())
            ap.append(ar)
    pcs = [train]
    if sp: pcs.append(pd.concat(sp, ignore_index=True))
    if ap: pcs.append(pd.concat(ap, ignore_index=True))
    aug = pd.concat(pcs, ignore_index=True)
    na = len(aug) - len(train); aa = sum(len(p) for p in ap)
    print(f"[aug] mixup: +{na:,} ({na-aa:,} mixup + {aa:,} avg-reader)"); return aug

def augment_genre_average_reader(train):
    train = train.copy()
    train["_g"]  = train["text"].str.extract(r"^([a-z]+)_", expand=False)
    train["_lc"] = train["word"].astype(str).str.lower().str.strip()
    hp = "participant_enc" in train.columns
    gp = float(train["participant_enc"].mean()) if hp else None
    rows = []
    for (g, t), grp in train.groupby(["_g","_lc"], sort=False):
        if len(grp) < 2: continue
        r = grp.iloc[[0]].copy(); r["answer"] = float(grp["answer"].mean())
        if hp: r["participant_enc"] = gp
        rows.append(r)
    aug = pd.concat([train] + rows, ignore_index=True).drop(columns=["_g","_lc"])
    print(f"[aug] genre-avg: +{len(aug)-len(train):,}"); return aug

def compute_rolling_features(df, pred, windows=ROLLING_WINDOWS, skip_thr=ROLLING_SKIP_THRESHOLD):
    df = df.copy(); df["__p"] = pred; df["__i"] = np.arange(len(df))
    ds = df.sort_values(["participant_id","doc_num","page_num","word_idx"])
    ds["session_word_idx"] = ds.groupby("participant_id").cumcount()
    ds["pass1_pred"] = ds["__p"]; nc = ["session_word_idx","pass1_pred"]; gm = float(pred.mean())
    for k in windows:
        g = ds.groupby("participant_id")["__p"]
        ds[f"roll_mean_{k}"] = g.transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean()).fillna(gm)
        ds[f"roll_skip_{k}"] = g.transform(
            lambda s: (s.shift(1) < skip_thr).astype(float).rolling(k, min_periods=1).mean()).fillna(0.3)
        nc += [f"roll_mean_{k}", f"roll_skip_{k}"]
    return ds.sort_values("__i").drop(columns=["__p","__i"]), nc

def _try_load_causal_lm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in LM_CANDIDATES:
        name = name.strip()
        if not name: continue
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(name).to(dev).eval()
            if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token
            print(f"[lm] {name} on {dev}"); return tok, mdl, name
        except Exception as e: print(f"[lm] {name}: {e}")
    return None, None, None

def compute_causal_surprisal(df, tokenizer, model):
    import torch
    dev = next(model.parameters()).device
    out = np.zeros(len(df), dtype=np.float32)
    df = df.reset_index().rename(columns={"index":"_oi"}).copy()
    df = df.sort_values(["text","participant_id","doc_num","page_num","word_idx"])
    @torch.no_grad()
    def score(ids):
        L = len(ids)
        if L < 2: return np.zeros(L, dtype=np.float32)
        lp = np.zeros(L, dtype=np.float32)
        if L <= LM_MAX_LEN:
            t = torch.tensor([ids], dtype=torch.long, device=dev)
            s = torch.log_softmax(model(t).logits[0], dim=-1)
            lp[1:] = (-s[:-1].gather(-1, t[0,1:].unsqueeze(-1)).squeeze(-1)).cpu().numpy()
            return lp
        st = LM_MAX_LEN // 2; first = True; pos = 0
        while pos < L:
            end = min(pos+LM_MAX_LEN, L)
            t = torch.tensor([ids[pos:end]], dtype=torch.long, device=dev)
            s = torch.log_softmax(model(t).logits[0], dim=-1)
            pred = s[:-1].gather(-1, t[0,1:].unsqueeze(-1)).squeeze(-1)
            fl = pos+1 if first else pos+st; fh = end
            if fl < fh: lp[fl:fh] = (-pred[fl-pos-1:fh-pos-1]).cpu().numpy()
            first = False
            if end >= L: break
            pos += st
        return lp
    uo = tokenizer.is_fast; bos = tokenizer.bos_token_id
    gs = df.groupby(["text","participant_id","doc_num","page_num"], sort=False); ng = len(gs)
    for gi, (_, bl) in enumerate(gs):
        words = bl["word"].astype(str).tolist()
        buf, sp, cur = [], [], 0
        for i, w in enumerate(words):
            if i > 0: buf.append(" "); cur += 1
            buf.append(w); sp.append((cur, cur+len(w))); cur += len(w)
        txt = "".join(buf)
        if uo:
            enc = tokenizer(txt, add_special_tokens=False, return_offsets_mapping=True)
            si = enc["input_ids"]; off = enc["offset_mapping"]
        else:
            si = tokenizer.encode(txt, add_special_tokens=False); off = None
        fi = ([bos] if bos else []) + list(si); sh = 1 if bos else 0
        ss = score(fi)[sh:]
        ws = np.zeros(len(words), dtype=np.float32)
        if off:
            wi = 0
            for k, (s, e) in enumerate(off):
                if s == e: continue
                while wi < len(words) and sp[wi][1] <= s: wi += 1
                if wi < len(words) and sp[wi][0] <= s < sp[wi][1]: ws[wi] += ss[k]
        else:
            ci = 0
            for wi, w in enumerate(words):
                iw = tokenizer.encode((" " if wi else "")+w, add_special_tokens=False)
                if ci+len(iw) <= len(ss): ws[wi] = ss[ci:ci+len(iw)].sum()
                ci += len(iw)
        out[bl["_oi"].to_numpy()] = ws
        if (gi+1) % 100 == 0 or (gi+1) == ng: print(f"[lm] {gi+1}/{ng}")
    return out

def _word_token_spans(tokenizer, words):
    buf, sp, cur = [], [], 0
    for i, w in enumerate(words):
        if i > 0: buf.append(" "); cur += 1
        buf.append(w); sp.append((cur, cur+len(w))); cur += len(w)
    enc = tokenizer("".join(buf), add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
    si = enc["input_ids"]; off = enc["offset_mapping"]
    pw = [[] for _ in range(len(words))]; wi = 0
    for k, (s, e) in enumerate(off):
        if s == e: continue
        while wi < len(words) and sp[wi][1] <= s: wi += 1
        if wi < len(words) and sp[wi][0] <= s < sp[wi][1]: pw[wi].append(k)
    return si, pw

def compute_bert_features(df, tokenizer, model, mask_id):
    import torch
    dev = next(model.parameters()).device; H = model.config.hidden_size
    na = np.zeros(len(df), dtype=np.float32); mf = np.zeros(len(df), dtype=np.float32)
    ms = np.zeros(len(df), dtype=np.float32); ea = np.zeros((len(df), H), dtype=np.float32)
    df = df.reset_index().rename(columns={"index":"_oi"}).copy()
    df = df.sort_values(["text","participant_id","doc_num","page_num","word_idx"])
    gs = df.groupby(["text","participant_id","doc_num","page_num"], sort=False); ng = len(gs)
    cls = tokenizer.cls_token_id; sep = tokenizer.sep_token_id
    @torch.no_grad()
    def run(csi, cpw):
        n = len(csi); nw = len(cpw); f1 = [0.0]*nw; fs = [0.0]*nw
        if n == 0: return np.zeros((0, H), dtype=np.float32), f1, fs
        t = torch.tensor([[cls]+csi+[sep]], dtype=torch.long, device=dev)
        se = torch.stack(model(t, output_hidden_states=True).hidden_states[-4:], 0).mean(0)[0][1:-1].cpu().numpy()
        vs, vw = [], []
        for lw, spl in enumerate(cpw):
            if not spl: continue
            v = list(csi)
            for p in spl: v[p] = mask_id
            vs.append(v); vw.append(lw)
        if not vs: return se, f1, fs
        for bs in range(0, len(vs), BERT_BATCH):
            bt = vs[bs:bs+BERT_BATCH]; bw = vw[bs:bs+BERT_BATCH]
            it = torch.tensor([[cls]+v+[sep] for v in bt], dtype=torch.long, device=dev)
            lp = torch.log_softmax(model(it).logits, dim=-1)
            for bi, lw in enumerate(bw):
                spl = cpw[lw]
                f1[lw] = float(-lp[bi, spl[0]+1, csi[spl[0]]].item())
                fs[lw] = sum(float(-lp[bi, p+1, csi[p]].item()) for p in spl)
        return se, f1, fs
    for gi, (_, bl) in enumerate(gs):
        words = bl["word"].astype(str).tolist()
        try: si, pwi = _word_token_spans(tokenizer, words)
        except Exception as e: print(f"[bert] {gi}: {e}"); continue
        nw = len(words); cap = BERT_MAX_LEN - 2; chunks = []; wi = 0
        while wi < nw:
            sc = 0; wj = wi
            while wj < nw:
                ws = len(pwi[wj])
                if sc+ws > cap: break
                sc += ws; wj += 1
            if wj == wi: wj = wi+1
            chunks.append((wi, wj, [si[idx] for k in range(wi, wj) for idx in pwi[k]])); wi = wj
        for (wlo, whi, csi) in chunks:
            if not csi: continue
            cpw = []; cur = 0
            for k in range(wlo, whi):
                ws = len(pwi[k]); cpw.append(list(range(cur, cur+ws))); cur += ws
            se, f1, fs = run(csi, cpw)
            for lw, spl in enumerate(cpw):
                ri = bl["_oi"].iloc[wlo+lw]
                if spl: ea[ri]=se[spl].mean(0); na[ri]=len(spl); mf[ri]=f1[lw]; ms[ri]=fs[lw]
        if (gi+1) % 50 == 0 or (gi+1) == ng: print(f"[bert] {gi+1}/{ng}")
    return {"n_bert_subwords": na, "mlm_logp_first": mf, "mlm_logp_sum": ms, "bert_emb": ea}

_LGB_BASE = dict(learning_rate=0.05, num_leaves=127, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=5, verbosity=-1)
def lgb_params_reg(seed): return {**_LGB_BASE, "objective":"tweedie","tweedie_variance_power":1.4,"metric":"rmse","min_data_in_leaf":200,"seed":seed}
def lgb_params_clf(seed): return {**_LGB_BASE, "objective":"binary","metric":"binary_logloss","min_data_in_leaf":200,"seed":seed}
def lgb_params_pos(seed): return {**_LGB_BASE, "objective":"regression","metric":"rmse","min_data_in_leaf":100,"seed":seed}

def fit_calibration(yt, yp):
    if np.std(yp) < 1e-9: return 1.0, 0.0
    a = np.cov(yp, yt, ddof=0)[0,1] / np.var(yp)
    return float(a), float(yt.mean() - a*yp.mean())

def optimize_blend(yt, p1, p2):
    bw, bs = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 51):
        s = comp_metric(yt, w*p1 + (1-w)*p2)
        if s > bs: bs, bw = s, float(w)
    return bw


def main():
    train = pd.read_csv(INPUT_DIR / "train_data.csv")
    test  = pd.read_csv(INPUT_DIR / "test_data.csv")
    print(f"[data] train={train.shape}  test={test.shape}")
    n_test_orig = len(test); test_ids = set(test["datapointID"].astype(int).tolist())
    assert len(test_ids) == n_test_orig

    train = build_word_features(train);  test = build_word_features(test)
    train = add_context_features(train); test = add_context_features(test)
    train, test = add_corpus_frequency(train, test)
    train, test, has_zipf = add_wordfreq_features(train, test)
    train, test = add_per_word_target_stats(train, test)
    train, test = add_genre_features(train, test)

    fc = ["word_len","alpha_len","n_syllables","is_punct","is_url","has_digit",
          "is_upper_first","is_all_upper","ends_with_punct",
          "doc_num","page_num","word_idx","pos_in_page","rel_pos_in_page","page_size",
          "log_freq","prev_log_freq","next_log_freq",
          "tok_mean","tok_median","tok_std","tok_count","tok_skip_rate",
          "prev_word_len","prev_alpha_len","prev_n_syllables","prev_is_punct","prev_has_digit",
          "next_word_len","next_alpha_len","next_n_syllables","next_is_punct","next_has_digit",
          "genre_id","genre_enc"]
    if has_zipf: fc += ["zipf_freq","prev_zipf_freq","next_zipf_freq"]

    if USE_LM:
        try:
            tok, lm, lmn = _try_load_causal_lm()
            if lm:
                train["surprisal"] = compute_causal_surprisal(train, tok, lm)
                test["surprisal"]  = compute_causal_surprisal(test,  tok, lm)
                for df in (train, test):
                    g = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
                    df["prev_surprisal"] = g["surprisal"].shift(1).fillna(0.0)
                    df["next_surprisal"] = g["surprisal"].shift(-1).fillna(0.0)
                fc += ["surprisal","prev_surprisal","next_surprisal"]; del lm, tok
            else: print("[lm] no model")
        except Exception as e: print(f"[lm] {e}")
    else: print("[lm] disabled")

    if USE_BERT:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[bert] {BERT_NAME} on {dev}")
            bt = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)
            bm = AutoModelForMaskedLM.from_pretrained(BERT_NAME).to(dev).eval()
            cn = ["word_id","word","text","doc_num","page_num","word_idx"]
            uniq = pd.concat([train[cn], test[cn]], axis=0).drop_duplicates("word_id").reset_index(drop=True)
            uniq["participant_id"] = 0
            print(f"[bert] {len(uniq)} unique word_ids")
            ubf = compute_bert_features(uniq, bt, bm, bt.mask_token_id)
            uf = pd.DataFrame({"word_id": uniq["word_id"].values,
                "n_bert_subwords": ubf["n_bert_subwords"],
                "mlm_logp_first":  ubf["mlm_logp_first"],
                "mlm_logp_sum":    ubf["mlm_logp_sum"]})
            i2p = {w: i for i, w in enumerate(uniq["word_id"].values)}
            train = train.merge(uf, on="word_id", how="left")
            test  = test.merge(uf,  on="word_id", how="left")
            te = ubf["bert_emb"][train["word_id"].map(i2p).values]
            ee = ubf["bert_emb"][test["word_id"].map(i2p).values]
            for df in (train, test):
                g = df.groupby(["text","participant_id","doc_num","page_num"], sort=False)
                df["prev_mlm_logp_first"] = g["mlm_logp_first"].shift(1).fillna(0.0)
                df["next_mlm_logp_first"] = g["mlm_logp_first"].shift(-1).fillna(0.0)
            fc += ["n_bert_subwords","mlm_logp_first","mlm_logp_sum","prev_mlm_logp_first","next_mlm_logp_first"]
            valid = np.linalg.norm(te, axis=1) > 1e-6
            nc = min(BERT_PCA_DIM, te.shape[1], int(valid.sum())-1)
            if nc > 0:
                pca = PCA(n_components=nc, random_state=SEED); pca.fit(te[valid])
                tp = pca.transform(te); ep = pca.transform(ee)
                ec = [f"bert_pc_{i}" for i in range(nc)]
                for i, c in enumerate(ec): train[c] = tp[:,i]; test[c] = ep[:,i]
                fc += ec; print(f"[bert] PCA var={pca.explained_variance_ratio_.sum():.3f}")
            del bm, bt
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[bert] {e}")
    else: print("[bert] disabled")

    train["_orig"] = True
    train = augment_participant_mixup(train)
    train = augment_genre_average_reader(train)
    train["_orig"] = train["_orig"].fillna(False)

    for c in fc: assert c in train.columns and c in test.columns, f"missing: {c}"
    for df in (train, test):
        n = int(df[fc].isna().sum().sum())
        if n: df[fc] = df[fc].fillna(0.0)
    print(f"[feat] {len(fc)} features")

    train = train.sort_values(["text","participant_id","doc_num","page_num","word_idx"]).reset_index(drop=True)
    omask = train["_orig"].values.astype(bool)
    X = train[fc].values; y = train["answer"].values.astype(float)
    grps = train["text"].values; Xt = test[fc].values
    ns = min(N_FOLDS, train["text"].nunique())
    gkf = GroupKFold(n_splits=ns); splits = list(gkf.split(X, y, grps))

    oa = np.zeros(len(train)); ta = np.zeros(len(test)); fsa = []
    for fi, (tri, vai) in enumerate(splits):
        so = np.zeros(len(vai)); st = np.zeros(len(test))
        for si in range(N_SEEDS):
            seed = SEED+1000*si+fi
            b = lgb.train(lgb_params_reg(seed), lgb.Dataset(X[tri], y[tri]),
                          num_boost_round=4000, valid_sets=[lgb.Dataset(X[vai], y[vai])],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            so += b.predict(X[vai], num_iteration=b.best_iteration)
            st += b.predict(Xt,     num_iteration=b.best_iteration)
        so /= N_SEEDS; st /= N_SEEDS; oa[vai] = so; ta += st/ns
        s = comp_metric(y[vai], so); fsa.append(s)
        print(f"[s1 f{fi}] {sorted(set(grps[vai]))} {s:.3f}")
    print(f"[s1] mean={np.mean(fsa):.3f} OOF={comp_metric(y,oa):.3f}")

    if USE_TWO_STAGE:
        ob = np.zeros(len(train)); tb = np.zeros(len(test)); fsb = []; ysk = (y==0).astype(float)
        for fi, (tri, vai) in enumerate(splits):
            sko = np.zeros(len(vai)); skt = np.zeros(len(test))
            rgo = np.zeros(len(vai)); rgt = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED+1000*si+fi
                clf = lgb.train(lgb_params_clf(seed), lgb.Dataset(X[tri], ysk[tri]),
                                num_boost_round=2000, valid_sets=[lgb.Dataset(X[vai], ysk[vai])],
                                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                sko += clf.predict(X[vai], num_iteration=clf.best_iteration)
                skt += clf.predict(Xt,     num_iteration=clf.best_iteration)
                ptr = tri[y[tri]>0]; pva = vai[y[vai]>0]
                if len(pva) < 10: pva = vai
                rg = lgb.train(lgb_params_pos(seed), lgb.Dataset(X[ptr], np.log1p(y[ptr])),
                               num_boost_round=4000, valid_sets=[lgb.Dataset(X[pva], np.log1p(y[pva]))],
                               callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                rgo += np.expm1(rg.predict(X[vai], num_iteration=rg.best_iteration))
                rgt += np.expm1(rg.predict(Xt,     num_iteration=rg.best_iteration))
            sko /= N_SEEDS; skt /= N_SEEDS; rgo /= N_SEEDS; rgt /= N_SEEDS
            sko = np.clip(sko,0,1); skt = np.clip(skt,0,1)
            rgo = np.clip(rgo,0,None); rgt = np.clip(rgt,0,None)
            ob[vai] = (1-sko)*rgo; tb += ((1-skt)*rgt)/ns
            s = comp_metric(y[vai], ob[vai]); fsb.append(s)
            print(f"[s2 f{fi}] {sorted(set(grps[vai]))} {s:.3f}")
        print(f"[s2] mean={np.mean(fsb):.3f} OOF={comp_metric(y,ob):.3f}")
    else:
        ob = oa.copy(); tb = ta.copy()

    w = optimize_blend(y, oa, ob) if USE_TWO_STAGE else 1.0
    print(f"[blend] w={w:.3f}")
    obl = w*oa + (1-w)*ob; tbl = w*ta + (1-w)*tb
    print(f"[blend] OOF={comp_metric(y,obl):.3f}")

    if USE_ROLLING:
        torig = train[omask].copy().reset_index(drop=True); oorig = obl[omask]
        torig, rc = compute_rolling_features(torig, oorig)
        test,  _  = compute_rolling_features(test, tbl)
        fc2 = fc + rc
        X2 = torig[fc2].values; y2 = torig["answer"].values.astype(float)
        g2 = torig["text"].values; X2t = test[fc2].values
        ns2 = min(N_FOLDS, len(np.unique(g2)))
        oc = np.zeros(len(torig)); tc2 = np.zeros(len(test)); fsc = []
        for fi, (tr2, va2) in enumerate(GroupKFold(n_splits=ns2).split(X2, y2, g2)):
            so = np.zeros(len(va2)); st = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED+3000+1000*si+fi
                b = lgb.train(lgb_params_reg(seed), lgb.Dataset(X2[tr2], y2[tr2]),
                              num_boost_round=4000, valid_sets=[lgb.Dataset(X2[va2], y2[va2])],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                so += b.predict(X2[va2], num_iteration=b.best_iteration)
                st += b.predict(X2t,    num_iteration=b.best_iteration)
            so /= N_SEEDS; st /= N_SEEDS; oc[va2] = so; tc2 += st/ns2
            s = comp_metric(y2[va2], so); fsc.append(s)
            print(f"[p2 f{fi}] {sorted(set(g2[va2]))} {s:.3f}")
        print(f"[p2] mean={np.mean(fsc):.3f} OOF={comp_metric(y2,oc):.3f}")
        w2 = optimize_blend(y2, oorig, oc); print(f"[p2 blend] w={w2:.3f}")
        of = w2*oorig + (1-w2)*oc; tbl = w2*tbl + (1-w2)*tc2
        print(f"[p2 blend] OOF={comp_metric(y2,of):.3f}")
        a, b = fit_calibration(y2, of)
    else:
        a, b = fit_calibration(y, obl)

    print(f"[calib] a={a:.4f} b={b:.4f}")
    tc = np.clip(a*tbl + b, 0, None)
    print(f"[pred] mean={tc.mean():.1f} #zeros={(tc<1).sum()}")

    sub = pd.DataFrame({"subtaskID": np.ones(len(test),dtype=int),
                        "datapointID": test["datapointID"].astype(int).values,
                        "answer": np.round(tc).astype(int)}
                       ).sort_values("datapointID").reset_index(drop=True)
    assert list(sub.columns) == ["subtaskID","datapointID","answer"]
    assert len(sub) == n_test_orig
    assert set(sub["datapointID"].tolist()) == test_ids
    assert sub["datapointID"].is_monotonic_increasing
    assert sub["subtaskID"].eq(1).all()
    assert sub["answer"].notna().all()
    assert (sub["answer"] >= 0).all()
    assert sub["answer"].nunique() > 1
    sub.to_csv(OUTPUT_DIR/"submission.csv", index=False)
    print(f"[out] {len(sub)} rows  checks passed")
    print(sub.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
