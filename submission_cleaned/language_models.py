from __future__ import annotations
import numpy as np
import pandas as pd
from .config import LM_CANDIDATES, LM_MAX_LEN, BERT_MAX_LEN, BERT_BATCH


def _try_load_causal_lm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name in LM_CANDIDATES:
        name = name.strip()
        if not name:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            print(f'[causal-lm] loaded {name} on {device}')
            return tokenizer, model, name
        except Exception as e:
            print(f'[causal-lm] {name} failed: {e}')
    return None, None, None


def compute_causal_surprisal(df: pd.DataFrame, tokenizer, model) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    out = np.zeros(len(df), dtype=np.float32)
    df = df.reset_index().rename(columns={'index': '_orig_idx'}).copy()
    df = df.sort_values(['text', 'participant_id', 'doc_num', 'page_num', 'word_idx'])

    @torch.no_grad()
    def score_ids(full_ids: list) -> np.ndarray:
        L = len(full_ids)
        if L < 2:
            return np.zeros(L, dtype=np.float32)
        logp = np.zeros(L, dtype=np.float32)
        if L <= LM_MAX_LEN:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            lp  = torch.log_softmax(model(ids).logits[0], dim=-1)
            logp[1:] = (-lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)).cpu().numpy()
            return logp
        stride = LM_MAX_LEN // 2
        first = True
        pos = 0
        while pos < L:
            end = min(pos + LM_MAX_LEN, L)
            ids = torch.tensor([full_ids[pos:end]], dtype=torch.long, device=device)
            lp = torch.log_softmax(model(ids).logits[0], dim=-1)
            pred = lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            fill_lo = pos + 1 if first else pos + stride
            fill_hi = end
            if fill_lo < fill_hi:
                logp[fill_lo:fill_hi] = (-pred[fill_lo - pos - 1:fill_hi - pos - 1]).cpu().numpy()
            first = False
            if end >= L:
                break
            pos += stride
        return logp

    use_offsets = tokenizer.is_fast
    bos = tokenizer.bos_token_id
    groups = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
    n_groups = len(groups)

    for gi, (_, block) in enumerate(groups):
        words = block['word'].astype(str).tolist()
        buf, spans, cur = [], [], 0
        for i, w in enumerate(words):
            if i > 0:
                buf.append(' '); cur += 1
            buf.append(w); spans.append((cur, cur + len(w))); cur += len(w)
        text = ''.join(buf)

        if use_offsets:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            sub_ids = enc['input_ids']
            offsets = enc['offset_mapping']
        else:
            sub_ids = tokenizer.encode(text, add_special_tokens=False)
            offsets = None

        full_ids = ([bos] if bos is not None else []) + list(sub_ids)
        shift = 1 if bos is not None else 0
        sub_surps = score_ids(full_ids)[shift:]
        word_surps = np.zeros(len(words), dtype=np.float32)

        if offsets is not None:
            wi = 0
            for k, (s, e) in enumerate(offsets):
                if s == e: continue
                while wi < len(words) and spans[wi][1] <= s: wi += 1
                if wi < len(words) and spans[wi][0] <= s < spans[wi][1]:
                    word_surps[wi] += sub_surps[k]
        else:
            cur_id = 0
            for wi, w in enumerate(words):
                ids_w = tokenizer.encode((' ' if wi > 0 else '') + w, add_special_tokens=False)
                if cur_id + len(ids_w) <= len(sub_surps):
                    word_surps[wi] = sub_surps[cur_id:cur_id + len(ids_w)].sum()
                cur_id += len(ids_w)

        out[block['_orig_idx'].to_numpy()] = word_surps
        if (gi + 1) % 100 == 0 or (gi + 1) == n_groups:
            print(f'[causal-lm] {gi + 1}/{n_groups}')
    return out


def _word_token_spans(tokenizer, words: list) -> tuple:
    buf, spans, cur = [], [], 0
    for i, w in enumerate(words):
        if i > 0:
            buf.append(' '); cur += 1
        buf.append(w); spans.append((cur, cur + len(w))); cur += len(w)
    enc = tokenizer(
        ''.join(buf),
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    sub_ids = enc['input_ids']
    offsets = enc['offset_mapping']
    per_word = [[] for _ in range(len(words))]
    wi = 0
    for k, (s, e) in enumerate(offsets):
        if s == e: continue
        while wi < len(words) and spans[wi][1] <= s: wi += 1
        if wi < len(words) and spans[wi][0] <= s < spans[wi][1]:
            per_word[wi].append(k)
    return sub_ids, per_word


def compute_bert_features(df: pd.DataFrame, tokenizer, model, mask_token_id: int) -> dict:
    import torch
    device = next(model.parameters()).device
    hidden = model.config.hidden_size

    n_subw_arr = np.zeros(len(df), dtype=np.float32)
    mlm_first_arr = np.zeros(len(df), dtype=np.float32)
    mlm_sum_arr = np.zeros(len(df), dtype=np.float32)
    emb_arr = np.zeros((len(df), hidden), dtype=np.float32)

    df = df.reset_index().rename(columns={'index': '_orig_idx'}).copy()
    df = df.sort_values(['text', 'participant_id', 'doc_num', 'page_num', 'word_idx'])
    groups = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
    n_groups = len(groups)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    @torch.no_grad()
    def run_chunk(chunk_sub_ids: list, chunk_per_word: list):
        n_sub = len(chunk_sub_ids)
        n_words = len(chunk_per_word)
        mlm_first = [0.0] * n_words
        mlm_sum = [0.0] * n_words
        if n_sub == 0:
            return np.zeros((0, hidden), dtype=np.float32), mlm_first, mlm_sum
        ids = torch.tensor([[cls_id] + chunk_sub_ids + [sep_id]], dtype=torch.long, device=device)
        sub_embs = torch.stack(
            model(ids, output_hidden_states=True).hidden_states[-4:], dim=0
        ).mean(0)[0][1:-1].cpu().numpy()
        variants, variant_words = [], []
        for lw, spl in enumerate(chunk_per_word):
            if not spl: continue
            v = list(chunk_sub_ids)
            for p in spl: v[p] = mask_token_id
            variants.append(v); variant_words.append(lw)
        if not variants:
            return sub_embs, mlm_first, mlm_sum
        for batch_start in range(0, len(variants), BERT_BATCH):
            batch = variants[batch_start:batch_start + BERT_BATCH]
            wids = variant_words[batch_start:batch_start + BERT_BATCH]
            ids_t = torch.tensor([[cls_id] + v + [sep_id] for v in batch], dtype=torch.long, device=device)
            lp = torch.log_softmax(model(ids_t).logits, dim=-1)
            for bi, lw in enumerate(wids):
                spl = chunk_per_word[lw]
                mlm_first[lw] = float(-lp[bi, spl[0] + 1, chunk_sub_ids[spl[0]]].item())
                mlm_sum[lw]   = sum(float(-lp[bi, p + 1, chunk_sub_ids[p]].item()) for p in spl)
        return sub_embs, mlm_first, mlm_sum

    for gi, (_, block) in enumerate(groups):
        words = block['word'].astype(str).tolist()
        try:
            sub_ids, per_word_idx = _word_token_spans(tokenizer, words)
        except Exception as e:
            print(f'[bert] group {gi} failed: {e}'); continue

        n_words = len(words)
        capacity = BERT_MAX_LEN - 2
        chunks = []
        wi = 0
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
                ws = len(per_word_idx[k]); cpw.append(list(range(cur, cur + ws))); cur += ws
            sub_embs, mlm_first, mlm_sum = run_chunk(csi, cpw)
            for lw, spl in enumerate(cpw):
                ri = block['_orig_idx'].iloc[wlo + lw]
                if spl:
                    emb_arr[ri] = sub_embs[spl].mean(axis=0)
                    n_subw_arr[ri] = len(spl)
                    mlm_first_arr[ri] = mlm_first[lw]
                    mlm_sum_arr[ri] = mlm_sum[lw]
        if (gi + 1) % 50 == 0 or (gi + 1) == n_groups:
            print(f'[bert] {gi + 1}/{n_groups}')

    return {
        'n_bert_subwords': n_subw_arr,
        'mlm_logp_first': mlm_first_arr,
        'mlm_logp_sum': mlm_sum_arr,
        'bert_emb': emb_arr,
    }
