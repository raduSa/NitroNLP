'''
Entry point. Run from the NitroNLP directory:
    python -m submission_cleaned.main
'''
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import lightgbm as lgb

from .config import (
    INPUT_DIR, OUTPUT_DIR, SEED, N_FOLDS, N_SEEDS,
    USE_LM, USE_BERT, USE_TWO_STAGE, USE_ROLLING,
    BERT_NAME, BERT_PCA_DIM,
)
from .metrics import comp_metric, fit_calibration, optimize_blend
from .word_features import build_word_features, add_context_features
from .frequency import add_corpus_frequency, add_wordfreq_features
from .target_stats import add_per_word_target_stats, add_genre_features
from .augmentation import augment_participant_mixup, augment_genre_average_reader
from .language_models import _try_load_causal_lm, compute_causal_surprisal, compute_bert_features
from .rolling import compute_rolling_features
from .lgb_params import lgb_params_reg, lgb_params_clf, lgb_params_pos


def main() -> None:
    train = pd.read_csv(INPUT_DIR / 'train_data.csv')
    test = pd.read_csv(INPUT_DIR / 'test_data.csv')
    print(f'[data] train={train.shape}  test={test.shape}')
    n_test_orig = len(test)
    test_ids_orig = set(test['datapointID'].astype(int).tolist())
    assert len(test_ids_orig) == n_test_orig

    # Feature engineering
    train = build_word_features(train); test = build_word_features(test)
    train = add_context_features(train); test = add_context_features(test)
    train, test = add_corpus_frequency(train, test)
    train, test, has_zipf = add_wordfreq_features(train, test)
    train, test = add_per_word_target_stats(train, test)
    train, test = add_genre_features(train, test)

    feature_cols = [
        'word_len', 'alpha_len', 'n_syllables', 'is_punct', 'is_url', 'has_digit',
        'is_upper_first', 'is_all_upper', 'ends_with_punct',
        'doc_num', 'page_num', 'word_idx', 'pos_in_page', 'rel_pos_in_page', 'page_size',
        'log_freq', 'prev_log_freq', 'next_log_freq',
        'tok_mean', 'tok_median', 'tok_std', 'tok_count', 'tok_skip_rate',
        'prev_word_len', 'prev_alpha_len', 'prev_n_syllables', 'prev_is_punct', 'prev_has_digit',
        'next_word_len', 'next_alpha_len', 'next_n_syllables', 'next_is_punct', 'next_has_digit',
        'genre_id', 'genre_enc',
    ]
    if has_zipf:
        feature_cols += ['zipf_freq', 'prev_zipf_freq', 'next_zipf_freq']

    # Causal LM surprisal
    if USE_LM:
        try:
            tokenizer, lm_model, lm_name = _try_load_causal_lm()
            if lm_model is not None:
                print(f'[causal-lm] scoring train+test ({lm_name})')
                train['surprisal'] = compute_causal_surprisal(train, tokenizer, lm_model)
                test['surprisal'] = compute_causal_surprisal(test,  tokenizer, lm_model)
                for df in (train, test):
                    grp = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
                    df['prev_surprisal'] = grp['surprisal'].shift(1).fillna(0.0)
                    df['next_surprisal'] = grp['surprisal'].shift(-1).fillna(0.0)
                feature_cols += ['surprisal', 'prev_surprisal', 'next_surprisal']
                del lm_model, tokenizer
            else:
                print('[causal-lm] no model loaded')
        except Exception as e:
            print(f'[causal-lm] failed: {e}')
    else:
        print('[causal-lm] disabled')

    # BERT contextual features
    if USE_BERT:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'[bert] loading {BERT_NAME} on {device}')
            btok = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)
            bmodel = AutoModelForMaskedLM.from_pretrained(BERT_NAME).to(device).eval()

            cols_needed = ['word_id', 'word', 'text', 'doc_num', 'page_num', 'word_idx']
            uniq = (
                pd.concat([train[cols_needed], test[cols_needed]], axis=0)
                .drop_duplicates(subset=['word_id'])
                .reset_index(drop=True)
            )
            uniq['participant_id'] = 0
            print(f'[bert] {len(uniq)} unique word_ids')
            ubf = compute_bert_features(uniq, btok, bmodel, btok.mask_token_id)

            uniq_feat = pd.DataFrame({
                'word_id': uniq['word_id'].values,
                'n_bert_subwords': ubf['n_bert_subwords'],
                'mlm_logp_first': ubf['mlm_logp_first'],
                'mlm_logp_sum': ubf['mlm_logp_sum'],
            })
            id2pos = {wid: i for i, wid in enumerate(uniq['word_id'].values)}
            train = train.merge(uniq_feat, on='word_id', how='left')
            test = test.merge(uniq_feat,  on='word_id', how='left')
            tr_emb = ubf['bert_emb'][train['word_id'].map(id2pos).values]
            te_emb = ubf['bert_emb'][test['word_id'].map(id2pos).values]

            for df in (train, test):
                grp = df.groupby(['text', 'participant_id', 'doc_num', 'page_num'], sort=False)
                df['prev_mlm_logp_first'] = grp['mlm_logp_first'].shift(1).fillna(0.0)
                df['next_mlm_logp_first'] = grp['mlm_logp_first'].shift(-1).fillna(0.0)
            feature_cols += [
                'n_bert_subwords', 'mlm_logp_first', 'mlm_logp_sum',
                'prev_mlm_logp_first', 'next_mlm_logp_first',
            ]

            valid = np.linalg.norm(tr_emb, axis=1) > 1e-6
            n_comp = min(BERT_PCA_DIM, tr_emb.shape[1], int(valid.sum()) - 1)
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=SEED)
                pca.fit(tr_emb[valid])
                tr_p = pca.transform(tr_emb); te_p = pca.transform(te_emb)
                emb_cols = [f'bert_pc_{i}' for i in range(n_comp)]
                for i, col in enumerate(emb_cols):
                    train[col] = tr_p[:, i]; test[col] = te_p[:, i]
                feature_cols += emb_cols
                print(f'[bert] PCA explained_var={pca.explained_variance_ratio_.sum()}')
            del bmodel, btok
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[bert] failed: {e}')
    else:
        print('[bert] disabled')

    # Augmentation
    train['_is_orig'] = True
    train = augment_participant_mixup(train)
    train = augment_genre_average_reader(train)
    train['_is_orig'] = train['_is_orig'].fillna(False)

    for col in feature_cols:
        assert col in train.columns and col in test.columns, f'missing feature: {col}'
    nan_tr = int(train[feature_cols].isna().sum().sum())
    nan_te = int(test[feature_cols].isna().sum().sum())
    if nan_tr or nan_te:
        print(f'[feat] WARNING NaNs train={nan_tr} test={nan_te}; filling 0')
        train[feature_cols] = train[feature_cols].fillna(0.0)
        test[feature_cols] = test[feature_cols].fillna(0.0)
    print(f'[feat] {len(feature_cols)} features')

    train = train.sort_values(['text', 'participant_id', 'doc_num', 'page_num', 'word_idx']).reset_index(drop=True)
    orig_mask = train['_is_orig'].values.astype(bool)
    X = train[feature_cols].values
    y = train['answer'].values.astype(float)
    groups = train['text'].values
    Xt = test[feature_cols].values

    n_splits = min(N_FOLDS, train['text'].nunique())
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    # Stage 1: Tweedie regressor
    oof_a = np.zeros(len(train)); test_a = np.zeros(len(test)); scores_a = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        s_oof = np.zeros(len(va_idx)); s_test = np.zeros(len(test))
        for si in range(N_SEEDS):
            seed = SEED + 1000 * si + fi
            dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
            dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
            b = lgb.train(lgb_params_reg(seed), dtr, num_boost_round=4000, valid_sets=[dva],
                             callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            s_oof += b.predict(X[va_idx], num_iteration=b.best_iteration)
            s_test += b.predict(Xt,        num_iteration=b.best_iteration)
        s_oof /= N_SEEDS; s_test /= N_SEEDS
        oof_a[va_idx] = s_oof; test_a += s_test / n_splits
        sc = comp_metric(y[va_idx], s_oof); scores_a.append(sc)
        print(f'[stage1 fold {fi}] {sorted(set(groups[va_idx]))}  score={sc}')
    print(f'[stage1] mean={np.mean(scores_a)}  OOF={comp_metric(y, oof_a)}')

    # Stage 2: skip classifier × log1p regressor
    if USE_TWO_STAGE:
        oof_b = np.zeros(len(train)); test_b = np.zeros(len(test)); scores_b = []
        y_skip = (y == 0).astype(float)
        for fi, (tr_idx, va_idx) in enumerate(splits):
            sk_oof = np.zeros(len(va_idx)); sk_test = np.zeros(len(test))
            rg_oof = np.zeros(len(va_idx)); rg_test = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED + 1000 * si + fi
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
            sk_oof = np.clip(sk_oof,  0, 1);    sk_test = np.clip(sk_test, 0, 1)
            rg_oof = np.clip(rg_oof,  0, None); rg_test = np.clip(rg_test, 0, None)
            oof_b[va_idx] = (1 - sk_oof) * rg_oof
            test_b += ((1 - sk_test) * rg_test) / n_splits
            sc = comp_metric(y[va_idx], oof_b[va_idx]); scores_b.append(sc)
            print(f'[stage2 fold {fi}] {sorted(set(groups[va_idx]))}  score={sc}')
        print(f'[stage2] mean={np.mean(scores_b)}  OOF={comp_metric(y, oof_b)}')
    else:
        oof_b = oof_a.copy(); test_b = test_a.copy()

    # Blend stage 1 + stage 2
    w_s1 = optimize_blend(y, oof_a, oof_b) if USE_TWO_STAGE else 1.0
    print(f'[blend] w_stage1={w_s1}')
    oof_blend = w_s1 * oof_a  + (1 - w_s1) * oof_b
    test_blend = w_s1 * test_a + (1 - w_s1) * test_b
    print(f'[blend] OOF={comp_metric(y, oof_blend)}')

    # Pass 2: rolling session features
    if USE_ROLLING:
        train_orig = train[orig_mask].copy().reset_index(drop=True)
        oof_orig = oof_blend[orig_mask]
        train_orig, roll_cols = compute_rolling_features(train_orig, oof_orig)
        test,  _ = compute_rolling_features(test, test_blend)
        p2_feature_cols = feature_cols + roll_cols
        X2 = train_orig[p2_feature_cols].values
        y2 = train_orig['answer'].values.astype(float)
        g2 = train_orig['text'].values
        Xt2 = test[p2_feature_cols].values

        n_splits2 = min(N_FOLDS, len(np.unique(g2)))
        oof_c = np.zeros(len(train_orig)); test_c = np.zeros(len(test)); scores_c = []
        for fi, (tr2, va2) in enumerate(GroupKFold(n_splits=n_splits2).split(X2, y2, g2)):
            s_oof = np.zeros(len(va2)); s_test = np.zeros(len(test))
            for si in range(N_SEEDS):
                seed = SEED + 3000 + 1000 * si + fi
                dtr2 = lgb.Dataset(X2[tr2], y2[tr2])
                dva2 = lgb.Dataset(X2[va2], y2[va2], reference=dtr2)
                b2 = lgb.train(lgb_params_reg(seed), dtr2, num_boost_round=4000, valid_sets=[dva2],
                                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                s_oof += b2.predict(X2[va2], num_iteration=b2.best_iteration)
                s_test += b2.predict(Xt2,     num_iteration=b2.best_iteration)
            s_oof /= N_SEEDS; s_test /= N_SEEDS
            oof_c[va2] = s_oof; test_c += s_test / n_splits2
            sc = comp_metric(y2[va2], s_oof); scores_c.append(sc)
            print(f'[pass2 fold {fi}] {sorted(set(g2[va2]))}  score={sc}')
        print(f'[pass2] mean={np.mean(scores_c)}  OOF={comp_metric(y2, oof_c)}')

        w_p2 = optimize_blend(y2, oof_orig, oof_c)
        print(f'[pass2 blend] w_pass1={w_p2}  w_pass2={1 - w_p2}')
        oof_final = w_p2 * oof_orig   + (1 - w_p2) * oof_c
        test_blend = w_p2 * test_blend + (1 - w_p2) * test_c
        print(f'[pass2 blend] OOF={comp_metric(y2, oof_final)}')
        calib_a, calib_b = fit_calibration(y2, oof_final)
    else:
        calib_a, calib_b = fit_calibration(y, oof_blend)

    # Linear calibration + submission writ
    print(f'[calib] a={calib_a}  b={calib_b}')
    test_cal = np.clip(calib_a * test_blend + calib_b, 0, None)
    print(f'[pred] min={test_cal.min()}  mean={test_cal.mean()}  max={test_cal.max()}  #zeros={(test_cal < 1).sum()}')

    sub = pd.DataFrame({
        'subtaskID':   np.ones(len(test), dtype=int),
        'datapointID': test['datapointID'].astype(int).values,
        'answer':      np.round(test_cal).astype(int),
    }).sort_values('datapointID').reset_index(drop=True)

    assert list(sub.columns) == ['subtaskID', 'datapointID', 'answer']
    assert len(sub) == n_test_orig
    assert set(sub['datapointID'].tolist()) == test_ids_orig
    assert sub['datapointID'].is_monotonic_increasing
    assert sub['subtaskID'].eq(1).all()
    assert sub['answer'].notna().all()
    assert (sub['answer'] >= 0).all()
    assert sub['answer'].nunique() > 1

    sub_path = OUTPUT_DIR / 'submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f'[out] wrote {sub_path}  rows={len(sub)}')
    print(sub.head(6).to_string(index=False))
    print('[out] all checks passed.')


if __name__ == '__main__':
    main()
