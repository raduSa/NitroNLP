# Nitro NLP — Estimating Total Reading Times in Romanian Texts
**Echipa "DragonIPV4"** — Chirilușs Antonie · Luparu Ioan Teodor · Savin Radu Andrei

---

## Slide 1 — The Problem

**What is Total Reading Time (TRT)?**
- Sum of all eye fixations on a word, measured in milliseconds
- TRT = 0 means the reader skipped the word entirely

**Why is it hard?**
- Long words, rare words, and contextually unexpected words → higher TRT
- Target distribution is zero-inflated (30.6% skips) with a long right tail

**Evaluation metric**
$$\text{score} = 100 \times \frac{\max(0,\, R^2) + |\rho|}{2}$$
- Requires both good ranking (Pearson) and correct scale (R²)

**Core challenge: generalisation**
- Test set has 5 readers and 3 texts that never appear in training
- Model must transfer to both unseen readers and unseen documents simultaneously

---

## Slide 2 — Dataset & Related Work

**Dataset**
- Train: 135,210 rows — 30 readers × 9 texts, 5 genres (literary, popular science, argumentation, encyclopaedic, instructional)
- Test: 9,425 rows — 5 held-out readers × 3 held-out texts, zero overlap with training

**Key findings from prior work (Hodivoianu et al., 2025)**
- Word length is the strongest predictor of TRT (Pearson 0.63)
- Word frequency (0.38) and MLM log-probability (0.37) follow
- Combining scalar features with BERT embeddings gives best results
- First-subword log-probability outperforms averaging all subword tokens

**Our additions over prior work**
- Causal surprisal from Romanian GPT-2
- Data augmentation for unseen reader generalisation
- Two-stage LightGBM with Tweedie + skip-gate architecture
- Session-level rolling features (pass 2)

---

## Slide 3 — Feature Engineering

**Word-level surface** — length, syllable count, is_punct, is_url, has_digit, is_upper_first, ends_with_punct

**Positional** — doc/page/word index from `word_id`, position within page, page size, relative position; same stats for previous and next word

**Frequency**
- Corpus log-frequency from train+test counts; prev/next neighbour features
- External Zipf frequency via `wordfreq` (Romanian), independent of corpus size

**Genre features**
- Genre prefix extracted from text name (`lit_`, `popsci_`, `arg_`, `enc_`, `ins_`)
- Leave-text-out genre mean TRT (`genre_enc`) — no lookahead within the current document

**Per-word target statistics**
- Mean, median, std, count, skip rate per lowercased token
- Computed leave-text-out to prevent target leakage

---

## Slide 4 — Language Model Features

**Causal surprisal** (Romanian GPT-2)
- $-\log p(w_t \mid w_1, \ldots, w_{t-1})$ — how unexpected is the word given its left context
- Fallback chain: RoGPT2-large → RoGPT2-medium → RoGPT2-base → gpt-neo-romanian-780m → gpt-neo-romanian-125m
- Sliding window with 50% overlap for pages > 1024 tokens
- Previous and next surprisal values added as neighbour features

**BERT MLM log-probability** (Romanian BERT, Dumitrescu et al., 2020)
- Replace subword tokens with `[MASK]`, score the original token
- `mlm_logp_first` — first subword token only (most predictive)
- `mlm_logp_sum` — sum over all subword tokens
- Conditions on both left and right context, complementary to causal surprisal

**BERT contextual embeddings**
- Average of last 4 hidden layers, pooled across subword tokens via character-offset alignment
- 768-dim → PCA → 24 principal components used as features
- `n_bert_subwords` added as morphological complexity proxy
- Extracted once per unique `word_id` (not repeated across 30 participants)

---

## Slide 5 — Data Augmentation

**Why augment?** Both readers and texts in the test set are fully unseen — the model must generalise to new reader profiles without new texts.

**Participant mixup**
- For each (text, page, word position) triplet, randomly pair two training readers
- Linearly interpolate their TRT values: $\alpha \cdot \text{TRT}_A + (1-\alpha) \cdot \text{TRT}_B$
- Also add one "average reader" row (mean TRT over all readers at that position)

**Genre average reader**
- For each (genre, word) pair, add a synthetic row with TRT = genre-wide mean for that word
- Anchors the model to genre-conditioned word baselines without needing the exact text

**Leakage protection**
- Augmented rows are flagged with `_is_orig = False`
- The second training pass (rolling features) trains only on original rows, whose pass-1 predictions are leakage-free

---

## Slide 6 — Two-Stage LightGBM

**Why two stages?** TRT is zero-inflated — a single model must learn both skip probability and non-zero duration, which have conflicting shapes.

**Stage 1 — Tweedie regressor**
- LightGBM with Tweedie loss (variance power 1.4)
- Designed for zero-inflated positive-valued targets; trained on all rows

**Stage 2 — skip classifier × log1p regressor**
- Binary classifier predicts $P(\text{skip})$
- Separate regressor trained only on non-zero rows with $\log(1+\text{TRT})$ target
- Combined prediction: $(1 - \hat{p}_{\text{skip}}) \times \hat{y}_{\text{nz}}$

**Blend**
- Final: $w \cdot \hat{y}_{\text{s1}} + (1-w) \cdot \hat{y}_{\text{s2}}$, with $w$ grid-searched on OOF to maximise the competition metric

**Cross-validation**
- GroupKFold by text — mirrors the unseen-text test structure exactly
- 3 seeds per fold, predictions averaged (variance reduction)
- Params: LR 0.05, 127 leaves, feature fraction 0.9, bagging fraction 0.9

---

## Slide 7 — Rolling Features & Calibration

**Rolling session features (pass 2)**

Reading is sequential — a reader's pace at word $t$ affects word $t+1$.

After pass-1 OOF predictions are available, compute per-participant:
- `session_word_idx` — absolute word position in the reading session
- `pass1_pred` — the blended pass-1 prediction for this word
- `roll_mean_k` — rolling mean of previous $k$ predictions ($k \in \{10, 20, 50\}$)
- `roll_skip_k` — rolling skip-rate estimate over previous $k$ words

A second GroupKFold LightGBM is trained on original rows using base + rolling features. Blend weight $w_2$ is optimised between pass-1 and pass-2 OOF.

**Linear calibration**

Tweedie models rank well but can drift in absolute scale. After training:
$$\hat{y}_{\text{final}} = a\hat{y} + b$$
OLS fit on OOF predictions. Pearson is unchanged; R² improves — free points on the metric.

---

## Slide 8 — Results & Conclusion

**Results (out-of-fold progression)**

| Stage | OOF Score |
|---|---|
| Stage 1 — Tweedie LGB (base features) | ~36–38 |
| + Stage 2 two-stage blend | improved |
| + Rolling features (pass 2) | improved |
| + Linear calibration | best |

Each addition contributes: two-stage beats Tweedie alone across all folds; rolling features add signal unavailable to pass 1; calibration lifts R² at no cost to Pearson.

**What drives performance**
- Word length and per-word target statistics are the strongest individual predictors
- LM surprisal and MLM log-probability provide the largest contextual gains
- GroupKFold by text is critical — random-split CV would give misleadingly high scores

**Conclusion**

A multi-stage pipeline combining handcrafted features, Romanian GPT-2 surprisal, Romanian BERT embeddings, reader-profile augmentation, and session-aware rolling features, all evaluated with leakage-free leave-text-out cross-validation.
