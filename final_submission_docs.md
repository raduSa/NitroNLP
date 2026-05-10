# final_submission.py — Romanian Eye-Tracking Reading Time Prediction

End-to-end solution for the Nitro NLP hackathon: predict Total Reading Time
(TRT, in milliseconds) that a Romanian reader will spend on each word, given
only the word, its position, and the document it belongs to.

The test set contains **5 readers and 3 documents that never appear in
training**, so the model must generalise across both unseen readers and unseen
texts. Every design decision is built around that constraint.

## Task summary

| Item | Value |
|---|---|
| Target | `answer` — TRT in ms (0 if the word was skipped) |
| Train rows | 135,210 (30 readers × 9 texts) |
| Test rows | 9,425 (5 readers × 3 texts, all unseen) |
| Metric | `100 * (max(0, R²) + |Pearson|) / 2` |
| Output format | `subtaskID, datapointID, answer` (CSV) |

A constant predictor scores 0 (R² = 0, Pearson = NaN). Both ranking quality
and absolute calibration matter, because the metric mixes correlation with R².

---

## The pipeline, step by step

### Step 1 — Word-level surface features

For every word we compute:

- `word_len`, `alpha_len` (length with and without punctuation)
- `n_syllables` (vowel-group count for Romanian)
- `is_punct`, `is_url`, `has_digit`
- `is_upper_first`, `is_all_upper` (proper noun / acronym signals)
- `ends_with_punct` (sentence boundary hint)
- `doc_num`, `page_num`, `word_idx` parsed from the `word_id` field
- `pos_in_page`, `page_size`, `rel_pos_in_page`

We also attach the **previous and next word's** surface stats to each row,
because the difficulty of a word is partly determined by what surrounds it.

### Step 2 — Corpus and external frequency

**Corpus frequency** counts every token across train + test and stores
`log(count / total)`. Rare words → long reading time. Previous and next
words' log-freq are added as neighbour features.

**Wordfreq / Zipf frequency** (`zipf_frequency` from the `wordfreq` library,
Romanian word list) gives a second, externally calibrated frequency signal
and its prev/next neighbours. Used only when the library is installed;
the rest of the pipeline degrades gracefully without it.

### Step 3 — Genre features

Each text name begins with a genre prefix (`lit_`, `popsci_`, `arg_`, `enc_`,
`ins_`). We extract it and create:

- `genre_id` — integer index of the genre
- `genre_enc` — leave-text-out mean TRT for the genre (computed without the
  current text to avoid leakage; test rows use the full training mean)

This tells the model that, e.g., literary texts are read more slowly than
encyclopaedic ones, even before seeing a single word.

### Step 4 — Per-word target statistics (leave-text-out)

For each unique lowercased token we compute across training readers:
mean TRT, median, standard deviation, occurrence count, skip rate.

The subtle trap: when computing these statistics for a training row, we
exclude every row from the same text (`leave-text-out`). This mirrors what
the model will see at inference time — where the word appeared in unseen
documents — and prevents target leakage.

For the test set, all 30 readers contribute because those texts are
completely held out.

### Step 5 — Augmentation

**Participant mixup.** For each (text, page, word position) triplet we draw
random pairs of readers and linearly interpolate their TRT values and any
numeric participant-level features. A synthetic "average reader" row (mean
over all readers at that position) is also added. This doubles the effective
diversity of reader profiles visible during training.

**Genre average reader.** For each (genre, lowercased word) pair we add
a synthetic row whose TRT is the mean over all training occurrences of that
word in that genre. It teaches the model the genre-conditioned word baseline
without requiring the model to see the exact same text again.

Augmented rows are tracked with an `_orig` flag so that the rolling pass
(Step 9) is trained only on original rows whose pass-1 predictions are
leakage-free.

### Step 6 — Surprisal from a Romanian language model

We feed each page of text into a Romanian GPT-2 and record
`−log p(word | context)` at every position. Ranges from ~0 (totally expected)
to ~20+ (totally unexpected).

Eye-tracking research shows surprisal explains most contextual variance in
reading time — a common word is still slow when it appears where it doesn't
belong. Surface features and frequency alone cannot capture this.

The pipeline tries a fallback chain:
`RoGPT2-large → RoGPT2-medium → RoGPT2-base → gpt-neo-romanian-780m → gpt-neo-romanian-125m`.

Previous and next surprisal values are also attached as neighbour features.

### Step 7 — Contextualised embeddings from Romanian BERT

We load `dumitrescustefan/bert-base-romanian-cased-v1` (masked LM) and for
each word:

- **MLM log-probability** — how surprised the bidirectional model is to see
  this word given its full left and right context (`mlm_logp_first`,
  `mlm_logp_sum` over sub-words). Unlike causal surprisal, this uses future
  context too.
- **Sub-word count** (`n_bert_subwords`) — long / rare words split into more
  pieces; the model can use this as a morphological complexity proxy.
- **Contextualised embedding** — last-four-layer average pooled over the
  word's sub-tokens, compressed to `BERT_PCA_DIM` (default 24) principal
  components. Each PC becomes a feature.

Processing is done once on unique `word_id`s (not duplicated per reader)
to minimise GPU time. Previous and next `mlm_logp_first` are added as
neighbour features.

### Step 8 — Two-stage LightGBM (skip gate + non-zero regressor)

TRT has a zero-inflated distribution: many words are skipped (TRT = 0) and
the rest have a long right tail. A single model can't handle both shapes well.

**Stage 1 — Tweedie regressor.** LightGBM with Tweedie loss (designed for
zero-inflated right-skewed targets). Trained on all rows including zeros.

**Stage 2 — Binary skip classifier + log1p regressor.**
- A binary classifier predicts `P(skip)`.
- A separate regression model is trained only on non-zero rows using
  `log1p(TRT)` as the target (then `expm1`-transformed back), giving it a
  more symmetric loss landscape.
- The two are combined as `(1 − P(skip)) × regressor_prediction`.

The final prediction is a blend `w × stage1 + (1−w) × stage2`, where `w`
is chosen by grid search over OOF predictions to maximise the competition
metric.

Both stages use **GroupKFold by text** (N_FOLDS=5), mirroring the unseen-text
test structure. Random-shuffle CV would overestimate performance.
Each fold trains **N_SEEDS=3** LightGBM models with different random seeds
and averages them (bagging to reduce variance).

LGB base params: `learning_rate=0.05`, `num_leaves=127`,
`feature_fraction=0.9`, `bagging_fraction=0.9`.

### Step 9 — Rolling session features (pass 2)

Reading is sequential: how tired or fluent a reader is 50 words in affects
how they read word 51. After pass-1 predictions are available we compute
per-participant rolling statistics:

- `session_word_idx` — absolute word position in the participant's reading
  session (across pages)
- `pass1_pred` — the blended pass-1 prediction for this word
- `roll_mean_{k}` — rolling mean of the *previous* k pass-1 predictions
  (windows: 10, 20, 50 words)
- `roll_skip_{k}` — rolling fraction of near-zero predictions in the
  previous k words (skip-rate estimate)

These features are added to both train and test. A new GroupKFold LGB is
then trained on the **original** (non-augmented) rows using all base features
plus rolling features.

The pass-2 OOF predictions are blended with the pass-1 blend using a second
weight `w2` optimised on the competition metric.

### Step 10 — Linear calibration

The competition metric mixes Pearson (ranking) and R² (scale). Tree models
with Tweedie loss rank well but their absolute scale can drift. After
training, a tiny linear regression `final = a × pred + b` is fitted on
OOF predictions to stretch the output to match the true ms distribution.

Pearson is unchanged; R² improves. Free points.

### Step 11 — Write the submission, paranoid mode

Before saving, seven assertions fire:

1. Columns are exactly `subtaskID, datapointID, answer` in that order
2. Row count matches test set
3. Every test `datapointID` is present exactly once
4. Rows are sorted by `datapointID` ascending
5. `subtaskID` is 1 on every row
6. No NaN values, no negative answers
7. Answers are not all identical (constant predictor scores 0)

Any failure raises an exception instead of silently writing a bad file.

---

## How to run

### On Kaggle (recommended — uses the P100 GPU for LM and BERT)

1. Create a new notebook. Settings → **Accelerator: GPU P100**, **Internet: ON**.
2. Attach a Kaggle dataset containing `train_data.csv` and `test_data.csv`
   in the same folder.
3. In the first cell:
   ```python
   !pip -q install lightgbm transformers accelerate wordfreq
   ```
4. In the next cell, paste the contents of `final_submission.py` or upload
   it as a utility script and `%run final_submission.py`.
5. `submission.csv` is written to `/kaggle/working/submission.csv`.

### Locally (no GPU, smoke-test without LM/BERT)

```bash
USE_LM=0 USE_BERT=0 python final_submission.py
```

This skips both neural feature steps. Useful for verifying feature
engineering and submission format.

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `USE_LM` | `1` | Set to `0` to skip causal-LM surprisal. |
| `USE_BERT` | `1` | Set to `0` to skip BERT features. |
| `USE_TWO_STAGE` | `1` | Set to `0` to use Tweedie stage only. |
| `USE_ROLLING` | `1` | Set to `0` to skip pass-2 rolling features. |
| `LM_CANDIDATES` | RoGPT2 chain | Comma-separated HuggingFace causal-LM IDs. |
| `LM_MAX_LEN` | `1024` | Max tokens per LM forward pass (sliding window beyond). |
| `BERT_NAME` | `dumitrescustefan/bert-base-romanian-cased-v1` | BERT checkpoint. |
| `BERT_MAX_LEN` | `512` | Max tokens per BERT chunk. |
| `BERT_BATCH` | `32` | Batch size for masked-LM inference. |
| `BERT_PCA_DIM` | `24` | Number of PCA components from BERT embeddings. |
| `N_FOLDS` | `5` | GroupKFold folds (capped at number of texts). |
| `N_SEEDS` | `3` | LightGBM seeds to bag per fold. |
| `ROLLING_WINDOWS` | `10,20,50` | Rolling window sizes for pass-2 features. |
| `ROLLING_SKIP_THRESHOLD` | `50.0` | Predicted TRT below this is counted as a skip. |

---

## Files

- `final_submission.py` — the entire pipeline, single file.
- `train_data.csv` — provided. 135,210 rows, columns: `word_id, word, answer, participant_id, text`.
- `test_data.csv` — provided. 9,425 rows, columns: `word_id, word, participant_id, text, datapointID`.
- `submission.csv` — written by the script after all self-checks pass.

---

## What changed from `original_improved_solution.py`

The original version had three flags (`USE_LM`, `USE_BERT`, `USE_TWO_STAGE`)
and a straightforward pipeline: word features → frequency → per-word stats →
surprisal → BERT → two-stage LGB → blend → calibrate.

`final_submission.py` adds five significant layers on top of that:

| Addition | Details |
|---|---|
| `USE_ROLLING` flag | Enables pass-2 session-aware training |
| Genre features | `genre_id` + leave-text-out `genre_enc` from text name prefix |
| Wordfreq / Zipf | External frequency signal (`zipf_freq`) + prev/next neighbours |
| Participant mixup augmentation | Interpolated reader pairs + "average reader" synthetic rows |
| Genre average reader augmentation | Per-(genre, word) mean TRT synthetic rows |
| `compute_rolling_features()` | `session_word_idx`, `pass1_pred`, `roll_mean_k`, `roll_skip_k` |
| Pass-2 training loop | Separate GroupKFold LGB trained on base + rolling features |
| Pass-2 blend | `w2` optimised between pass-1 blend and pass-2 OOF |
| `_LGB_BASE` dict | Consolidates shared LGB params (LR=0.05, 127 leaves) |
| `_orig` flag | Tracks original vs synthetic rows so rolling pass trains clean |

The LGB hyperparameters themselves are identical to the original
(`learning_rate=0.05`, `num_leaves=127`, `min_data_in_leaf=200`).
The gains come from richer features, more training data via augmentation,
and the second training pass that can condition on session dynamics.

---

## Architecture diagram

```
       word surface        corpus / zipf       per-word stats
           |                    |                    |
  [len, syllables,       [log_freq,          [tok_mean, tok_skip_rate,
   is_punct, ...]     prev/next zipf_freq]    leave-text-out stats]
           |                    |                    |
           +--------+-----------+          +---------+
                    |                      |
                    v                      v
         genre features          surprisal (RoGPT2)
         [genre_id,              [surprisal,
          genre_enc]              prev/next surp.]
                    |                      |
                    +----------+-----------+
                               |
                        BERT features
                        [mlm_logp_*, bert_pc_0..23]
                               |
                               v
                     AUGMENTATION
                  [mixup + average_reader
                   + genre_average_reader]
                               |
                 +-------------+-------------+
                 |                           |
       Stage 1: Tweedie LGB      Stage 2: clf × log1p LGB
                 |                           |
                 +--------[blend w]----------+
                               |
                         pass-1 blend OOF
                               |
                   compute_rolling_features()
                   [session_word_idx,
                    pass1_pred,
                    roll_mean_10/20/50,
                    roll_skip_10/20/50]
                               |
                 Pass-2 LGB (base + rolling features)
                               |
                   [blend w2 pass-1 ↔ pass-2]
                               |
                    linear calibration (a*x + b)
                               |
                       submission.csv
```
