# Nitro NLP — Romanian Eye-Tracking Reading Time Prediction

End-to-end solution for the Nitro NLP hackathon: predict the Total Reading Time
(TRT, in milliseconds) that a Romanian reader will spend on each word of a
text, given only the word, its position, and the document it belongs to.

The test set contains **5 readers and 3 documents that never appear in
training**, so the model must generalise across both unseen readers and unseen
texts. The whole pipeline is engineered around that constraint.

## Task summary

| Item | Value |
|---|---|
| Target | `answer` — TRT in ms (0 if the word was skipped) |
| Train rows | 135,210 (30 readers × 9 texts) |
| Test rows | 9,425 (5 readers × 3 texts, all unseen) |
| Metric | `100 * (max(0, R²) + |Pearson|) / 2` |
| Output format | `subtaskID, datapointID, answer` (CSV) |

A constant predictor scores 0 (R² = 0, Pearson = NaN). Both ranking quality and
absolute calibration matter, because the metric mixes correlation with R².

---

## The pipeline, step by step

### Step 1 — Describe each word with cheap, hand-built numbers

For every word we compute:

- letter count, syllable count
- is it a punctuation mark, a URL, a number?
- does it start with a capital letter (proper noun!)
- where does it sit on the page (1st word? last? middle?)

These are dirt cheap and capture the "how long is the word" signal.
Already gets you a non-trivial baseline.

### Step 2 — Frequency

We count every word's occurrences across the whole corpus and take
`log(count / total)`. Common words → high log-freq → low reading time.
Rare words → low log-freq → long reading time.

We also attach the **previous and next word's frequency** to each row,
because reading-time research shows you read a word faster when its
neighbours are common (your brain processes ahead).

### Step 3 — Borrow knowledge from training readers

For each unique Romanian token, we look at all training readers and compute:
average TRT, median, standard deviation, skip rate. Now when "de" shows
up in the test set, we can hand the model "the average reader spent ~73 ms
on this word last time we saw it" — even though the test reader is brand new.

There's a subtle trap here: when we compute these averages for *training*
rows, we have to exclude the row's own text from the average, otherwise the
model "cheats" by seeing future targets. We do **leave-text-out** averages
— same logic as the test split.

### Step 4 — Surprisal from a Romanian language model

This is the killer feature. We feed each page of text into a Romanian GPT-2
and ask it: *"at every position, how surprised are you to see this word
given everything before it?"*. Mathematically that's `−log p(word | context)`.
Ranges from ~0 (totally expected) to ~20+ (totally unexpected).

Eye-tracking research from the last decade shows this single number explains
most of the *contextual* variance in reading time — exactly what hand-built
features can't capture. A common word like "vacă" might still be slow if it
appears somewhere it doesn't belong; surprisal catches that.

The pipeline tries a fallback chain of Romanian causal LMs in order
(`RoGPT2-large` → `RoGPT2-medium` → `RoGPT2-base` → `gpt-neo-romanian-780m`
→ `gpt-neo-romanian-125m`) and uses whichever one loads first.

### Step 5 — Train a tree-based model

Throw all those features into LightGBM (gradient-boosted trees). It's the
workhorse for tabular data: handles missing values, interactions,
non-linearities, and trains in seconds.

We use the **Tweedie loss** instead of regular squared-error because our
target has a weird shape: a fat spike at zero (skips) and a long right tail
(occasional 2-second stares). Tweedie was literally designed for that
distribution shape.

We bag **3 seeds × 5 folds = 15 models** and average. Bagging just reduces
variance; trees are noisy individually, an ensemble of them is calmer.

The cross-validation is a `GroupKFold` by `text`, mirroring the unseen-text
structure of the official test split. If we used random-shuffle CV the
scores would lie to us.

### Step 6 — Calibration (the metric trick)

The competition metric is `(R² + |Pearson|) / 2`. Two pieces:

- **Pearson** only cares about ranking — does the model put high-TRT words
  above low-TRT ones?
- **R²** also cares about magnitude — are the predictions actually
  *centred* around the true ms numbers?

A tree model with Tweedie loss is great at ranking but its absolute scale
can drift (Tweedie is multiplicative). So after training, we do one tiny
linear regression: `final = a * pred + b`, fitted on out-of-fold
predictions. This stretches the predictions to match the true scale without
changing their ranking.

Pearson is unchanged, R² goes up, score goes up. Free points.

### Step 7 — Write the submission, paranoid mode

Before saving the file, we hard-assert seven things:

1. column order is exactly `subtaskID, datapointID, answer`
2. row count matches the test set
3. every test `datapointID` is present and only once
4. rows are sorted by `datapointID` ascending
5. `subtaskID` is 1 on every row
6. no NaN values, no negative answers
7. not all answers identical (a constant predictor scores 0)

If any of those breaks, the script raises an exception **instead of silently
writing a bad file**. Means you can't submit garbage by accident.

---

## How to run

### On Kaggle (recommended — uses the P100 GPU for surprisal)

1. Create a new notebook. Settings → **Accelerator: GPU P100**, **Internet: ON**.
2. Attach a Kaggle dataset that contains `train_data.csv` and `test_data.csv`
   in the same folder. The script auto-discovers it via
   `/kaggle/input/**/train_data.csv`.
3. In the first cell:
   ```python
   !pip -q install lightgbm transformers accelerate
   ```
4. In the next cell, paste the contents of `solution.py` (or upload it as a
   utility script and `%run solution.py`).
5. `submission.csv` is written to `/kaggle/working/submission.csv`.

### Locally (no GPU, fast smoke-test without the LM)

```bash
USE_LM=0 python3 solution.py
```

This skips the Romanian-LM surprisal feature and runs the rest of the
pipeline. Useful for verifying the feature engineering and submission format
without waiting for the LM.

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `USE_LM` | `1` | Set to `0` to skip the LM surprisal step. |
| `LM_CANDIDATES` | RoGPT2 chain | Comma-separated list of HuggingFace causal-LM IDs. |
| `LM_MAX_LEN` | `1024` | Max tokens per LM forward pass; longer pages use a sliding window. |
| `N_FOLDS` | `5` | Number of GroupKFold folds (capped at the number of texts). |
| `N_SEEDS` | `3` | Number of LightGBM seeds to bag per fold. |

---

## Files

- `solution.py` — the entire pipeline, single file.
- `train_data.csv` — provided. 135,210 rows, columns: `word_id, word, answer, participant_id, text`.
- `test_data.csv` — provided. 9,425 rows, columns: `word_id, word, participant_id, text, datapointID`.
- `submission.csv` — written by the script after all self-checks pass.

---

## Expected output

A successful run prints something like:

```
[cfg] INPUT_DIR=...  OUTPUT_DIR=...  USE_LM=True  N_SEEDS=3
[data] train=(135210, 5)  test=(9425, 5)
[lm] loaded readerbench/RoGPT2-large on cuda
[lm] computing surprisal for train (...)
...
[feat] 34 features, NaNs train/test = 0/0
[fold 0] held-out=['lit_solaris']  score=41.034
[fold 1] held-out=['popsci_multipleye']  score=37.594
[fold 2] held-out=['arg_pisarapanui', 'enc_wikimoon', 'lit_northwind']  score=36.699
[fold 3] held-out=['ins_humanrights', 'lit_brokenapril']  score=38.587
[fold 4] held-out=['lit_magicmountain', 'popsci_caveman']  score=38.704
[cv] folds: mean=38.524  std=1.451
[cv] global OOF score (raw)        = 38.843
[calib] a=1.0596  b=-8.3879
[cv] global OOF score (calibrated) = 38.910
[pred] test pred stats: min=11.7  mean=275.1  max=932.7  std=163.5  #zeros=0
[out] wrote .../submission.csv  rows=9425
[out] all self-checks passed.
```

---

## Where the score comes from, intuitively

Three families of signal flow into one strong tabular model, then a tiny
calibration straightens the scale:

```
       word            "context"           reader-stats
        |                  |                     |
 [length, syllables,  [LM surprisal,   [tok_mean from
  punct flag, ...]    prev/next freq]   training readers]
        |                  |                     |
        +----+   +---------+         +-----------+
             |   |                   |
             v   v                   v
        +---------------------------------+
        | LightGBM (15 seed-bagged trees) |
        +---------------------------------+
                       |
                       v
            raw prediction in ms
                       |
                       v
        a*x + b   (linear calibration)
                       |
                       v
            final prediction → submission
```

Every piece earns its keep. Words are described by what they *are*
(length, syllables), what's *around* them (neighbour frequency, LM context),
and what other readers *did* with them (per-token target stats). The model
learns the non-linear combination; calibration handles scale.
