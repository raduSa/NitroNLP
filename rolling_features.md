# Rolling Session Features — Pass-2 Layer

## Motivation

All five test participants are unseen. The model cannot learn their individual
reading-speed bias during training, so pass-1 predictions for a given word rely
entirely on word-level features and genre priors. In practice, a reader who is
currently moving slowly through a passage will also tend to fixate longer on the
next word, and a reader who is skimming will continue to skim. These within-session
autocorrelations are invisible to a model that treats each word independently.

Rolling features derived from the pass-1 predictions expose this signal without
any lookahead: for word at session position *i*, the rolling window covers only
positions *0 … i−1*.

---

## Two-pass architecture

```
Pass 1  ──►  LightGBM blend (single-stage Tweedie + two-stage skip×regressor)
               │
               │  oof_blend  (train, out-of-fold)
               │  test_blend (test, batch)
               ▼
           compute_rolling_features()
               │
               │  session_word_idx, pass1_pred,
               │  roll_mean_{10,20,50}, roll_skip_{10,20,50}
               ▼
Pass 2  ──►  LightGBM (original features + rolling features)
               │
               ▼
           optimize_blend(oof_orig, oof_pass2) → w
               │
               ▼
           fit_calibration on original rows → final test_cal
```

Pass 1 is the existing pipeline, unchanged. Pass 2 is an additional LightGBM
trained on the original (non-augmented) training rows using the OOF predictions
from pass 1 as inputs. Because OOF predictions are unbiased (computed by models
that never saw those rows' texts), there is no leakage.

---

## Features added

| Column | Description |
|---|---|
| `session_word_idx` | 0-indexed position of the word within the participant's full reading session. Captures fatigue and warm-up: TRT often rises near the end of long sessions. |
| `pass1_pred` | Raw pass-1 blend prediction for this word. Gives pass 2 an explicit anchor to correct rather than re-predict from scratch. |
| `roll_mean_10/20/50` | Rolling mean of pass-1 predictions over the previous *k* words. Tracks current reading pace — a recently slow participant will likely stay slow. |
| `roll_skip_10/20/50` | Rolling fraction of pass-1 predictions below `ROLLING_SKIP_THRESHOLD` (default 50 ms) over the previous *k* words. Tracks skimming tendency. |

Windows *k* ∈ {10, 20, 50} capture short-term pace (10 words ≈ one clause),
medium-term state (20 words ≈ one sentence), and long-term session trend (50 words
≈ a paragraph). All three windows are independent features; LightGBM selects
whichever are useful.

---

## Implementation notes

- `_is_orig = True` is stamped onto every real training row before augmentation;
  synthetic rows get `False`. After the pass-1 training loop, `orig_mask` isolates
  the original rows so that pass-2 trains only on real observations.
- `compute_rolling_features` sorts by `(participant_id, doc_num, page_num,
  word_idx)` internally to ensure correct temporal ordering, then restores the
  caller's row order before returning.
- The same function is applied to both train (using `oof_blend[orig_mask]`) and
  test (using `test_blend`), so train and test distributions of rolling features
  are computed identically.
- Pass-2 uses the same `GroupKFold(by text)` and seed-bagging strategy as pass 1.
- The final `test_blend` is replaced by `w × test_blend_pass1 + (1−w) × test_c`
  where *w* is OOF-optimised. Calibration is then re-fitted on the original rows
  only (the same population as the test participants).

---

## Config flags

| Env variable | Default | Effect |
|---|---|---|
| `USE_ROLLING` | `1` | Set to `0` to skip pass 2 entirely and fall back to pass-1 calibration |
| `ROLLING_WINDOWS` | `10,20,50` | Comma-separated window sizes |
| `ROLLING_SKIP_THRESHOLD` | `50.0` | Predictions below this value (ms) count as a skip in `roll_skip_k` |
