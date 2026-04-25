# Changes to baseline_solution.py

## 1. Genre features (`add_genre_features`)

The text name encodes its genre as a prefix (`lit_`, `popsci_`, `arg_`, `ins_`, `enc_`).
Two new features are derived from this and appended to `FEATURE_COLS`:

**`genre_id`** — integer label for the genre (stable across train and test via sorted
enumeration). Lets LightGBM split directly on genre identity.

**`genre_enc`** — mean TRT for all words in that genre.
On train rows a leave-text-out scheme is used (the genre mean is computed from all texts
in the genre *except* the current one) to match the leakage discipline already applied by
`add_per_word_target_stats`. On test rows the full-train genre mean is used.

**Why this helps over the existing `tok_mean`/`tok_median`:** all three test texts belong
to genres present in training (`arg`, `ins`, `lit`), but are themselves unseen. Without
genre encoding their text-level signal falls back to the global mean. `genre_enc` gives
a genre-specific prior instead, narrowing the prediction range before any word-level
features are considered.

---

## 2. Participant Mixup augmentation (`augment_participant_mixup`)

**Motivation:** all five test participants are unseen. The model must generalise to
unknown readers entirely from word-level features and the global-mean fallback for
`participant_enc`. Mixup explicitly extends the training distribution of
(participant reading speed, TRT) pairs beyond the 30 observed readers.

**Mechanism:** words are grouped by position `(text, page_num, word_idx)`. Within each
group all ~30 participant rows share identical word-level features; only `answer`,
`participant_enc`, `doc_num`, and `surprisal` (when present) differ. For each group,
`round(n × multiplier)` random participant pairs are drawn and their interpolatable
columns are blended with a random α ~ Uniform(0, 1):

```
answer_synth        = α · answer_p1        + (1−α) · answer_p2
participant_enc_syn = α · participant_enc1 + (1−α) · participant_enc2
```

All other columns (word length, frequency, position, genre, etc.) are taken from the
anchor row unchanged. With `multiplier=1.0` this adds ~135 k synthetic rows, doubling
the training set.

---

## 3. Average reader augmentation (part of `augment_participant_mixup`)

Enabled by default via `add_average_reader=True`.

For each word position one additional row is appended whose interpolatable columns are
the group mean across all 30 participants. This creates an explicit training example
representing a "prototypical" reader — the scenario closest to what the model will see
at test time, where `participant_enc` falls back to the global mean. The signal
partially overlaps with `tok_mean` but correctly pairs the mean TRT with the global
`participant_enc` value rather than leaving that association implicit.

Adds ~4,500 rows (one per unique word position).

---

## 4. Genre average reader augmentation (`augment_genre_average_reader`)

**Motivation:** the per-word-position average reader inside `augment_participant_mixup`
averages across participants for one specific text occurrence. This complementary
augmentation averages across both participants *and* texts within the same genre, giving
the model a genre-conditioned signal for every word it has seen in that genre.

All three test texts are in genres present in training (`arg`, `ins`, `lit`). A word
that appears in multiple training texts of the same genre accumulates a genre-specific
mean TRT that is more informative than the global average. These rows teach the model:
"when participant_enc is at the global mean and you are reading this word in this genre
context, expect this TRT."

**Mechanism:** training rows are grouped by `(genre, lowercase-word-form)`. For each
group with at least two observations one synthetic row is appended:

```
answer          = mean TRT of that word across all participants and all texts in genre
participant_enc = global mean of participant_enc (when column is present)
```

All other word-level features are taken from a representative real row in the group.
`participant_enc` is only written when the column already exists in the DataFrame —
the baseline pipeline does not create it, so the assignment is safely skipped.

Adds ~2,185 rows (one per unique word type per genre).

---

## Summary of row counts (approximate, with defaults)

| Source | Rows |
|---|---|
| Original training data | 135,210 |
| Participant mixup | +135,000 |
| Average reader (per word position) | +4,507 |
| Genre average reader (per word × genre) | ~2,185 |
| **Total** | **~276,900** |
