import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
train = pd.read_csv(BASE / "train_data.csv")
test  = pd.read_csv(BASE / "test_data.csv")

# helper
def check(label, actual, claimed, fmt=","):
    tag = "OK" if actual == claimed else "MISMATCH"
    av = f"{actual:{fmt}}" if fmt == "," else str(actual)
    cv = f"{claimed:{fmt}}" if fmt == "," else str(claimed)
    print(f"  [{tag}] {label}: actual={av}  claimed={cv}")

# train
print("=" * 60)
print("TRAIN")
print("=" * 60)
n_train_rows        = len(train)
n_train_readers     = train["participant_id"].nunique()
n_train_texts       = train["text"].nunique()
train_genres        = sorted(train["text"].str.extract(r"^([a-z]+)_", expand=False).dropna().unique())
rows_per_reader     = train.groupby("participant_id").size()
rows_per_text       = train.groupby("text").size()

print(f"  Rows            : {n_train_rows:,}")
print(f"  Participants    : {n_train_readers}")
print(f"  Texts           : {n_train_texts}")
print(f"  Text names      : {sorted(train['text'].unique())}")
print(f"  Genres detected : {train_genres}")
print(f"  Rows/reader     : min={rows_per_reader.min():,}  max={rows_per_reader.max():,}  mean={rows_per_reader.mean():.0f}")
print(f"  Rows/text       : min={rows_per_text.min():,}  max={rows_per_text.max():,}  mean={rows_per_text.mean():.0f}")
print(f"  Columns         : {list(train.columns)}")
print()
print("  Fact-checking main.tex claims:")
check("train rows",        n_train_rows,    135_210)
check("train readers",     n_train_readers, 30)
check("train texts",       n_train_texts,   9)

# test
print()
print("=" * 60)
print("TEST")
print("=" * 60)
n_test_rows     = len(test)
n_test_readers  = test["participant_id"].nunique()
n_test_texts    = test["text"].nunique()
overlap_readers = set(test["participant_id"].unique()) & set(train["participant_id"].unique())
overlap_texts   = set(test["text"].unique())           & set(train["text"].unique())

print(f"  Rows            : {n_test_rows:,}")
print(f"  Participants    : {n_test_readers}")
print(f"  Texts           : {n_test_texts}")
print(f"  Text names      : {sorted(test['text'].unique())}")
print(f"  Columns         : {list(test.columns)}")
print(f"  Reader overlap with train : {overlap_readers if overlap_readers else 'none'}")
print(f"  Text overlap with train   : {overlap_texts   if overlap_texts   else 'none'}")
print()
print("  Fact-checking main.tex claims:")
check("test rows",        n_test_rows,    9_425)
check("test readers",     n_test_readers, 5)
check("test texts",       n_test_texts,   3)
check("reader overlap",   len(overlap_readers), 0)
check("text overlap",     len(overlap_texts),   0)

# target distribution
print()
print("=" * 60)
print("TARGET (train)")
print("=" * 60)
skip_n    = int((train["answer"] == 0).sum())
skip_pct  = 100 * skip_n / n_train_rows
print(f"  Zero (skipped)  : {skip_n:,}  ({skip_pct:.1f}%)")
print(f"  Non-zero        : {n_train_rows - skip_n:,}  ({100-skip_pct:.1f}%)")
print(f"  Mean TRT        : {train['answer'].mean():.1f} ms")
print(f"  Median TRT      : {train['answer'].median():.1f} ms")
print(f"  Max TRT         : {train['answer'].max():.1f} ms")
