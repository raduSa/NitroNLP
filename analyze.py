import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np

df = pd.read_csv("train_data.csv")

print("=" * 60)
print("BASIC INFO")
print("=" * 60)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

print("\n" + "=" * 60)
print("SAMPLE ROWS")
print("=" * 60)
print(df.head(10).to_string())

print("\n" + "=" * 60)
print("ANSWER COLUMN (target variable)")
print("=" * 60)
print(df["answer"].describe())
print(f"\nZero values: {(df['answer'] == 0).sum():,} ({(df['answer'] == 0).mean()*100:.1f}%)")
print(f"Non-zero values: {(df['answer'] != 0).sum():,} ({(df['answer'] != 0).mean()*100:.1f}%)")
print(f"\nValue distribution (top 10 most frequent):\n{df['answer'].value_counts().head(10)}")

print("\n" + "=" * 60)
print("PARTICIPANTS")
print("=" * 60)
print(f"Unique participants: {df['participant_id'].nunique()}")
print(f"Participant IDs: {sorted(df['participant_id'].unique())}")
print(f"\nRows per participant:\n{df['participant_id'].value_counts().describe()}")

print("\n" + "=" * 60)
print("TEXTS")
print("=" * 60)
print(f"Unique texts: {df['text'].nunique()}")
print(f"Text names:\n{df['text'].value_counts()}")

print("\n" + "=" * 60)
print("WORDS")
print("=" * 60)
print(f"Total word tokens: {len(df):,}")
print(f"Unique words (case-sensitive): {df['word'].nunique():,}")
print(f"\nMost frequent words:\n{df['word'].value_counts().head(20)}")

print("\n" + "=" * 60)
print("WORD LENGTH ANALYSIS")
print("=" * 60)
df["word_len"] = df["word"].str.len()
print(df["word_len"].describe())
print(f"\nWord length vs mean answer (fixation time):")
print(df.groupby("word_len")["answer"].mean().sort_index().to_string())

print("\n" + "=" * 60)
print("ANSWER BY PARTICIPANT")
print("=" * 60)
print(df.groupby("participant_id")["answer"].describe().to_string())

print("\n" + "=" * 60)
print("ANSWER BY TEXT")
print("=" * 60)
print(df.groupby("text")["answer"].agg(["count", "mean", "std", "min", "max"]).to_string())
