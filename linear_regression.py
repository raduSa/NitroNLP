import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
from scipy.linalg import lstsq

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv("train_data.csv")
test  = pd.read_csv("test_data.csv")

# ── Feature engineering ──────────────────────────────────────────────────────
# word length
train["word_len"] = train["word"].str.len()
test["word_len"]  = test["word"].str.len()

# text mean-encoding: map each text name → mean answer in training
text_means = train.groupby("text")["answer"].mean()
global_mean = train["answer"].mean()

train["text_enc"] = train["text"].map(text_means)
# unseen texts in test fall back to the global training mean
test["text_enc"]  = test["text"].map(text_means).fillna(global_mean)

# ── Build matrices ───────────────────────────────────────────────────────────
FEATURES = ["word_len", "text_enc"]

X_train = np.column_stack([np.ones(len(train)), train[FEATURES].values])
y_train = train["answer"].values.astype(float)

X_test  = np.column_stack([np.ones(len(test)),  test[FEATURES].values])

# ── Fit with scipy lstsq ─────────────────────────────────────────────────────
coeffs, residuals, rank, sv = lstsq(X_train, y_train)

intercept, coef_word_len, coef_text_enc = coeffs
print("=" * 55)
print("MODEL COEFFICIENTS")
print("=" * 55)
print(f"  Intercept  : {intercept:>10.4f}")
print(f"  word_len   : {coef_word_len:>10.4f}  (ms per extra character)")
print(f"  text_enc   : {coef_text_enc:>10.4f}  (ms per ms of text mean)")

# ── Training-set evaluation ──────────────────────────────────────────────────
y_pred_train = X_train @ coeffs

ss_res = np.sum((y_train - y_pred_train) ** 2)
ss_tot = np.sum((y_train - y_train.mean()) ** 2)
r2     = 1 - ss_res / ss_tot
mae    = np.mean(np.abs(y_train - y_pred_train))
rmse   = np.sqrt(np.mean((y_train - y_pred_train) ** 2))

print("\n" + "=" * 55)
print("TRAINING SET PERFORMANCE")
print("=" * 55)
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : {mae:.2f} ms")
print(f"  RMSE : {rmse:.2f} ms")

# ── Test-set predictions ─────────────────────────────────────────────────────
test["predicted_answer"] = X_test @ coeffs

print("\n" + "=" * 55)
print("TEST SET PREDICTIONS (first 10 rows)")
print("=" * 55)
print(test[["word_id", "word", "text", "word_len", "predicted_answer"]].head(10).to_string(index=False))

print(f"\nPrediction stats:")
print(test["predicted_answer"].describe())

# ── Save predictions ─────────────────────────────────────────────────────────
out = test[["datapointID", "predicted_answer"]].rename(columns={"predicted_answer": "answer"})
out.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")
