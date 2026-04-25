import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import re
import numpy as np
import pandas as pd
from wordfreq import word_frequency
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from eval_metric import eval_metric

# ── Load ─────────────────────────────────────────────────────────────────────
train = pd.read_csv("train_data.csv")
test  = pd.read_csv("test_data.csv")

# ── Feature engineering ──────────────────────────────────────────────────────
def parse_word_id(wid):
    """Return (page_num, word_idx) from word_id like 'enc_wikimoon_13_page_1_42'."""
    parts = wid.split("_page_")
    if len(parts) == 2:
        tail = parts[1].split("_")
        return int(tail[0]), int(tail[-1])
    return 0, 0

def build_features(df, text_means, participant_means, global_mean):
    df = df.copy()

    # --- word-level scalar features ---
    df["word_len"]      = df["word"].str.len()
    df["word_freq"]     = df["word"].str.lower().apply(lambda w: word_frequency(w, "ro"))
    df["log_word_freq"] = np.log1p(df["word_freq"])

    # --- surface type flags ---
    df["is_punct"]      = df["word"].apply(lambda w: bool(re.fullmatch(r'[^\w\s]+', w))).astype(int)
    df["is_url"]        = df["word"].str.startswith("http").astype(int)
    df["is_number"]     = df["word"].apply(lambda w: bool(re.match(r'^\d', w))).astype(int)
    df["is_capitalized"]= df["word"].apply(lambda w: w[0].isupper() if w else False).astype(int)

    # --- context group from word_id ---
    parsed           = df["word_id"].apply(parse_word_id)
    df["page_num"]   = parsed.apply(lambda x: x[0])
    df["word_idx"]   = parsed.apply(lambda x: x[1])
    df["ctx_key"]    = (df["text"] + "_" + df["participant_id"].astype(str)
                        + "_" + df["page_num"].astype(str))

    df = df.sort_values(["ctx_key", "word_idx"])

    # --- positional features within page ---
    df["page_len"]       = df.groupby("ctx_key")["word_idx"].transform("count")
    df["word_pos_norm"]  = df["word_idx"] / df["page_len"].clip(lower=1)

    # --- spillover: previous word length ---
    df["prev_word_len"]  = (df.groupby("ctx_key")["word_len"]
                              .shift(1)
                              .fillna(df["word_len"].mean()))

    # --- mean encodings (fall back to global mean for unseen categories) ---
    df["text_enc"]        = df["text"].map(text_means).fillna(global_mean)
    df["participant_enc"] = df["participant_id"].map(participant_means).fillna(global_mean)

    return df

print("Building features...")
text_means        = train.groupby("text")["answer"].mean()
participant_means = train.groupby("participant_id")["answer"].mean()
global_mean       = train["answer"].mean()

train = build_features(train, text_means, participant_means, global_mean)
test  = build_features(test,  text_means, participant_means, global_mean)

FEATURES = [
    "word_len", "word_freq", "log_word_freq",
    "is_punct", "is_url", "is_number", "is_capitalized",
    "page_len", "word_pos_norm", "prev_word_len",
    "text_enc", "participant_enc",
]

# ── Train / validation split (by context group, no sentence leakage) ─────────
ctx_keys = train["ctx_key"].unique()
rng = np.random.default_rng(42)
rng.shuffle(ctx_keys)
split      = int(0.8 * len(ctx_keys))
train_ctxs = set(ctx_keys[:split])
val_ctxs   = set(ctx_keys[split:])

tr  = train[train["ctx_key"].isin(train_ctxs)]
val = train[train["ctx_key"].isin(val_ctxs)]

X_tr,  y_tr  = tr[FEATURES].values,  tr["answer"].values.astype(float)
X_val, y_val = val[FEATURES].values, val["answer"].values.astype(float)
X_test       = test[FEATURES].values

print(f"Train: {len(tr):,}  |  Val: {len(val):,}  |  Test: {len(test):,}\n")

# ── Model definitions ────────────────────────────────────────────────────────
def scaled(model):
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("model", model)])

models = {
    "Linear Regression" : scaled(LinearRegression()),
    "Ridge Regression"  : scaled(Ridge(alpha=10.0)),
    "Linear SVR"        : scaled(LinearSVR(C=1.0, max_iter=5000)),
    "Random Forest"     : RandomForestRegressor(n_estimators=200, max_depth=12,
                                                 min_samples_leaf=5, random_state=42,
                                                 n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                     learning_rate=0.05,
                                                     subsample=0.8, random_state=42),
    "Neural Network"    : scaled(MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                              activation="relu", max_iter=400,
                                              early_stopping=True, validation_fraction=0.1,
                                              random_state=42)),
}

# ── Train, evaluate, report ──────────────────────────────────────────────────
print(f"{'Model':<22} {'Eval Score':>11} {'R²':>8} {'Pearson':>9}")
print("-" * 55)

best_score, best_name, best_model = -np.inf, None, None

for name, model in models.items():
    print(f"  training {name}...", end="\r")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    score   = eval_metric(y_val, preds)
    r2      = max(0.0, r2_score(y_val, preds))
    pearson = float(np.abs(pearsonr(y_val, preds)[0]))

    print(f"{name:<22} {score:>11.2f} {r2:>8.4f} {pearson:>9.4f}")

    if score > best_score:
        best_score, best_name, best_model = score, name, model

print("-" * 55)
print(f"\nBest: {best_name}  (score={best_score:.2f})")

# ── Retrain best model on full training set, predict test ────────────────────
print(f"\nRetraining {best_name} on full training data...")
best_model.fit(train[FEATURES].values, train["answer"].values.astype(float))
test["predicted_answer"] = best_model.predict(X_test)

out = test[["datapointID", "predicted_answer"]].rename(columns={"predicted_answer": "answer"})
out.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
