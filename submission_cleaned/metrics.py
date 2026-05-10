from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def comp_metric(y_true, preds):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds,  dtype=float)
    if np.std(preds) < 1e-12:
        return 0.0
    r2 = max(0.0, r2_score(y_true, preds, force_finite=True))
    pears = pearsonr(y_true, preds)[0]
    return 100.0 * (abs(pears if not np.isnan(pears) else 0.0) + r2) / 2.0


def fit_calibration(y_true, y_pred):
    if np.std(y_pred) < 1e-9:
        return 1.0, 0.0
    a = np.cov(y_pred, y_true, ddof=0)[0, 1] / np.var(y_pred)
    return float(a), float(y_true.mean() - a * y_pred.mean())


def optimize_blend(y_true, pred1, pred2):
    best_w, best_score = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 51):
        score = comp_metric(y_true, w * pred1 + (1 - w) * pred2)
        if score > best_score:
            best_score, best_w = score, float(w)
    return best_w
