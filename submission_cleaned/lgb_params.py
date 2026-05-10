from __future__ import annotations

_LGB_BASE = dict(
    learning_rate = 0.05,
    num_leaves = 127,
    feature_fraction = 0.9,
    bagging_fraction = 0.9,
    bagging_freq = 5,
    verbosity = -1,
)


def lgb_params_reg(seed: int) -> dict:
    return {
        **_LGB_BASE,
        'objective': 'tweedie',
        'tweedie_variance_power': 1.4,
        'metric': 'rmse',
        'min_data_in_leaf': 200,
        'seed': seed,
    }


def lgb_params_clf(seed: int) -> dict:
    return {
        **_LGB_BASE,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'min_data_in_leaf': 200,
        'seed': seed,
    }


def lgb_params_pos(seed: int) -> dict:
    return {
        **_LGB_BASE,
        'objective': 'regression',
        'metric': 'rmse',
        'min_data_in_leaf': 100,
        'seed': seed,
    }
