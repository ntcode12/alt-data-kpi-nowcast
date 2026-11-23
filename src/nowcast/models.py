"""Forecast models (DESIGN.md §7.3). Each fits on a training frame and predicts one target row.

Training frames carry columns y, y_lag, x (and dy, dx for M3), all known at the prediction's as-of date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def m0_naive(row) -> float:
    """Growth persists: y_hat = y_{q-1}. Implied acceleration is always 0."""
    return float(row.y_lag)


def m1_ticker_ols(train: pd.DataFrame, row) -> float:
    """Per-ticker OLS: y = a + b*y_lag + c*x."""
    X = np.column_stack([np.ones(len(train)), train["y_lag"], train["x"]])
    beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
    return float(beta @ [1.0, row.y_lag, row.x])


def m2_pooled_fe(train: pd.DataFrame, row) -> float:
    """Pooled OLS with ticker fixed effects: y = a_i + b*y_lag + c*x."""
    tickers = sorted(train["ticker"].unique())
    fe = (train["ticker"].to_numpy()[:, None] == np.array(tickers)[None, :]).astype(float)
    X = np.column_stack([train["y_lag"], train["x"], fe])
    beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
    fe_row = np.array([1.0 if t == row.ticker else 0.0 for t in tickers])
    return float(beta[:2] @ [row.y_lag, row.x] + beta[2:] @ fe_row)


def m3_direction_logit(train: pd.DataFrame, row) -> float:
    """Pooled logistic regression of 1[dy > 0] on dx; scaler fitted on the training window only."""
    y = (train["dy"] > 0).astype(int).to_numpy()
    if y.min() == y.max():
        return float(y[0])
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(train[["dx"]].to_numpy(), y)
    return float(model.predict_proba(np.array([[row.dx]]))[0, 1])
