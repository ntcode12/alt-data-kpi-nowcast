"""Walk-forward evaluation with an expanding window (DESIGN.md §7.4, pre-registration §16).

Leakage rules enforced here:
  1. a target enters training only if its `filed` date is strictly before the prediction's as-of date;
     the lagged target used as a predictor must also be filed before the as-of date.
  2. features come from `features.py`, which reads only pageviews dated <= as-of (tested there).
  3. the winsorization cap is computed from data <= each row's as-of (features.py), which is inside the
     training window; the M3 scaler is fitted on the training window only.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

from nowcast.features import CUTS
from nowcast.models import m0_naive, m1_ticker_ols, m2_pooled_fe, m3_direction_logit

MIN_TRAIN = 12
CONSECUTIVE_DAYS = (75, 105)
COVID = (dt.date(2020, 4, 1), dt.date(2021, 6, 30))
MODELS = ("M0", "M1", "M2", "M3", "MB")  # MB: base-rate direction benchmark (added post hoc, §16 note)
PRED_COLUMNS = ["ticker", "fiscal_q", "end", "asof", "model", "y_hat", "p_accel", "y_true", "y_lag",
                "direction_hat", "direction_true"]


def build_panel(quarters: pd.DataFrame, features: pd.DataFrame, cut: str, variant: str) -> pd.DataFrame:
    f = features[features["cut"] == cut][["ticker", "end", "views_yoy", "site_yoy"]]
    p = quarters[["ticker", "fiscal_q", "end", "filed", "revenue_yoy"]].merge(f, on=["ticker", "end"], how="left")
    p = p.rename(columns={"revenue_yoy": "y"})
    p["x"] = p["views_yoy"] - (p["site_yoy"] if variant == "site" else 0.0)
    p["asof"] = [e - dt.timedelta(days=CUTS[cut]) for e in p["end"]]
    p = p.sort_values(["ticker", "end"]).reset_index(drop=True)
    prev = p.groupby("ticker")[["end", "y", "x", "filed"]].shift(1)
    gap = (pd.to_datetime(p["end"]) - pd.to_datetime(prev["end"])).dt.days
    ok = gap.between(*CONSECUTIVE_DAYS)
    p["y_lag"] = prev["y"].where(ok)
    p["x_lag"] = prev["x"].where(ok)
    p["y_lag_filed"] = prev["filed"].where(ok)
    p["dy"] = p["y"] - p["y_lag"]
    p["dx"] = p["x"] - p["x_lag"]
    return p.drop(columns=["views_yoy", "site_yoy"])


def walk_forward(panel: pd.DataFrame, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    complete = panel.dropna(subset=["y", "y_lag", "x"])
    targets = complete.dropna(subset=["x_lag"])
    targets = targets[targets["y_lag_filed"] < targets["asof"]].sort_values(["asof", "ticker"])
    out = []
    for row in targets.itertuples():
        train = complete[complete["filed"] < row.asof]                      # leakage rule 1
        train_t = train[train["ticker"] == row.ticker]
        if len(train_t) < min_train:
            continue
        train_dir = train.dropna(subset=["dx", "dy"])
        preds = {
            "M0": (m0_naive(row), np.nan),
            "M1": (m1_ticker_ols(train_t, row), np.nan),
            "M2": (m2_pooled_fe(train, row), np.nan),
            "M3": (np.nan, m3_direction_logit(train_dir, row)),
            "MB": (np.nan, float((train_dir["dy"] > 0).mean())),
        }
        for model, (y_hat, p_accel) in preds.items():
            if model == "M0":
                d_hat = np.nan
            elif model in ("M3", "MB"):
                d_hat = float(p_accel > 0.5)
            else:
                d_hat = float(y_hat > row.y_lag)
            out.append(dict(ticker=row.ticker, fiscal_q=row.fiscal_q, end=row.end, asof=row.asof,
                            model=model, y_hat=y_hat, p_accel=p_accel, y_true=row.y, y_lag=row.y_lag,
                            direction_hat=d_hat, direction_true=float(row.y > row.y_lag)))
    return pd.DataFrame(out, columns=PRED_COLUMNS)


def binomial_p(k: int, n: int) -> float:
    """One-sided P(X >= k), X ~ Binomial(n, 0.5)."""
    return float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else np.nan


def _nw_var(d: np.ndarray) -> float:
    T = len(d)
    L = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    v = dc @ dc / T
    for lag in range(1, L + 1):
        v += 2 * (1 - lag / (L + 1)) * (dc[lag:] @ dc[:-lag]) / T
    return float(v)


def diebold_mariano(e_model, e_naive, periods) -> tuple[float, float]:
    """DM on absolute-error differentials averaged per period; negative stat = model better."""
    d = pd.Series(np.abs(e_model) - np.abs(e_naive)).groupby(np.asarray(periods)).mean().sort_index().to_numpy()
    if len(d) < 3:
        return np.nan, np.nan
    v = _nw_var(d)
    if v <= 1e-18:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / len(d))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def mean_ic(df: pd.DataFrame, min_names: int = 5) -> tuple[float, float, int]:
    """Average per-period Spearman correlation of dx vs dy, its t-stat, and the number of periods."""
    ics = []
    for _, g in df.groupby("period"):
        g = g.dropna(subset=["dx", "dy"])
        if len(g) >= min_names and g["dx"].nunique() > 1 and g["dy"].nunique() > 1:
            ics.append(g["dx"].rank().corr(g["dy"].rank()))
    if not ics:
        return np.nan, np.nan, 0
    ics = np.array(ics)
    sd = ics.std(ddof=1) if len(ics) > 1 else 0.0
    t = ics.mean() / (sd / np.sqrt(len(ics))) if sd > 0 else np.nan
    return float(ics.mean()), float(t), len(ics)


def _period(ends) -> np.ndarray:
    return pd.PeriodIndex(pd.to_datetime(pd.Series(ends)), freq="Q").astype(str).to_numpy()


def summarize(preds: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    groups = preds.groupby(list(by)) if by else [((), preds)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        naive = g[g["model"] == "M0"].set_index(["ticker", "end"])
        mae0 = (naive["y_hat"] - naive["y_true"]).abs().mean()
        for m in MODELS:
            gm = g[g["model"] == m].set_index(["ticker", "end"])
            if gm.empty:
                continue
            r = dict(zip(by, key), model=m, n=len(gm))
            if m not in ("M3", "MB"):
                err = gm["y_hat"] - gm["y_true"]
                r["mae"] = err.abs().mean()
                r["skill"] = 1 - r["mae"] / mae0 if mae0 > 0 else np.nan
                if m != "M0":
                    e0 = (naive["y_hat"] - naive["y_true"]).reindex(gm.index)
                    r["dm_stat"], r["dm_p"] = diebold_mariano(err.to_numpy(), e0.to_numpy(),
                                                              _period(gm.index.get_level_values("end")))
            if m == "M0":
                r["hit_rate"], r["hit_p"] = np.nan, np.nan
            else:
                hits = int((gm["direction_hat"] == gm["direction_true"]).sum())
                r["hit_rate"], r["hit_p"] = hits / len(gm), binomial_p(hits, len(gm))
            rows.append(r)
    cols = list(by) + ["model", "n", "mae", "skill", "hit_rate", "hit_p", "hit_vs_base", "dm_stat", "dm_p"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    base = out[out["model"] == "MB"].set_index(list(by))["hit_rate"] if by else \
        out.loc[out["model"] == "MB", "hit_rate"].squeeze()
    if by:
        out["hit_vs_base"] = out["hit_rate"] - out.set_index(list(by)).index.map(base).to_numpy()
    else:
        out["hit_vs_base"] = out["hit_rate"] - base
    out.loc[out["model"].isin(["M0", "MB"]), "hit_vs_base"] = np.nan
    return out


def exclude_covid(panel: pd.DataFrame) -> pd.DataFrame:
    """Robustness: drop COVID quarters from training and evaluation (lags still from the full series)."""
    e = panel["end"]
    return panel[~((e >= COVID[0]) & (e <= COVID[1]))]


def ic_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(period=_period(panel["end"]))[["ticker", "period", "dx", "dy"]]
