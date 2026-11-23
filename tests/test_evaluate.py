"""M3 walk-forward: leakage rules (DESIGN.md §7.4) and metric definitions."""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from nowcast.evaluate import (
    MIN_TRAIN,
    binomial_p,
    build_panel,
    diebold_mariano,
    mean_ic,
    summarize,
    walk_forward,
)

D = dt.date


def synth(n=40, tickers=("A", "B", "C"), seed=0, b=0.5, c=0.8, noise=0.01):
    """Calendar quarters; y = b*y_lag + c*x + noise. Filed 30 days after quarter end."""
    rng = np.random.default_rng(seed)
    qs, fs = [], []
    for t in tickers:
        y = 0.1
        for i in range(n):
            end = (pd.Timestamp("2014-03-31") + pd.offsets.QuarterEnd(i)).date()
            start = (pd.Timestamp(end) - pd.offsets.QuarterBegin(startingMonth=1)).date()
            x = rng.normal(0, 0.1)
            y = b * y + c * x + rng.normal(0, noise)
            qs.append(dict(ticker=t, fiscal_q=f"Q{i}", start=start, end=end, revenue=1,
                           filed=end + dt.timedelta(days=30), revenue_yoy=y))
            for cut, off in [("full", 0), ("m30", 30)]:
                fs.append(dict(ticker=t, fiscal_q=f"Q{i}", end=end, cut=cut,
                               asof=end - dt.timedelta(days=off), views_yoy=x, site_yoy=0.0))
    return pd.DataFrame(qs), pd.DataFrame(fs)


def test_panel_lags_only_consecutive_quarters():
    q, f = synth(n=8, tickers=("A",))
    q = q.drop(index=4)                      # gap in the series
    p = build_panel(q, f, cut="full", variant="raw")
    row = p[p["end"] == q.iloc[4]["end"]].iloc[0]   # first quarter after the gap
    assert math.isnan(row.y_lag) and math.isnan(row.x_lag)
    nxt = p.iloc[5]
    assert nxt.y_lag == pytest.approx(p.iloc[4].y)


def test_site_variant_subtracts_site_yoy():
    q, f = synth(n=6, tickers=("A",))
    f["site_yoy"] = 0.05
    raw = build_panel(q, f, "full", "raw")
    site = build_panel(q, f, "full", "site")
    assert np.allclose(site["x"], raw["x"] - 0.05)


def test_first_prediction_needs_min_training_quarters():
    q, f = synth()
    preds = walk_forward(build_panel(q, f, "full", "raw"))
    for t, g in preds[preds.model == "M1"].groupby("ticker"):
        first = g["end"].min()
        panel = build_panel(q, f, "full", "raw")
        asof = panel[(panel.ticker == t) & (panel.end == first)].iloc[0]["asof"]
        n_train = ((panel.ticker == t) & (panel.filed < asof) & panel.y_lag.notna()).sum()
        assert n_train >= MIN_TRAIN


def test_all_models_scored_on_identical_sample():
    q, f = synth()
    preds = walk_forward(build_panel(q, f, "full", "raw"))
    keys = {m: set(zip(g.ticker, g.end)) for m, g in preds.groupby("model")}
    assert keys["M0"] == keys["M1"] == keys["M2"] == keys["M3"] == keys["MB"]


def test_leakage_rule1_targets_filed_after_asof_never_used():
    q, f = synth()
    base = walk_forward(build_panel(q, f, "full", "raw"))
    target = base[base.model == "M2"].iloc[10]
    asof = target["asof"]
    # tamper with every target filed on/after this prediction's as-of date (incl. the target itself)
    q2 = q.copy()
    late = pd.to_datetime(q2["filed"]) >= pd.Timestamp(asof)
    q2.loc[late, "revenue_yoy"] = 99.0
    tampered = walk_forward(build_panel(q2, f, "full", "raw"))
    for m in ["M0", "M1", "M2", "M3"]:
        a = base[(base.model == m) & (base.asof == asof)].set_index(["ticker", "end"])
        b = tampered[(tampered.model == m) & (tampered.asof == asof)].set_index(["ticker", "end"])
        cols = ["y_hat", "p_accel"]
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False)


def test_prediction_skipped_when_lagged_target_not_yet_filed():
    q, f = synth(n=30, tickers=("A",))
    # make one quarter's filing slow: filed 80 days after end -> after the next quarter's m30 as-of
    i = 20
    q.loc[i, "filed"] = q.loc[i, "end"] + dt.timedelta(days=80)
    preds = walk_forward(build_panel(q, f, "m30", "raw"))
    assert q.loc[i + 1, "end"] not in set(preds["end"])
    assert q.loc[i + 2, "end"] in set(preds["end"])


def test_naive_predicts_last_growth_and_has_no_direction():
    q, f = synth()
    preds = walk_forward(build_panel(q, f, "full", "raw"))
    m0 = preds[preds.model == "M0"]
    assert np.allclose(m0["y_hat"], m0["y_lag"])
    assert m0["direction_hat"].isna().all()


def test_ols_recovers_signal_and_beats_naive():
    q, f = synth(noise=0.005)
    preds = walk_forward(build_panel(q, f, "full", "raw"))
    s = summarize(preds).set_index("model")
    assert s.loc["M1", "skill"] > 0.5 and s.loc["M2", "skill"] > 0.5
    assert s.loc["M3", "hit_rate"] > 0.7
    assert math.isnan(s.loc["M0", "hit_rate"])


def test_binomial_p_one_sided():
    assert binomial_p(5, 10) == pytest.approx(0.623046875)
    assert binomial_p(9, 10) == pytest.approx(0.0107421875)


def test_diebold_mariano_sign_and_identity():
    rng = np.random.default_rng(1)
    e0 = rng.normal(0, 1, 200)
    periods = np.arange(200)
    stat, p = diebold_mariano(e0 * 0.5, e0, periods)
    assert stat < -3 and p < 0.01            # model errors half as large -> significantly better
    stat, p = diebold_mariano(e0, e0, periods)
    assert math.isnan(stat) or stat == 0


def test_mean_ic_perfect_rank():
    df = pd.DataFrame({"period": [1] * 5 + [2] * 5, "dx": list(range(5)) * 2, "dy": list(range(5)) * 2})
    ic, t, n = mean_ic(df, min_names=5)
    assert ic == pytest.approx(1.0) and n == 2


def test_base_rate_model_uses_training_majority_only():
    q, f = synth()
    panel = build_panel(q, f, "full", "raw")
    preds = walk_forward(panel)
    mb = preds[preds.model == "MB"].iloc[5]
    train = panel.dropna(subset=["y", "y_lag", "x", "dx", "dy"])
    train = train[train["filed"] < mb["asof"]]
    assert mb["direction_hat"] == float((train["dy"] > 0).mean() > 0.5)


def test_summary_reports_hit_rate_edge_over_base_rate():
    q, f = synth(noise=0.005)
    s = summarize(walk_forward(build_panel(q, f, "full", "raw"))).set_index("model")
    assert "hit_vs_base" in s.columns
    assert s.loc["M3", "hit_vs_base"] == pytest.approx(s.loc["M3", "hit_rate"] - s.loc["MB", "hit_rate"])
