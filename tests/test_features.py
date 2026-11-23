"""Feature tests, including the leakage rules of DESIGN.md §7.4 (rule 2: only views dated <= asof)."""

import datetime as dt
import math

import pandas as pd
import pytest

from nowcast.features import CUTS, build_features, winsor_cap

D = dt.date


def quarters(*spans, ticker="T"):
    return pd.DataFrame([
        {"ticker": ticker, "fiscal_q": f"Q{i}", "start": s, "end": e} for i, (s, e) in enumerate(spans)
    ])


def daily(start, end, fn):
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": days.date, "views": [fn(d.date()) for d in days]})


Q_2019 = (D(2019, 1, 1), D(2019, 3, 31))
Q_2020 = (D(2020, 1, 1), D(2020, 3, 31))     # 91 days (leap year)


def feats(q, pv, **kw):
    f = build_features(q, {"T": pv}, **kw)
    return {(r.end, r.cut): r for r in f.itertuples()}


def test_cuts_match_design():
    assert CUTS == {"full": 0, "m30": 30}


def test_full_cut_sums_exact_fiscal_period():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100)
    r = feats(quarters(Q_2019, Q_2020), pv)[(Q_2020[1], "full")]
    assert r.asof == Q_2020[1]
    assert r.views_period == pytest.approx(100 * 91)
    assert r.coverage_days == 91 and r.period_days == 91


def test_m30_cut_scales_quarter_to_date():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100)
    r = feats(quarters(Q_2019, Q_2020), pv)[(Q_2020[1], "m30")]
    assert r.asof == D(2020, 3, 1)
    assert r.coverage_days == 61                       # Jan 1 .. Mar 1
    assert r.views_period == pytest.approx(100 * 91)   # mean daily views x period days


def test_yoy_log_ratio_on_matched_periods():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 200 if d.year == 2020 else 100)
    r = feats(quarters(Q_2019, Q_2020), pv)[(Q_2020[1], "full")]
    # 2019Q1 has 90 days, 2020Q1 91: YoY compares totals over the exact fiscal periods
    assert r.views_yoy == pytest.approx(math.log((200 * 91) / (100 * 90)))
    assert math.isnan(feats(quarters(Q_2019, Q_2020), pv)[(Q_2019[1], "full")].views_yoy)


def test_m30_yoy_uses_same_cut_for_prior_year():
    # prior-year quarter: first 60 days 100/day, last 30 days 1000/day; same pattern this year x2
    def fn(d):
        base = 2 if d.year == 2020 else 1
        late = d >= D(d.year, 3, 2)
        return base * (1000 if late else 100)

    pv = daily(D(2018, 1, 1), D(2020, 12, 31), fn)
    r = feats(quarters(Q_2019, Q_2020), pv, winsor_q=1.0)[(Q_2020[1], "m30")]
    assert r.views_yoy == pytest.approx(math.log(2), abs=0.02)


def test_leakage_views_after_asof_do_not_change_features():
    q = quarters(Q_2019, Q_2020)
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100 + d.toordinal() % 7)
    before = feats(q, pv)
    tampered = pv.copy()
    after_asof = pd.to_datetime(tampered["date"]) > pd.Timestamp(D(2020, 3, 1))
    tampered.loc[after_asof, "views"] = 10_000_000
    after = feats(q, tampered)
    for key in [(Q_2020[1], "m30"), (Q_2019[1], "full"), (Q_2019[1], "m30")]:
        assert after[key].views_period == pytest.approx(before[key].views_period), key
        b, a = before[key].views_yoy, after[key].views_yoy
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), key
    # sanity: the full-quarter cut *does* see the tampered March days
    assert after[(Q_2020[1], "full")].views_period > before[(Q_2020[1], "full")].views_period


def test_winsor_cap_uses_only_past_data():
    pv = daily(D(2019, 1, 1), D(2020, 12, 31), lambda d: 100)
    s = pv.set_index(pd.to_datetime(pv["date"]))["views"]
    cap_before = winsor_cap(s, D(2020, 3, 1), 0.995)
    s2 = s.copy()
    s2[pd.Timestamp("2020-06-01")] = 10**9
    assert winsor_cap(s2, D(2020, 3, 1), 0.995) == cap_before


def test_spike_inside_window_is_clipped_by_past_cap():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100 + (d.toordinal() % 10))
    pv.loc[pd.to_datetime(pv["date"]) == pd.Timestamp("2020-02-01"), "views"] = 1_000_000
    r = feats(quarters(Q_2019, Q_2020), pv)[(Q_2020[1], "full")]
    assert r.views_period < 120 * 91


def test_missing_days_reduce_coverage_not_volume():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100)
    pv = pv[pd.to_datetime(pv["date"]) != pd.Timestamp("2020-02-10")]
    r = feats(quarters(Q_2019, Q_2020), pv)[(Q_2020[1], "full")]
    assert r.coverage_days == 90
    assert r.views_period == pytest.approx(100 * 91)


def test_no_pageviews_gives_nan_feature():
    pv = daily(D(2020, 6, 1), D(2020, 12, 31), lambda d: 100)
    r = feats(quarters(Q_2020), pv)[(Q_2020[1], "full")]
    assert r.coverage_days == 0 and math.isnan(r.views_period)


def test_site_yoy_computed_on_same_window_and_cut():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100)
    site = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 2000 if d.year == 2020 else 1000)
    f = build_features(quarters(Q_2019, Q_2020), {"T": pv}, site=site)
    r = f[(f["end"] == Q_2020[1]) & (f["cut"] == "full")].iloc[0]
    assert r.site_yoy == pytest.approx(math.log((2000 * 91) / (1000 * 90)))
    assert r.views_yoy == pytest.approx(math.log(91 / 90))


def test_site_yoy_ignores_site_views_after_asof():
    pv = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 100)
    site = daily(D(2018, 1, 1), D(2020, 12, 31), lambda d: 1000)
    q = quarters(Q_2019, Q_2020)
    a = build_features(q, {"T": pv}, site=site)
    site.loc[pd.to_datetime(site["date"]) > pd.Timestamp(D(2020, 3, 1)), "views"] = 10**12
    b = build_features(q, {"T": pv}, site=site)
    m30 = lambda f: f[(f["end"] == Q_2020[1]) & (f["cut"] == "m30")].iloc[0].site_yoy
    assert m30(a) == pytest.approx(m30(b))
