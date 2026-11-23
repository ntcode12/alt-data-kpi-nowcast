"""Pageview features aligned to exact fiscal periods, point in time (DESIGN.md §7.2, §7.4 rule 2).

For each fiscal quarter q and cut c (asof = end - offset):
- only daily views dated <= asof are read -- including for the winsorization cap;
- daily views are clipped at the 99.5th percentile of the article's history up to asof;
- views_period = mean clipped daily views over the days observed in [start, min(end, asof)] x period_days
  (= the full-quarter sum when coverage is complete; a scaled quarter-to-date estimate for early cuts);
- views_yoy = ln(views_period_q / views_period_{q-4}), where q-4 uses the same cut offset and the same cap.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nowcast.quarters import YOY_DAYS

CUTS = {"full": 0, "m30": 30}
WINSOR_Q = 0.995
FEATURE_COLUMNS = ["ticker", "fiscal_q", "end", "cut", "asof", "views_period", "views_yoy",
                   "coverage_days", "period_days", "site_yoy"]


def _series(pv: pd.DataFrame) -> pd.Series:
    return pv.set_index(pd.to_datetime(pv["date"]))["views"].sort_index().astype(float)


def winsor_cap(s: pd.Series, asof: dt.date, q: float = WINSOR_Q) -> float:
    past = s[s.index <= pd.Timestamp(asof)]
    return float(past.quantile(q)) if len(past) else np.inf


def _window(s: pd.Series, start: dt.date, end: dt.date, offset: int, cap: float) -> tuple[float, int]:
    asof = end - dt.timedelta(days=offset)
    period_days = (end - start).days + 1
    w = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(min(end, asof)))]
    if w.empty:
        return np.nan, 0
    return float(w.clip(upper=cap).mean() * period_days), len(w)


def _yoy(a: float, b: float) -> float:
    return float(np.log(a / b)) if pd.notna(a) and pd.notna(b) and a > 0 and b > 0 else np.nan


def build_features(quarters: pd.DataFrame, pageviews: dict[str, pd.DataFrame],
                   cuts: dict[str, int] = CUTS, winsor_q: float = WINSOR_Q,
                   site: pd.DataFrame | None = None) -> pd.DataFrame:
    """`site`: total en.wikipedia daily views; adds `site_yoy` on the same window/cut (no clipping)."""
    site_s = _series(site) if site is not None and not site.empty else None
    rows = []
    for ticker, q in quarters.groupby("ticker", sort=False):
        pv = pageviews.get(ticker)
        if pv is None or pv.empty:
            continue
        s = _series(pv)
        q = q.sort_values("end").reset_index(drop=True)
        ends = pd.to_datetime(q["end"])
        for i, r in q.iterrows():
            gap = (ends[i] - ends).dt.days
            cand = gap[(gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1])]
            prev = q.loc[(cand - 364).abs().idxmin()] if not cand.empty else None
            for cut, offset in cuts.items():
                asof = r["end"] - dt.timedelta(days=offset)
                cap = winsor_cap(s, asof, winsor_q) if winsor_q < 1 else np.inf
                views, cov = _window(s, r["start"], r["end"], offset, cap)
                yoy = site_yoy = np.nan
                if prev is not None:
                    pviews, _ = _window(s, prev["start"], prev["end"], offset, cap)
                    yoy = _yoy(views, pviews)
                    if site_s is not None:
                        site_yoy = _yoy(_window(site_s, r["start"], r["end"], offset, np.inf)[0],
                                        _window(site_s, prev["start"], prev["end"], offset, np.inf)[0])
                rows.append({
                    "ticker": ticker, "fiscal_q": r["fiscal_q"], "end": r["end"], "cut": cut,
                    "asof": asof, "views_period": views, "views_yoy": yoy,
                    "coverage_days": cov, "period_days": (r["end"] - r["start"]).days + 1,
                    "site_yoy": site_yoy,
                })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
