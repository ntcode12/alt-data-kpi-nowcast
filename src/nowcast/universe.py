"""Universe config and the M1 data check (DESIGN.md §4 criteria)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nowcast.quarters import YOY_DAYS

SHORT = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "ASC606",
    "Revenues": "Revenues",
    "SalesRevenueNet": "SalesRevenueNet",
}


@dataclass
class Ticker:
    ticker: str
    cik: int
    article: str
    rationale: str
    status: str = "candidate"
    note: str | None = None
    article_aliases: list[str] = field(default_factory=list)

    @property
    def titles(self) -> list[str]:
        """Primary article plus former titles (page moves); views are summed across them."""
        return list(dict.fromkeys([self.article, *self.article_aliases]))


@dataclass
class Universe:
    pageviews_start: dt.date
    pageviews_end: dt.date
    min_overlap_quarters: int
    min_coverage: float
    min_median_daily_views: int
    tickers: list[Ticker] = field(default_factory=list)

    def active(self) -> list[Ticker]:
        return [t for t in self.tickers if t.status != "excluded"]


def load_universe(path: Path) -> Universe:
    doc = yaml.safe_load(Path(path).read_text())
    w, c = doc["data_window"], doc["criteria"]
    return Universe(
        pageviews_start=w["pageviews_start"], pageviews_end=w["pageviews_end"],
        min_overlap_quarters=c["min_overlap_quarters"], min_coverage=c["min_coverage"],
        min_median_daily_views=c["min_median_daily_views"],
        tickers=[Ticker(**t) for t in doc["tickers"]],
    )


def combine_titles(pv: pd.DataFrame) -> pd.DataFrame:
    """Sum daily views across a ticker's titles (current article + pre-move titles)."""
    if pv.empty:
        return pd.DataFrame(columns=["date", "views"])
    return pv.groupby("date", as_index=False)["views"].sum().sort_values("date").reset_index(drop=True)


def ticker_pageviews(con, t: Ticker) -> pd.DataFrame:
    pv = con.execute("SELECT * FROM raw_pageviews WHERE article IN (SELECT unnest(?))", [t.titles]).df()
    pv["date"] = pd.to_datetime(pv["date"]).dt.date
    return combine_titles(pv)


def concept_summary(q: pd.DataFrame) -> str:
    """e.g. 'Revenues (2015-2017) → ASC606 (2018-2026)' -- concept used per period, in time order."""
    parts: list[list] = []
    for r in q.sort_values("end").itertuples():
        name = SHORT.get(r.concept, r.concept)
        if parts and parts[-1][0] == name:
            parts[-1][2] = r.end.year
        else:
            parts.append([name, r.end.year, r.end.year])
    return " → ".join(f"{n} ({a}–{b})" if a != b else f"{n} ({a})" for n, a, b in parts)


def quarter_coverage(q: pd.DataFrame, pv: pd.DataFrame) -> pd.Series:
    """Share of days in each [start, end] with a pageview observation."""
    days = set(pd.to_datetime(pv["date"]).dt.date)
    cov = []
    for r in q.itertuples():
        n = (r.end - r.start).days + 1
        have = sum(1 for k in range(n) if r.start + dt.timedelta(days=k) in days)
        cov.append(have / n)
    return pd.Series(cov, index=q.index, dtype=float)


def max_step_ratio(pv: pd.DataFrame, window: int = 91) -> float:
    """Largest ratio between medians of consecutive 91-day blocks (flags page moves / title changes)."""
    s = pv.set_index(pd.to_datetime(pv["date"]))["views"].sort_index()
    med = s.resample(f"{window}D").median().dropna()
    med = med[med > 0]
    if len(med) < 2:
        return float("nan")
    r = (med / med.shift(1)).dropna()
    return float(max(r.max(), (1 / r).max()))


def check_ticker(t: Ticker, q: pd.DataFrame, pv: pd.DataFrame, u: Universe) -> dict:
    q = q.sort_values("end").reset_index(drop=True)
    notes = []
    if q.empty:
        return dict(ticker=t.ticker, concept="—", revenue_quarters=0, overlap_quarters=0,
                    yoy_usable=0, median_views=np.nan, passed=False, note="no revenue facts")
    in_window = (q["start"] >= u.pageviews_start) & (q["end"] <= u.pageviews_end)
    cov = quarter_coverage(q, pv) if not pv.empty else pd.Series(0.0, index=q.index)
    overlap = in_window & (cov >= u.min_coverage)
    ends = pd.to_datetime(q["end"])
    yoy_ok = []
    for i, e in enumerate(ends):
        gap = (e - ends).dt.days
        partner = gap[(gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1])].index
        yoy_ok.append(bool(overlap[i] and len(partner) and overlap[partner].any()
                           and pd.notna(q.loc[i, "revenue_yoy"])))
    med = float(pv["views"].median()) if not pv.empty else 0.0
    n_overlap = int(overlap.sum())
    passed = n_overlap >= u.min_overlap_quarters and med >= u.min_median_daily_views

    # gaps in the quarterly revenue series inside the overlap window
    ov_ends = ends[overlap].reset_index(drop=True)
    gaps = int(((ov_ends.diff().dt.days) > 100).sum())
    if gaps:
        notes.append(f"{gaps} gap(s) in revenue series")
    if med < u.min_median_daily_views:
        notes.append(f"median views {med:.0f} < {u.min_median_daily_views}")
    step = max_step_ratio(pv) if not pv.empty else np.nan
    if pd.notna(step) and step > 5:
        notes.append(f"pageview step change ×{step:.0f} (check page moves)")
    if n_overlap < u.min_overlap_quarters:
        notes.append(f"short history: first quarter {q['end'].min()}")
    n_q4 = int(q.loc[overlap, "derived_q4"].sum())
    notes.append(f"{n_q4} derived Q4s")
    return dict(ticker=t.ticker, concept=concept_summary(q[overlap] if overlap.any() else q),
                revenue_quarters=len(q), overlap_quarters=n_overlap, yoy_usable=int(sum(yoy_ok)),
                median_views=med, passed=passed, note="; ".join(notes))


def to_markdown(df: pd.DataFrame, u: Universe) -> str:
    lines = [
        "# Universe data check (Milestone 1)",
        "",
        f"Criteria: ≥{u.min_overlap_quarters} fiscal quarters with first-filed revenue (incl. derived Q4) "
        f"and ≥{u.min_coverage:.0%} daily pageview coverage of the exact fiscal period; "
        f"median daily views ≥{u.min_median_daily_views}. Pageview window "
        f"{u.pageviews_start} → {u.pageviews_end}.",
        "",
        "| ticker | article | concept used | revenue qtrs | overlapping qtrs | YoY-usable | median views/day | result | note |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.ticker} | {r.article} | {r.concept} | {r.revenue_quarters} | {r.overlap_quarters} | "
            f"{r.yoy_usable} | {r.median_views:,.0f} | {'PASS' if r.passed else 'FAIL'} | {r.note} |"
        )
    return "\n".join(lines) + "\n"
