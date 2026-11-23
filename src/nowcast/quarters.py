"""Point-in-time quarterly revenue from raw XBRL facts (DESIGN.md §7.1).

Rules:
- quarter facts span 80-100 days; annual facts 330-380 days; only 10-Q / 10-K (+ amendments, 10-KT).
- as-reported value of a period = the earliest-filed fact across all revenue concepts; the §7.1 concept
  order breaks ties inside a single filing. Later differing values are logged as restatements, never used.
- Q4 = FY - (Q1+Q2+Q3) when the three quarters tile the fiscal year; `filed` = the 10-K's filed date.
- quarters are keyed by period end; YoY partner is the quarter ending 350-380 days earlier.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nowcast.ingest.sec_facts import REVENUE_CONCEPTS

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (330, 380)
YOY_DAYS = (350, 380)
CURRENT_PERIOD_MAX_LAG = 120  # filed within this many days of period end => period is the filing's own
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-KT/A"}
QUARTER_COLUMNS = ["ticker", "fiscal_q", "start", "end", "revenue", "concept",
                   "derived_q4", "filed", "accn", "revenue_yoy"]
RESTATEMENT_COLUMNS = ["ticker", "start", "end", "concept", "first_val", "first_filed",
                       "later_val", "later_filed", "later_accn"]


def period_facts(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    lo, hi = QUARTER_DAYS if kind == "quarter" else ANNUAL_DAYS
    df = raw[raw["form"].isin(FORMS) & raw["start"].notna()].copy()
    days = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
    return df[(days >= lo) & (days <= hi)].reset_index(drop=True)


def first_filed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest-filed value per (ticker, concept, start, end); later differing values -> restatements."""
    key = ["ticker", "concept", "start", "end"]
    df = df.sort_values(key + ["filed", "accn"])
    firsts = df.drop_duplicates(subset=key, keep="first")
    merged = df.merge(firsts[key + ["val", "filed", "accn"]], on=key, suffixes=("", "_first"))
    later = merged[(merged["accn"] != merged["accn_first"]) & (merged["val"] != merged["val_first"])]
    restated = pd.DataFrame({
        "ticker": later["ticker"], "start": later["start"], "end": later["end"], "concept": later["concept"],
        "first_val": later["val_first"], "first_filed": later["filed_first"],
        "later_val": later["val"], "later_filed": later["filed"], "later_accn": later["accn"],
    }, columns=RESTATEMENT_COLUMNS).drop_duplicates()
    return firsts.reset_index(drop=True), restated.reset_index(drop=True)


def resolve_concept(firsts: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, end): earliest filed across concepts, concept priority as tie-break."""
    rank = {c: i for i, c in enumerate(REVENUE_CONCEPTS)}
    df = firsts.assign(_rank=firsts["concept"].map(rank).fillna(len(rank)))
    df = df.sort_values(["ticker", "end", "filed", "_rank"])
    return df.drop_duplicates(subset=["ticker", "end"], keep="first").drop(columns="_rank").reset_index(drop=True)


def _label(fy, fp) -> str | None:
    if pd.isna(fy) or fp not in {"Q1", "Q2", "Q3", "Q4"}:
        return None
    return f"FY{int(fy)}{fp}"


def _current_period_labels(raw_q: pd.DataFrame) -> dict[tuple[str, dt.date], str]:
    """fiscal label of each quarter from a filing in which it is the current (not comparative) period."""
    lag = (pd.to_datetime(raw_q["filed"]) - pd.to_datetime(raw_q["end"])).dt.days
    cur = raw_q[(lag >= 0) & (lag <= CURRENT_PERIOD_MAX_LAG) & raw_q["fp"].isin(["Q1", "Q2", "Q3"])]
    cur = cur.sort_values("filed")
    out: dict[tuple[str, dt.date], str] = {}
    for r in cur.itertuples():
        out.setdefault((r.ticker, r.end), _label(r.fy, r.fp))
    return out


def derive_q4(quarters: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Append Q4 = FY - (Q1+Q2+Q3) for each annual period whose Q1-Q3 tile it and that lacks a direct Q4."""
    new = []
    one = dt.timedelta(days=1)
    for a in annual.itertuples():
        tq = quarters[(quarters["ticker"] == a.ticker) & (quarters["start"] >= a.start) & (quarters["end"] <= a.end)]
        if (tq["end"] == a.end).any():
            continue  # direct Q4 fact exists
        tq = tq.sort_values("start")
        if len(tq) != 3 or tq.iloc[0]["start"] != a.start:
            continue
        if any(tq.iloc[i + 1]["start"] != tq.iloc[i]["end"] + one for i in range(2)):
            continue
        q4_start = tq.iloc[2]["end"] + one
        if not QUARTER_DAYS[0] <= (a.end - q4_start).days + 1 <= QUARTER_DAYS[1] + 1:
            continue
        new.append({
            "ticker": a.ticker, "start": q4_start, "end": a.end, "val": int(a.val - tq["val"].sum()),
            "concept": a.concept, "derived_q4": True,
            "filed": max(a.filed, tq["filed"].max()), "accn": a.accn,
            "fiscal_q": f"FY{int(a.fy)}Q4" if pd.notna(a.fy) and a.fp == "FY"
            and 0 <= (a.filed - a.end).days <= CURRENT_PERIOD_MAX_LAG else None,
        })
    if not new:
        return quarters
    return pd.concat([quarters, pd.DataFrame(new)], ignore_index=True)


def _add_yoy(q: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in q.groupby("ticker", sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        ends = pd.to_datetime(g["end"])
        yoy = []
        labels = [None if pd.isna(x) else x for x in g["fiscal_q"]]
        for i, e in enumerate(ends):
            gap = (e - ends).dt.days
            cand = gap[(gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1])]
            if cand.empty:
                yoy.append(np.nan)
                continue
            j = (cand - 364).abs().idxmin()
            prev, cur = g.loc[j, "revenue"], g.loc[i, "revenue"]
            yoy.append(np.log(cur / prev) if prev > 0 and cur > 0 else np.nan)
        # fill missing labels from the YoY neighbour (pre-IPO comparatives)
        for _ in range(3):
            for i, e in enumerate(ends):
                if labels[i] is not None:
                    continue
                for j, f in enumerate(ends):
                    gap = (f - e).days
                    if labels[j] and YOY_DAYS[0] <= abs(gap) <= YOY_DAYS[1]:
                        fy, fp = int(labels[j][2:6]), labels[j][6:]
                        labels[i] = f"FY{fy - 1 if gap > 0 else fy + 1}{fp}"
                        break
        out.append(g.assign(revenue_yoy=yoy, fiscal_q=labels))
    return pd.concat(out, ignore_index=True) if out else q.assign(revenue_yoy=pd.Series(dtype=float))


def build_quarters(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_q = period_facts(raw, "quarter")
    q_first, q_restated = first_filed(raw_q)
    a_first, a_restated = first_filed(period_facts(raw, "annual"))
    quarters = resolve_concept(q_first)
    annual = resolve_concept(a_first)

    labels = _current_period_labels(raw_q)
    quarters = quarters.assign(
        derived_q4=False,
        fiscal_q=[labels.get((t, e)) for t, e in zip(quarters["ticker"], quarters["end"])],
    )
    # a direct quarter fact first filed in a 10-K is that year's Q4
    in_10k = quarters["form"].str.startswith("10-K") & quarters["fiscal_q"].isna()
    quarters.loc[in_10k, "fiscal_q"] = [
        f"FY{int(fy)}Q4" if pd.notna(fy) else None for fy in quarters.loc[in_10k, "fy"]
    ]
    quarters = derive_q4(quarters, annual).rename(columns={"val": "revenue"})
    quarters = _add_yoy(quarters)
    quarters["revenue"] = quarters["revenue"].astype("int64")
    quarters["derived_q4"] = quarters["derived_q4"].astype(bool)
    restated = pd.concat([q_restated, a_restated], ignore_index=True)
    return quarters[QUARTER_COLUMNS].sort_values(["ticker", "end"]).reset_index(drop=True), restated
