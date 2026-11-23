import datetime as dt
import math

import pandas as pd
import pytest

from nowcast.ingest.sec_facts import RAW_FACTS_COLUMNS
from nowcast.quarters import build_quarters, first_filed, period_facts

D = dt.date
REV = "Revenues"
ASC = "RevenueFromContractWithCustomerExcludingAssessedTax"


def fact(start, end, val, filed, accn, form="10-Q", fy=None, fp=None, concept=REV, ticker="T"):
    return dict(cik=1, ticker=ticker, concept=concept, unit="USD", start=start, end=end, val=val,
                form=form, fp=fp, fy=fy, filed=filed, accn=accn, frame=None)


def raw(*rows):
    return pd.DataFrame(list(rows), columns=RAW_FACTS_COLUMNS)


def calendar_year(year, vals, concept=REV, fy_total=None, q4_direct=False):
    """Q1-Q3 10-Q facts + FY 10-K fact for a calendar fiscal year."""
    q = [(D(year, 1, 1), D(year, 3, 31)), (D(year, 4, 1), D(year, 6, 30)), (D(year, 7, 1), D(year, 9, 30))]
    rows = [
        fact(s, e, v, e + dt.timedelta(days=30), f"{year}-q{i+1}", fy=year, fp=f"Q{i+1}", concept=concept)
        for i, ((s, e), v) in enumerate(zip(q, vals[:3]))
    ]
    total = fy_total if fy_total is not None else sum(vals)
    rows.append(fact(D(year, 1, 1), D(year, 12, 31), total, D(year + 1, 2, 1), f"{year}-k",
                     form="10-K", fy=year, fp="FY", concept=concept))
    if q4_direct:
        rows.append(fact(D(year, 10, 1), D(year, 12, 31), vals[3], D(year + 1, 2, 1), f"{year}-k",
                         form="10-K", fy=year, fp="FY", concept=concept))
    return rows


def test_period_facts_duration_filter_and_forms():
    df = raw(
        fact(D(2020, 1, 1), D(2020, 3, 31), 1, D(2020, 4, 30), "a"),          # 90d quarter
        fact(D(2020, 1, 1), D(2020, 6, 30), 2, D(2020, 7, 30), "b"),          # 6-month YTD
        fact(D(2020, 1, 1), D(2020, 12, 31), 3, D(2021, 2, 1), "c", form="10-K"),
        fact(D(2020, 1, 1), D(2020, 3, 31), 4, D(2020, 4, 20), "d", form="8-K"),
        fact(D(2020, 4, 1), D(2020, 6, 30), 5, D(2020, 8, 1), "e", form="10-Q/A"),
    )
    q = period_facts(df, "quarter")
    assert sorted(q["val"]) == [1, 5]
    a = period_facts(df, "annual")
    assert a["val"].tolist() == [3]


def test_first_filed_wins_and_restatement_logged():
    df = raw(
        fact(D(2020, 1, 1), D(2020, 3, 31), 100, D(2020, 4, 30), "orig"),
        fact(D(2020, 1, 1), D(2020, 3, 31), 105, D(2021, 4, 30), "later"),
        fact(D(2020, 1, 1), D(2020, 3, 31), 100, D(2022, 4, 30), "same"),
    )
    firsts, restated = first_filed(df)
    assert firsts["val"].tolist() == [100] and firsts["accn"].tolist() == ["orig"]
    assert len(restated) == 1
    r = restated.iloc[0]
    assert (r.first_val, r.later_val, r.later_accn) == (100, 105, "later")


def test_q4_derived_from_annual_minus_q1_q3():
    q, _ = build_quarters(raw(*calendar_year(2020, [10, 20, 30, 45])))
    q4 = q[q["end"] == D(2020, 12, 31)].iloc[0]
    assert q4.revenue == 45
    assert q4.derived_q4
    assert q4.start == D(2020, 10, 1)
    assert q4.filed == D(2021, 2, 1)       # 10-K filed date, not the 10-Qs'
    assert q4.accn == "2020-k"
    assert q4.fiscal_q == "FY2020Q4"
    assert len(q) == 4


def test_direct_q4_fact_is_used_not_derived():
    q, _ = build_quarters(raw(*calendar_year(2020, [10, 20, 30, 45], fy_total=999, q4_direct=True)))
    q4 = q[q["end"] == D(2020, 12, 31)].iloc[0]
    assert q4.revenue == 45 and not q4.derived_q4


def test_q4_not_derived_when_q1_q3_incomplete():
    rows = [r for r in calendar_year(2020, [10, 20, 30, 45]) if r["fp"] != "Q2"]
    q, _ = build_quarters(raw(*rows))
    assert D(2020, 12, 31) not in set(q["end"])


def test_concept_switch_uses_earliest_filed_across_concepts():
    rows = calendar_year(2017, [10, 20, 30, 40], concept=REV) + calendar_year(2018, [11, 22, 33, 44], concept=ASC)
    # the 2018 10-Q re-reports Q1 2017 under the new concept as a comparative, filed a year later
    rows.append(fact(D(2017, 1, 1), D(2017, 3, 31), 10, D(2018, 4, 30), "2018-q1", fy=2018, fp="Q1", concept=ASC))
    q, _ = build_quarters(raw(*rows))
    q1_17 = q[q["end"] == D(2017, 3, 31)].iloc[0]
    assert q1_17.concept == REV and q1_17.filed == D(2017, 4, 30)   # as-reported date preserved
    assert q[q["end"] == D(2018, 3, 31)].iloc[0].concept == ASC
    assert len(q) == 8


def test_concept_priority_breaks_ties_within_one_filing():
    rows = [
        fact(D(2020, 1, 1), D(2020, 3, 31), 100, D(2020, 4, 30), "a", concept=REV, fy=2020, fp="Q1"),
        fact(D(2020, 1, 1), D(2020, 3, 31), 98, D(2020, 4, 30), "a", concept=ASC, fy=2020, fp="Q1"),
    ]
    q, _ = build_quarters(raw(*rows))
    assert q.iloc[0].concept == ASC and q.iloc[0].revenue == 98


def test_yoy_log_growth():
    q, _ = build_quarters(raw(*calendar_year(2019, [10, 20, 30, 40]), *calendar_year(2020, [12, 20, 30, 40])))
    q1 = q[q["end"] == D(2020, 3, 31)].iloc[0]
    assert q1.revenue_yoy == pytest.approx(math.log(12 / 10))
    assert math.isnan(q[q["end"] == D(2019, 3, 31)].iloc[0].revenue_yoy)


def test_yoy_matches_across_53_week_year():
    # retailer-style 13-week quarters; 2nd year has a 14-week Q4 (53-week year)
    def wk(d, w):
        return d + dt.timedelta(weeks=w)

    s = D(2019, 2, 3)
    ends = [wk(s, 13 * i) - dt.timedelta(days=1) for i in range(1, 9)]
    ends[7] = ends[7] + dt.timedelta(weeks=1)            # 53-week year
    starts = [s] + [e + dt.timedelta(days=1) for e in ends[:-1]]
    rows = [fact(st, en, 100 + i, en + dt.timedelta(days=30), f"a{i}", fy=2019 + i // 4, fp=f"Q{i % 4 + 1}")
            for i, (st, en) in enumerate(zip(starts, ends))]
    rows.append(fact(D(2021, 2, 7), D(2021, 5, 8), 200, D(2021, 6, 8), "a8", fy=2021, fp="Q1"))
    q, _ = build_quarters(raw(*rows))
    q = q.sort_values("end").reset_index(drop=True)
    assert q.loc[4, "revenue_yoy"] == pytest.approx(math.log(104 / 100))
    assert q.loc[8, "revenue_yoy"] == pytest.approx(math.log(200 / 104))   # 371 days after a 53-wk year


def test_fiscal_label_from_current_period_filing_not_comparative():
    rows = calendar_year(2020, [10, 20, 30, 40])
    # pre-IPO style: 2019 Q1 only ever appears as a comparative in the FY2020 Q1 10-Q
    rows.append(fact(D(2019, 1, 1), D(2019, 3, 31), 8, D(2020, 4, 30), "2020-q1", fy=2020, fp="Q1"))
    q, _ = build_quarters(raw(*rows))
    assert q[q["end"] == D(2019, 3, 31)].iloc[0].fiscal_q == "FY2019Q1"
    assert q[q["end"] == D(2020, 3, 31)].iloc[0].fiscal_q == "FY2020Q1"


def test_multiple_tickers_kept_separate():
    a = calendar_year(2020, [1, 2, 3, 4])
    b = [dict(r, ticker="U", val=r["val"] * 10) for r in calendar_year(2020, [1, 2, 3, 4])]
    q, _ = build_quarters(raw(*a, *b))
    assert q.groupby("ticker")["revenue"].sum().to_dict() == {"T": 10, "U": 100}
