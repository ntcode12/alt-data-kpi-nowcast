import datetime as dt

import pandas as pd

from nowcast.universe import Ticker, combine_titles, quarter_coverage

D = dt.date


def test_ticker_titles_include_aliases_once():
    t = Ticker("LULU", 1, "Lululemon", "r", article_aliases=["Lululemon Athletica", "Lululemon"])
    assert t.titles == ["Lululemon", "Lululemon Athletica"]


def test_combine_titles_sums_views_per_day():
    pv = pd.DataFrame({
        "article": ["Old", "Old", "New", "New"],
        "date": [D(2025, 3, 8), D(2025, 3, 9), D(2025, 3, 9), D(2025, 3, 10)],
        "views": [100, 10, 90, 105],
    })
    out = combine_titles(pv)
    assert out["date"].tolist() == [D(2025, 3, 8), D(2025, 3, 9), D(2025, 3, 10)]
    assert out["views"].tolist() == [100, 100, 105]


def test_quarter_coverage_counts_days_with_data():
    q = pd.DataFrame({"start": [D(2020, 1, 1)], "end": [D(2020, 1, 10)]})
    pv = pd.DataFrame({"date": [D(2020, 1, d) for d in range(1, 10)], "views": 1})
    assert quarter_coverage(q, pv).iloc[0] == 0.9
