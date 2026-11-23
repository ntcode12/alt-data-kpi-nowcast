import datetime as dt

import pandas as pd

from nowcast.db import connect, init_schema, replace_rows


def _pv(views):
    return pd.DataFrame(
        {"article": ["A", "A"], "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)], "views": views}
    )


def test_schema_has_design_columns(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    cols = {r[0] for r in con.execute("DESCRIBE raw_facts").fetchall()}
    assert {"cik", "ticker", "concept", "start", "end", "val", "form", "fp", "fy", "filed", "accn"} <= cols
    cols = [r[0] for r in con.execute("DESCRIBE raw_pageviews").fetchall()]
    assert cols == ["article", "date", "views"]


def test_replace_rows_is_idempotent(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    replace_rows(con, "raw_pageviews", _pv([1, 2]), key_col="article", keys=["A"])
    replace_rows(con, "raw_pageviews", _pv([3, 4]), key_col="article", keys=["A"])
    rows = con.execute("SELECT views FROM raw_pageviews ORDER BY date").fetchall()
    assert rows == [(3,), (4,)]


def test_replace_rows_leaves_other_keys(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    replace_rows(con, "raw_pageviews", _pv([1, 2]), key_col="article", keys=["A"])
    b = _pv([5, 6]).assign(article="B")
    replace_rows(con, "raw_pageviews", b, key_col="article", keys=["B"])
    assert con.execute("SELECT count(*) FROM raw_pageviews").fetchone()[0] == 4


def test_init_schema_twice_is_safe(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    init_schema(con)
