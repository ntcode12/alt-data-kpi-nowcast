"""DuckDB storage: schema and idempotent loaders."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_facts (
    cik INTEGER, ticker VARCHAR, concept VARCHAR, unit VARCHAR,
    start DATE, "end" DATE, val BIGINT,
    form VARCHAR, fp VARCHAR, fy INTEGER, filed DATE, accn VARCHAR, frame VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_pageviews (
    article VARCHAR, date DATE, views BIGINT,
    PRIMARY KEY (article, date)
);
CREATE TABLE IF NOT EXISTS quarters (
    ticker VARCHAR, fiscal_q VARCHAR, start DATE, "end" DATE, revenue BIGINT,
    concept VARCHAR, derived_q4 BOOLEAN, filed DATE, accn VARCHAR, revenue_yoy DOUBLE,
    PRIMARY KEY (ticker, "end")
);
CREATE TABLE IF NOT EXISTS restatements (
    ticker VARCHAR, start DATE, "end" DATE, concept VARCHAR, first_val BIGINT, first_filed DATE,
    later_val BIGINT, later_filed DATE, later_accn VARCHAR
);
CREATE TABLE IF NOT EXISTS features (
    ticker VARCHAR, fiscal_q VARCHAR, "end" DATE, cut VARCHAR, "asof" DATE,
    views_period DOUBLE, views_yoy DOUBLE, coverage_days INTEGER, period_days INTEGER,
    PRIMARY KEY (ticker, "end", cut)
);
ALTER TABLE features ADD COLUMN IF NOT EXISTS site_yoy DOUBLE;
CREATE TABLE IF NOT EXISTS predictions (
    cut VARCHAR, variant VARCHAR, sample VARCHAR, ticker VARCHAR, fiscal_q VARCHAR, "end" DATE,
    "asof" DATE, model VARCHAR, y_hat DOUBLE, p_accel DOUBLE, y_true DOUBLE, y_lag DOUBLE,
    direction_hat DOUBLE, direction_true DOUBLE
);
CREATE TABLE IF NOT EXISTS raw_releases (
    ticker VARCHAR, accn VARCHAR PRIMARY KEY, filed DATE, period_end DATE, exhibit VARCHAR, text VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_site_pageviews (date DATE PRIMARY KEY, views BIGINT);
"""


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)


def replace_rows(con, table: str, df: pd.DataFrame, key_col: str, keys: list[str]) -> int:
    """Delete all rows for `keys`, then insert `df` -- in one transaction, so re-runs are idempotent."""
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    df = df.reindex(columns=cols)
    con.execute("BEGIN")
    try:
        con.execute(f'DELETE FROM {table} WHERE "{key_col}" IN (SELECT unnest(?))', [list(keys)])
        if not df.empty:
            con.register("_incoming", df)
            col_list = ", ".join(f'"{c}"' for c in cols)
            con.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _incoming")
            con.unregister("_incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)
