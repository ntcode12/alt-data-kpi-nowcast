"""CLI: nowcast ingest | check-universe | build."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from nowcast.db import connect, init_schema, replace_rows
from nowcast.http import sec_client, wiki_client
from nowcast.ingest.sec_facts import fetch_company_tickers, fetch_companyfacts, parse_companyfacts
from nowcast.ingest.wiki_pageviews import (
    fetch_pageviews,
    fetch_site_pageviews,
    parse_pageviews,
    parse_site_pageviews,
)
from nowcast.ingest.wiki_titles import resolve_title
from nowcast.settings import PROJECT_ROOT, load_settings
from nowcast.universe import check_ticker, load_universe, ticker_pageviews, to_markdown

log = logging.getLogger("nowcast")


def cmd_ingest(args, settings) -> int:
    u = load_universe(settings.universe_path)
    tickers = [t for t in u.tickers if not args.tickers or t.ticker in args.tickers]
    sec, wiki = sec_client(settings), wiki_client(settings)
    con = connect(settings.db_path)
    init_schema(con)
    sec_map = fetch_company_tickers(sec)
    problems = []
    for t in tickers:
        if sec_map.get(t.ticker) != t.cik:
            problems.append(f"{t.ticker}: CIK {t.cik} in yaml, SEC says {sec_map.get(t.ticker)}")
        canonical, exists = resolve_title(t.article, wiki)
        if not exists or canonical != t.article:
            problems.append(f"{t.ticker}: article '{t.article}' -> canonical '{canonical}' (exists={exists})")
        facts = parse_companyfacts(fetch_companyfacts(t.cik, sec), t.ticker)
        replace_rows(con, "raw_facts", facts, key_col="ticker", keys=[t.ticker])
        n_pv = 0
        for title in t.titles:
            pv = parse_pageviews(fetch_pageviews(title, u.pageviews_start, u.pageviews_end, wiki), title)
            replace_rows(con, "raw_pageviews", pv, key_col="article", keys=[title])
            n_pv += len(pv)
        log.info("%-5s raw_facts=%5d  raw_pageviews=%5d  (%s)", t.ticker, len(facts), n_pv, " + ".join(t.titles))
    site = parse_site_pageviews(fetch_site_pageviews(u.pageviews_start, u.pageviews_end, wiki))
    con.execute("DELETE FROM raw_site_pageviews")
    con.execute("INSERT INTO raw_site_pageviews SELECT date, views FROM site")
    log.info("site     raw_site_pageviews=%d (en.wikipedia total, agent=user)", len(site))
    log.info("network requests made: SEC=%d Wikimedia=%d", sec.requests_made, wiki.requests_made)
    for p in problems:
        log.warning(p)
    return 1 if problems else 0


def cmd_check_universe(args, settings) -> int:
    from nowcast.quarters import build_quarters

    u = load_universe(settings.universe_path)
    con = connect(settings.db_path)
    quarters, _ = build_quarters(_load_raw_facts(con))
    rows = []
    for t in u.tickers:
        pv = ticker_pageviews(con, t)
        res = check_ticker(t, quarters[quarters["ticker"] == t.ticker], pv, u)
        rows.append({**res, "article": " + ".join(t.titles)})
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "universe_check.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown(df, u))
    print(to_markdown(df, u))
    print(f"passing: {int(df['passed'].sum())}/{len(df)}  -> {out.relative_to(PROJECT_ROOT)}")
    return 0


def _load_raw_facts(con) -> pd.DataFrame:
    raw = con.execute("SELECT * FROM raw_facts").df()
    for c in ["start", "end", "filed"]:
        raw[c] = pd.to_datetime(raw[c]).dt.date
    return raw


def cmd_build(args, settings) -> int:
    from nowcast.features import build_features
    from nowcast.quarters import build_quarters

    u = load_universe(settings.universe_path)
    active = u.active()
    con = connect(settings.db_path)
    init_schema(con)
    raw = _load_raw_facts(con)
    quarters, restated = build_quarters(raw[raw["ticker"].isin([t.ticker for t in active])])
    keys = [t.ticker for t in active]
    replace_rows(con, "quarters", quarters, key_col="ticker", keys=keys)
    replace_rows(con, "restatements", restated, key_col="ticker", keys=keys)
    site = con.execute("SELECT date, views FROM raw_site_pageviews").df()
    site["date"] = pd.to_datetime(site["date"]).dt.date
    feats = build_features(quarters, {t.ticker: ticker_pageviews(con, t) for t in active}, site=site)
    replace_rows(con, "features", feats, key_col="ticker", keys=keys)
    log.info("quarters=%d restatements=%d features=%d", len(quarters), len(restated), len(feats))
    return 0


def cmd_spot_check(args, settings) -> int:
    con = connect(settings.db_path)
    for t in args.tickers:
        df = con.execute(
            """SELECT q.fiscal_q, q.start, q."end", q.revenue / 1e6 AS revenue_musd, q.derived_q4, q.filed,
                      q.accn, round(q.revenue_yoy, 4) AS rev_yoy, round(f.views_yoy, 4) AS views_yoy_full,
                      f.coverage_days
               FROM quarters q LEFT JOIN features f
                 ON f.ticker = q.ticker AND f."end" = q."end" AND f.cut = 'full'
               WHERE q.ticker = ? ORDER BY q."end" DESC LIMIT ?""", [t, args.n]).df()
        cik = con.execute("SELECT any_value(cik) FROM raw_facts WHERE ticker = ?", [t]).fetchone()[0]
        df["filing"] = [f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/" for a in df["accn"]]
        print(f"\n== {t} ==")
        print(df.drop(columns="accn").to_string(index=False))
    return 0


def cmd_evaluate(args, settings) -> int:
    from nowcast.run_eval import insample, leave_one_out, report, run_all

    con = connect(settings.db_path)
    init_schema(con)
    q = con.execute("SELECT * FROM quarters").df()
    f = con.execute("SELECT * FROM features").df()
    for df, cols in [(q, ["start", "end", "filed"]), (f, ["end", "asof"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.date
    preds, ics = run_all(q, f)
    con.execute("DELETE FROM predictions")
    replace_rows(con, "predictions", preds, key_col="cut", keys=["full", "m30"])
    loo = leave_one_out(q, f, sorted(q["ticker"].unique()))
    md = report(preds, ics, loo, insample(q, f))
    out = PROJECT_ROOT / "reports" / "results_m3.md"
    out.write_text(md)
    print(md)
    log.info("predictions=%d -> %s", len(preds), out.relative_to(PROJECT_ROOT))
    return 0


def cmd_ingest_releases(args, settings) -> int:
    import datetime as dt

    from nowcast.ingest.sec_filings import assign_period_end, earnings_8ks, fetch_release_text, fetch_submissions

    u = load_universe(settings.universe_path)
    sec = sec_client(settings)
    con = connect(settings.db_path)
    init_schema(con)
    quarters = con.execute('SELECT ticker, "end" FROM quarters').df()
    quarters["end"] = pd.to_datetime(quarters["end"]).dt.date
    for t in u.active():
        rel = earnings_8ks(fetch_submissions(t.cik, sec))
        rel = rel[rel["filed"] >= dt.date(2015, 1, 1)].assign(ticker=t.ticker)
        texts = [fetch_release_text(t.cik, a, sec) for a in rel["accn"]]
        rel = rel.assign(exhibit=[x[0] for x in texts], text=[x[1] for x in texts])
        rel = assign_period_end(rel, quarters)
        replace_rows(con, "raw_releases", rel, key_col="ticker", keys=[t.ticker])
        log.info("%-5s releases=%3d  with EX-99.1=%3d  mapped to a quarter=%3d", t.ticker, len(rel),
                 rel["text"].notna().sum(), rel["period_end"].notna().sum())
    log.info("network requests made: SEC=%d", sec.requests_made)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="nowcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="fetch SEC facts + pageviews (cached) into DuckDB")
    ing.add_argument("--tickers", nargs="*")
    sub.add_parser("check-universe", help="data check against the §4 criteria")
    sub.add_parser("build", help="M2: quarters, restatements, features tables")
    sub.add_parser("ingest-releases", help="M4: 8-K Item 2.02 earnings releases (EX-99.1) -> raw_releases")
    sub.add_parser("evaluate", help="M3: walk-forward evaluation of every pre-registered spec")
    sc = sub.add_parser("spot-check", help="print recent quarters for manual checks against 10-Qs")
    sc.add_argument("tickers", nargs="+")
    sc.add_argument("-n", type=int, default=6)
    args = p.parse_args(argv)
    settings = load_settings()
    return {"ingest": cmd_ingest, "check-universe": cmd_check_universe,
            "build": cmd_build, "spot-check": cmd_spot_check,
            "evaluate": cmd_evaluate, "ingest-releases": cmd_ingest_releases}[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
