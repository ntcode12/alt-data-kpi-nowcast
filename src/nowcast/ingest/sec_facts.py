"""SEC XBRL company facts -> raw revenue facts (one row per fact per filing)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
RAW_FACTS_COLUMNS = [
    "cik", "ticker", "concept", "unit", "start", "end", "val",
    "form", "fp", "fy", "filed", "accn", "frame",
]
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def companyfacts_url(cik: int) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"


def fetch_companyfacts(cik: int, client) -> dict:
    return client.get_json(companyfacts_url(cik), f"sec/companyfacts/CIK{int(cik):010d}.json")


def fetch_company_tickers(client) -> dict[str, int]:
    """Ticker -> CIK map from SEC, used to verify universe.yaml."""
    doc = client.get_json(COMPANY_TICKERS_URL, "sec/company_tickers.json")
    return {row["ticker"].upper(): int(row["cik_str"]) for row in doc.values()}


def _date(s: str | None) -> dt.date | None:
    return dt.date.fromisoformat(s) if s else None


def parse_companyfacts(doc: dict, ticker: str, concepts=REVENUE_CONCEPTS) -> pd.DataFrame:
    """Flatten us-gaap revenue concepts (USD only). All concepts are kept; resolution is in quarters.py."""
    cik = int(doc.get("cik", 0))
    gaap = doc.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept in concepts:
        for f in gaap.get(concept, {}).get("units", {}).get("USD", []):
            rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "concept": concept,
                    "unit": "USD",
                    "start": _date(f.get("start")),
                    "end": _date(f["end"]),
                    "val": int(f["val"]),
                    "form": f.get("form"),
                    "fp": f.get("fp"),
                    "fy": f.get("fy"),
                    "filed": _date(f["filed"]),
                    "accn": f["accn"],
                    "frame": f.get("frame"),
                }
            )
    df = pd.DataFrame(rows, columns=RAW_FACTS_COLUMNS)
    if df.empty:
        return df
    # exact duplicates (same fact repeated within one filing) carry no information
    key = ["ticker", "concept", "start", "end", "accn", "val"]
    df = df.sort_values(key + ["frame"], na_position="last").drop_duplicates(subset=key, keep="first")
    df["fy"] = df["fy"].astype("Int64")
    return df.reset_index(drop=True)
