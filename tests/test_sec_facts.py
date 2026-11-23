import datetime as dt
import json
from pathlib import Path

from nowcast.ingest.sec_facts import (
    RAW_FACTS_COLUMNS,
    REVENUE_CONCEPTS,
    companyfacts_url,
    fetch_companyfacts,
    parse_companyfacts,
)

FIXTURE = Path(__file__).parent / "fixtures" / "companyfacts_min.json"


def _doc():
    return json.loads(FIXTURE.read_text())


def test_concept_fallback_order_matches_design():
    assert REVENUE_CONCEPTS == (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    )


def test_url_zero_pads_cik():
    assert companyfacts_url(1065280) == "https://data.sec.gov/api/xbrl/companyfacts/CIK0001065280.json"


def test_parse_columns_match_data_model():
    df = parse_companyfacts(_doc(), "NFLX")
    assert list(df.columns) == RAW_FACTS_COLUMNS
    for col in ["cik", "ticker", "concept", "start", "end", "val", "form", "fp", "fy", "filed", "accn"]:
        assert col in df.columns


def test_parse_keeps_only_usd_revenue_concepts_and_dedups():
    df = parse_companyfacts(_doc(), "NFLX")
    assert set(df["concept"]) == {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"}
    assert (df["unit"] == "USD").all()
    assert (df["ticker"] == "NFLX").all() and (df["cik"] == 1065280).all()
    # 4 Revenues rows, one an exact duplicate -> 3; plus 1 ASC 606 row
    assert len(df) == 4


def test_parse_types_and_values():
    df = parse_companyfacts(_doc(), "NFLX")
    row = df[(df.concept == "Revenues") & (df.form == "10-K")].iloc[0]
    assert row["start"] == dt.date(2017, 1, 1)
    assert row["end"] == dt.date(2017, 12, 31)
    assert row["filed"] == dt.date(2018, 1, 29)
    assert row["val"] == 11692713000
    assert row["fy"] == 2017 and row["fp"] == "FY"
    assert row["frame"] == "CY2017"
    # missing frame -> None, not NaN string
    assert df["frame"].isna().sum() == 1


def test_parse_handles_company_without_revenue_concepts():
    doc = {"cik": 1, "facts": {"us-gaap": {}}}
    df = parse_companyfacts(doc, "X")
    assert df.empty and list(df.columns) == RAW_FACTS_COLUMNS


def test_fetch_uses_padded_cache_key():
    class Spy:
        def get_json(self, url, cache_key):
            self.args = (url, cache_key)
            return {}

    spy = Spy()
    fetch_companyfacts(320193, spy)
    assert spy.args == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "sec/companyfacts/CIK0000320193.json",
    )
