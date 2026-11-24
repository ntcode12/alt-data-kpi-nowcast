"""M4 extraction guardrails (DESIGN.md §8) -- the Anthropic client is mocked; no API calls."""

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from nowcast.agent.extract import (
    EarningsExtract,
    Extractor,
    estimate_cost,
    guidance_rows,
    revenue_accuracy,
    verify,
)

D = dt.date
TEXT = """Netflix, Inc. reported revenue of $9,370,440 thousand for the quarter.
For Q1'24 we forecast revenue of $9.24 billion. Paid memberships grew to 260.28 million."""


def extract(**kw):
    base = dict(
        revenue_reported_usd=9_370_440_000.0,
        revenue_evidence="reported revenue of $9,370,440 thousand for the quarter",
        kpis=[dict(name="paid memberships", value=260.28, unit="million", evidence="Paid memberships grew to 260.28 million")],
        guidance=[dict(target_period="Q1'24", metric="revenue", low_usd=9.24e9, high_usd=9.24e9,
                       is_next_quarter=True, evidence="For Q1'24 we forecast revenue of $9.24 billion")],
    )
    base.update(kw)
    return EarningsExtract(**base)


class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed, self.stop_reason, self.calls = parsed, stop_reason, 0

    def parse(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=5000, output_tokens=800), stop_details=None)


def fake_client(parsed, **kw):
    return SimpleNamespace(messages=FakeMessages(parsed, **kw))


def test_verify_accepts_quotes_present_in_source_whitespace_insensitive():
    e = extract(revenue_evidence="reported revenue of $9,370,440 thousand\n for the quarter")
    v = verify(e, TEXT)
    assert v.revenue_verified and v.kpis_verified == [True] and v.guidance_verified == [True]


def test_verify_rejects_fabricated_quote():
    v = verify(extract(revenue_evidence="revenue of $10 billion"), TEXT)
    assert not v.revenue_verified


def test_extractor_caches_by_accession(tmp_path):
    client = fake_client(extract())
    ex = Extractor(client, model="claude-haiku-5-5", cache_dir=tmp_path)
    r1 = ex.run("NFLX", "0001065280-24-000006", D(2023, 12, 31), TEXT)
    r2 = ex.run("NFLX", "0001065280-24-000006", D(2023, 12, 31), TEXT)
    assert client.messages.calls == 1
    strip = lambda r: {k: v for k, v in r.items() if k != "cached"}
    assert strip(r1) == strip(r2) and r1["input_tokens"] == 5000 and r1["cached"] is False and r2["cached"] is True


def test_extractor_sends_full_text_and_model(tmp_path):
    client = fake_client(extract())
    Extractor(client, model="claude-haiku-5-5", cache_dir=tmp_path).run("NFLX", "a-1", D(2023, 12, 31), TEXT)
    kw = client.messages.kwargs
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["output_format"] is EarningsExtract
    assert TEXT in kw["messages"][0]["content"]          # never truncated


def test_extractor_records_refusal_without_crashing(tmp_path):
    client = fake_client(None, stop_reason="refusal")
    r = Extractor(client, model="claude-haiku-5-5", cache_dir=tmp_path).run("NFLX", "a-2", D(2023, 12, 31), TEXT)
    assert r["error"] == "refusal" and r["extract"] is None


def test_estimate_cost_haiku():
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_revenue_accuracy_against_xbrl():
    ext = pd.DataFrame({"ticker": ["T", "T", "T", "T"], "period_end": [D(2023, 3, 31), D(2023, 6, 30), D(2023, 9, 30), D(2023, 12, 31)],
                        "revenue_reported_usd": [100_000_000.0, 12.05e9, 50.0, None], "revenue_verified": [True, True, True, True]})
    q = pd.DataFrame({"ticker": ["T"] * 4, "end": [D(2023, 3, 31), D(2023, 6, 30), D(2023, 9, 30), D(2023, 12, 31)],
                      "revenue": [100_000_000, 12_050_762_000, 1_000_000, 5]})
    acc = revenue_accuracy(ext, q)
    assert acc.set_index("period_end")["match"].tolist() == [True, True, False, False]


def test_guidance_rows_only_verified_next_quarter_revenue():
    e = extract(guidance=[
        dict(target_period="Q1'24", metric="revenue", low_usd=9.2e9, high_usd=9.3e9, is_next_quarter=True,
             evidence="For Q1'24 we forecast revenue of $9.24 billion"),
        dict(target_period="FY24", metric="revenue", low_usd=38e9, high_usd=39e9, is_next_quarter=False,
             evidence="For Q1'24 we forecast revenue of $9.24 billion"),
        dict(target_period="Q1'24", metric="revenue", low_usd=1.0, high_usd=2.0, is_next_quarter=True,
             evidence="made up quote"),
    ])
    rows = guidance_rows("NFLX", "a-1", D(2024, 1, 23), D(2023, 12, 31), e, verify(e, TEXT), "claude-haiku-5-5")
    assert len(rows) == 1
    r = rows[0]
    assert r["guide_mid"] == pytest.approx(9.25e9) and r["source_accn"] == "a-1"
    assert r["source_filed"] == D(2024, 1, 23)           # leakage rule 4: known only from this date


def test_verify_ignores_table_cell_separators_and_zero_width_chars():
    src = "Three Months Ended |  | 2026 |  | Revenue |  | $ | 1,173,929 |  |  | $ | 1,049,205 |​ Cost​ ​of revenue"
    e = extract(revenue_reported_usd=1_173_929_000.0, revenue_evidence="Revenue | $ | 1,173,929 | | $ | 1,049,205",
                kpis=[], guidance=[])
    assert verify(e, src).revenue_verified
    e2 = extract(revenue_reported_usd=1.0, revenue_evidence="Revenue $ 1,173,930", kpis=[], guidance=[])
    assert not verify(e2, src).revenue_verified          # a changed digit still fails


def test_revenue_accuracy_separates_nulls_from_errors():
    ext = pd.DataFrame({"ticker": ["T", "T"], "period_end": [D(2023, 3, 31), D(2023, 6, 30)],
                        "revenue_reported_usd": [None, 1.0], "revenue_verified": [False, True]})
    q = pd.DataFrame({"ticker": ["T", "T"], "end": [D(2023, 3, 31), D(2023, 6, 30)], "revenue": [5, 5]})
    acc = revenue_accuracy(ext, q).set_index("period_end")
    assert acc.loc[D(2023, 3, 31), "outcome"] == "no_revenue_extracted"
    assert acc.loc[D(2023, 6, 30), "outcome"] == "wrong"
