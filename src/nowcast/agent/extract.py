"""LLM extraction of reported revenue, KPIs and next-quarter guidance from 8-K EX-99.1 releases (DESIGN.md §8).

One structured-output call per release (`client.messages.parse` with a Pydantic schema). Guardrails:
- the full release text is sent; nothing is truncated;
- every number carries a verbatim evidence quote that is checked against the source text, and numbers whose
  quote is not found are never used downstream;
- results are cached by accession number, so re-runs make no API calls;
- the model only extracts what the text says. It is never asked to forecast (point-in-time caveat, §8).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# USD per million tokens (input, output), prompts <= 100K tokens. Source: claude-api skill model table, 2026-11-03.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}

SYSTEM = """You extract facts from a US public company's quarterly earnings press release (SEC 8-K exhibit 99.1).
Extract only what the text states. Do not estimate, infer, or use outside knowledge, and never forecast.

- revenue_reported_usd: total revenue (or net revenue / total revenues) for the three months just ended, in US
  dollars as a plain number (convert "in thousands" / "in millions" table units and "$9.37 billion" to dollars).
  Use null if the release does not state it.
- kpis: up to 8 operating KPIs the release reports for the quarter (e.g. paid memberships, comparable sales growth,
  daily active users, gross merchandise sales), with numeric value and unit as written.
- guidance: management's forward-looking numeric outlook items. For each give the target period as written, the
  metric, low and high in US dollars when the metric is in dollars (low = high for a point estimate, null if not a
  dollar amount), and is_next_quarter = true only if the target is the single fiscal quarter immediately after
  the quarter being reported.
- Every number needs an evidence quote copied verbatim from the release (a short contiguous span, max ~30 words)
  that contains the number."""


class KPI(BaseModel):
    name: str
    value: float
    unit: str
    evidence: str


class Guidance(BaseModel):
    target_period: str
    metric: str
    low_usd: float | None = None
    high_usd: float | None = None
    is_next_quarter: bool
    evidence: str


class EarningsExtract(BaseModel):
    revenue_reported_usd: float | None = None
    revenue_evidence: str | None = None
    kpis: list[KPI] = Field(default_factory=list)
    guidance: list[Guidance] = Field(default_factory=list)


@dataclass
class Verification:
    revenue_verified: bool
    kpis_verified: list[bool]
    guidance_verified: list[bool]


_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def _norm(s: str) -> str:
    """Whitespace-, case- and table-separator-insensitive form. Words and digits must still match exactly;
    '|' cell separators and zero-width characters are artifacts of our HTML-to-text step, not content."""
    s = s.translate(_INVISIBLE).replace("\xa0", " ").replace("|", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _in_source(quote: str | None, text_norm: str) -> bool:
    return bool(quote) and len(_norm(quote)) >= 8 and _norm(quote) in text_norm


def verify(e: EarningsExtract, text: str) -> Verification:
    t = _norm(text)
    return Verification(
        revenue_verified=e.revenue_reported_usd is not None and _in_source(e.revenue_evidence, t),
        kpis_verified=[_in_source(k.evidence, t) for k in e.kpis],
        guidance_verified=[_in_source(g.evidence, t) for g in e.guidance],
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


class Extractor:
    def __init__(self, client, model: str, cache_dir: Path, max_tokens: int = 16000):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.cache_dir = Path(cache_dir) / "llm" / model
        self.api_calls = 0

    def run(self, ticker: str, accn: str, period_end: dt.date | None, text: str) -> dict:
        path = self.cache_dir / f"{accn}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        quarter = f"the fiscal quarter ended {period_end}" if period_end else "the quarter being reported"
        self.api_calls += 1
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Company ticker: {ticker}. The release reports on {quarter}.\n\n"
                                                  f"<release>\n{text}\n</release>"}],
            output_format=EarningsExtract,
        )
        error = None if resp.stop_reason not in ("refusal", "max_tokens") else resp.stop_reason
        parsed = resp.parsed_output if error is None else None
        rec = dict(ticker=ticker, accn=accn, period_end=str(period_end) if period_end else None, model=self.model,
                   extract=parsed.model_dump() if parsed is not None else None, error=error,
                   input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
        return {**rec, "cached": False}


def revenue_accuracy(ext: pd.DataFrame, quarters: pd.DataFrame, tol: float = 0.005) -> pd.DataFrame:
    """Extracted vs XBRL first-filed revenue. Match = verified quote and relative error <= tol
    (0.5% covers press releases that round, e.g. "$12.05 billion")."""
    m = ext.merge(quarters.rename(columns={"end": "period_end", "revenue": "xbrl_revenue"}),
                  on=["ticker", "period_end"], how="inner")
    rel = (m["revenue_reported_usd"] - m["xbrl_revenue"]).abs() / m["xbrl_revenue"]
    m["rel_error"] = rel
    m["match"] = (rel <= tol).fillna(False) & m["revenue_verified"].astype(bool)
    m["outcome"] = np.select(
        [m["revenue_reported_usd"].isna(), m["match"], (rel <= tol).fillna(False)],
        ["no_revenue_extracted", "match", "right_value_unverified_quote"], default="wrong")
    return m


def guidance_rows(ticker, accn, filed, period_end, e: EarningsExtract, v: Verification, model) -> list[dict]:
    """Verified next-quarter revenue guidance in dollars. Usable only from the release's filed date (rule 4)."""
    rows = []
    for g, ok in zip(e.guidance, v.guidance_verified):
        if not (ok and g.is_next_quarter and "revenue" in g.metric.lower()
                and g.low_usd is not None and g.high_usd is not None):
            continue
        lo, hi = sorted([g.low_usd, g.high_usd])
        rows.append(dict(ticker=ticker, base_period_end=period_end, target_period=g.target_period,
                         guide_low=lo, guide_high=hi, guide_mid=(lo + hi) / 2,
                         source_accn=accn, source_filed=filed, extracted_by=model))
    return rows[:1]
