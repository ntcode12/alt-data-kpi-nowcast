# Alt-Data KPI Nowcast

Predicting quarterly **revenue growth before earnings** from a free alternative dataset (Wikipedia pageviews), with point-in-time, walk-forward validation against a naive baseline, plus an LLM agent that extracts KPIs and management guidance from earnings press releases.

**Question:** does YoY growth in pageviews over a company's fiscal quarter tell us whether reported revenue growth will **accelerate or decelerate**, and whether it will beat **management's guidance**?

**Status:** design phase. See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design: data sources, methodology, leakage rules, evaluation, and milestones.

## Data

| Source | Use |
|---|---|
| SEC XBRL company facts | Reported quarterly revenue (point in time, first-filed values) |
| SEC 8-K EX-99.1 | Earnings press releases → KPIs and guidance (LLM extraction) |
| Wikimedia pageviews API | Alternative-data signal, aggregated over exact fiscal periods |

## Setup

```bash
cp .env.example .env   # add a contact email for the SEC / Wikimedia User-Agent and your Anthropic key
uv sync
```

## Results

_To be added after milestone 3._
