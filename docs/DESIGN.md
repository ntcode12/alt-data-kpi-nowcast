# Design: Alt-Data KPI Nowcast

**Status:** Draft v1 · **Author:** Nichollas Tidow · **Created:** 2026-11-23

## 1. Summary

Predict a company's **reported quarterly revenue growth before earnings** using a free alternative dataset (Wikipedia pageviews), validate the signal with strict point-in-time, walk-forward testing, and use an LLM agent to pull reported KPIs and **management guidance** out of earnings press releases. The output is a PM-style research note per company: *"we expect revenue growth to accelerate / decelerate vs last quarter and to land above / below the guidance midpoint, here's the evidence, here's how reliable it has been."*

This mirrors the core sector data science workflow: **frame an investment question → map alt data to a KPI → clean and align it → test it honestly → turn it into something actionable.**

## 2. The investment question

> For a set of consumer and TMT companies, does year-over-year growth in Wikipedia pageviews over a company's fiscal quarter tell us, **before the earnings release**, whether reported revenue growth will **accelerate or decelerate**, and whether revenue will come in **above or below management's guidance**?

Why these framings:
- Stocks move on **surprise**, not levels. Real consensus estimates aren't free, so we use two proxies: (a) **acceleration vs the prior quarter's growth** (a naive "same growth as last quarter" expectation), and (b) **the guidance midpoint**, which management publishes in the prior quarter's press release.
- **Direction** (accelerate vs decelerate, beat vs miss guidance) is what a PM acts on, so it's the headline metric, ahead of raw forecast error.

**A negative result is a valid result.** If pageviews don't beat the naive baseline out of sample, the write-up says so and explains why. That honesty is part of the point.

## 3. Goals / non-goals

**Goals**
- Reproducible pipeline from public data → features → walk-forward evaluation → per-company report.
- **Point-in-time correctness:** every prediction uses only data available on its as-of date.
- Honest evaluation against a naive baseline, with uncertainty.
- An LLM extraction agent with a **measured accuracy**, validated against XBRL ground truth.
- A short written research note that a non-technical PM could read in two minutes.

**Non-goals**
- Trading strategy, backtested P&L, or price prediction. The target is a fundamental KPI, not returns.
- Paid data (card panels, real consensus estimates).
- Production deployment, scheduling, or a web app. Notebooks + CLI are enough.
- Large universes. 8–12 well-chosen names beat 500 noisy ones.

## 4. Data sources

All free and public. Each client sends a descriptive `User-Agent` with a contact email read from `.env`. SEC and Wikimedia both require one.

| Source | What we get | Endpoint | Notes |
|---|---|---|---|
| **SEC XBRL company facts** | Reported quarterly revenue with period start/end dates, form, and **filed date** | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | The ground-truth target. Max 10 requests/sec. |
| **SEC submissions** | Filing index, used to find 8-K earnings releases | `https://data.sec.gov/submissions/CIK##########.json` | Filter 8-Ks with Item 2.02; the press release is usually exhibit EX-99.1. |
| **8-K EX-99.1 press releases** | Revenue, KPIs, **next-quarter guidance** | EDGAR archives | Input to the LLM agent. |
| **Wikimedia pageviews** | Daily views per article | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{article}/daily/{start}/{end}` | Available from 2015-07. `agent=user` filters known bots. |
| *(Optional)* Google Trends | Search interest | unofficial `pytrends` | Brittle and rate-limited. A stretch goal for triangulation only. |

### Universe (initial, adjust after data checks)

Consumer and TMT names where **the Wikipedia article is a reasonable proxy for consumer interest** and history is long enough: e.g. Netflix, Chipotle, Etsy, Lululemon, Shake Shack, Wingstop, Planet Fitness, Snap, Pinterest, Peloton, Zoom, Roblox.

Selection criteria (documented per ticker in `config/universe.yaml`):
- A US filer with 10-Q/10-K XBRL (excludes 20-F filers such as Spotify).
- At least **24 quarters** of overlapping revenue and pageview history.
- An article that is clearly the company or its main product. Avoid ambiguous names (e.g. "Crocs" the brand vs the animal) or document the choice.
- Mostly one segment, so that consumer interest plausibly drives revenue.

## 5. Architecture

```
            ┌────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
 ingest     │ sec_facts.py       │    │ wiki_pageviews.py    │    │ sec_filings.py     │
            │ XBRL revenue facts │    │ daily views/article  │    │ 8-K EX-99.1 text   │
            └─────────┬──────────┘    └──────────┬───────────┘    └─────────┬──────────┘
                      ▼                          ▼                          ▼
 storage    ┌────────────────────────────────────────────────────────────────────────┐
            │ DuckDB (data/nowcast.duckdb): raw_facts, raw_pageviews, raw_releases     │
            └─────────┬──────────────────────────┬──────────────────────────┬────────┘
                      ▼                          ▼                          ▼
 transform  ┌────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
            │ quarters.py        │    │ features.py          │    │ agent/extract.py   │
            │ fiscal periods,    │    │ views summed over    │    │ LLM → KPIs +       │
            │ Q4 derivation,     │──▶ │ exact fiscal period, │    │ guidance (Pydantic │
            │ first-filed values │    │ YoY, QTD as-of cuts  │    │ schema), validated │
            └─────────┬──────────┘    └──────────┬───────────┘    └─────────┬──────────┘
                      └──────────────┬───────────┘                          │
                                     ▼                                      ▼
 model      ┌────────────────────────────────────┐            ┌────────────────────────┐
            │ models.py: naive, OLS, pooled      │            │ guidance table          │
            │ evaluate.py: walk-forward, metrics │◀───────────│ (beat/miss vs midpoint) │
            └─────────────────┬──────────────────┘            └────────────────────────┘
                              ▼
 output     reports/<ticker>.md + charts (Plotly) · reports/summary.md
```

**Tech:** Python 3.11+, `uv`, DuckDB, pandas, statsmodels, scikit-learn, Plotly, httpx, Pydantic AI + Anthropic SDK, pytest. Ingestion and DuckDB patterns are reused from `alpha_trading_pipeline`.

## 6. Data model (DuckDB)

| Table | Grain | Key columns |
|---|---|---|
| `raw_facts` | one XBRL fact per filing | `cik, ticker, concept, start, end, val, form, fp, fy, filed, accn` |
| `raw_pageviews` | article × day | `article, date, views` |
| `raw_releases` | one 8-K earnings release | `ticker, accn, filed, period_end, text` |
| `quarters` | ticker × fiscal quarter | `ticker, fiscal_q, start, end, revenue, revenue_yoy, filed (as-of date)` |
| `features` | ticker × fiscal quarter × as-of cut | `ticker, fiscal_q, asof, views_period, views_yoy, coverage_days` |
| `guidance` | ticker × target quarter | `ticker, fiscal_q, guide_low, guide_high, guide_mid, source_accn, extracted_by` |
| `predictions` | ticker × quarter × model × as-of | `model, y_hat, y_true, direction_hat, direction_true` |

## 7. Methodology

### 7.1 Building the target: quarterly revenue, point in time
- **Concept resolution:** try, in order, `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`. Record which concept was used per ticker (companies switched concepts around the 2018 ASC 606 adoption).
- **Quarter facts:** keep facts whose `end − start` spans roughly 80–100 days.
- **Q4 derivation:** most companies report Q4 only inside the annual 10-K. Q4 = FY value − (Q1 + Q2 + Q3) for the same fiscal year, and its `filed` date is the 10-K's filed date.
- **First-filed values:** the same period appears in later filings, sometimes restated. Use the **earliest `filed`** value as the "as reported" number. Restatements are logged, not used.
- **Target:** `y_q = ln(rev_q / rev_{q−4})`, the YoY log growth. Using YoY removes seasonality, and log growth behaves well in regressions.
- **Acceleration:** `Δy_q = y_q − y_{q−1}`. Its sign is the headline label (accelerate / decelerate).

### 7.2 Building the feature: pageviews aligned to fiscal periods
- Sum daily views over **the exact `[start, end]` of each fiscal quarter** taken from XBRL. This handles 52/53-week and non-calendar fiscal years automatically. No calendar guessing.
- `x_q = ln(views_q / views_{q−4})`, YoY on matched fiscal periods. `Δx_q = x_q − x_{q−1}`.
- **As-of cuts (nowcasting):** compute features as of (a) the period end, and (b) **period end − 30 days** (quarter-to-date, scaled by days covered). This shows how early the signal works.
- **Cleaning:** winsorize daily views at the 99.5th percentile per article (bot spikes, viral events); flag days with zero or missing data; record `coverage_days`.
- **Known structural breaks to test:** Wikipedia mobile-traffic shifts, the 2023 desktop redesign, COVID 2020–21. YoY differencing absorbs slow drift but not step changes. Run robustness checks that exclude 2020Q2–2021Q2.

### 7.3 Models (simple first; complexity only if it wins out of sample)
| ID | Model | Purpose |
|---|---|---|
| M0 | **Naive:** `ŷ_q = y_{q−1}` (growth persists) | The baseline everything must beat. Its predicted acceleration is always 0. |
| M1 | **Per-ticker OLS:** `y_q = a + b·y_{q−1} + c·x_q` | Does the alt data add information beyond momentum? |
| M2 | **Pooled panel OLS** with ticker fixed effects, same features | More data per coefficient; tests whether the relationship is general. |
| M3 | **Direction classifier:** logistic regression of `sign(Δy_q)` on `Δx_q` | Directly models what a PM cares about. |
| M4 *(stretch)* | Ridge with the extra features (QTD cut, lagged `x`) | Only if M1/M2 show signal. |

Use HAC (Newey-West) standard errors for in-sample coefficient inference. Out-of-sample metrics decide everything.

### 7.4 Validation (the part that matters most)
- **Walk-forward with an expanding window:** train on all quarters with `filed < asof`, predict the next quarter, roll forward. The first prediction comes after ≥12 training quarters per ticker (M1) or the pooled equivalent (M2).
- **Leakage rules (enforced in code and tested):**
  1. Target values enter training only after their `filed` date.
  2. Features use only pageviews dated ≤ the as-of cut.
  3. Winsorization thresholds come from the training window only.
  4. Guidance comes from the press release *before* the target quarter.
- **Metrics:**
  - MAE of `ŷ` vs `y`, and **skill vs naive** = `1 − MAE_model / MAE_naive`
  - **Directional hit rate** on acceleration, with a binomial test against 50%
  - **Hit rate on beat/miss vs the guidance midpoint**, where guidance exists
  - Cross-sectional rank correlation (IC) of `Δx` vs `Δy` per quarter, averaged over time
  - Diebold-Mariano test for M1/M2 vs M0 forecast errors
- **Robustness:** exclude COVID quarters, leave one ticker out (pooled model), and compare the 30-day-early cut against the full-quarter cut.
- **Multiple-testing hygiene:** fix the universe, features, and metrics *before* looking at out-of-sample results (this doc), and report every ticker, not just the ones that worked.

## 8. LLM agent: KPI and guidance extraction

**Why:** guidance is the best free proxy for "what the market expects", and KPIs beyond revenue (subscribers, comparable sales) are where sector research lives. Both are buried in unstructured press releases.

- **Input:** 8-K EX-99.1 text, cleaned from HTML.
- **Agent:** Pydantic AI with the Anthropic SDK. Model is configurable via env; default `claude-sonnet-5-5`. Tools: `get_release(ticker, accn)`, `search_section(text, query)`.
- **Output schema** (Pydantic, validated):
  ```python
  class EarningsExtract(BaseModel):
      ticker: str
      period_end: date
      revenue_reported: Decimal | None      # in USD
      kpis: list[KPI]                       # name, value, unit, yoy_change
      guidance: Guidance | None             # target period, metric, low, high, unit
      evidence: list[str]                   # verbatim source quotes for every number
  ```
- **Guardrails:** every extracted number must come with a verbatim `evidence` quote that is checked to exist in the source text. Per-call usage limits cap tool loops. Results are cached by `accn` so re-runs are free and idempotent.
- **Evaluation (reported in the README):**
  - Revenue extraction accuracy vs XBRL ground truth (exact match within rounding), over all releases.
  - A hand-labeled set of ~30 releases for guidance and KPIs → precision and recall.
- **Point-in-time caveat:** an LLM may "know" historical outcomes from its training data. That's harmless for *extraction* (the source text is given) but must never be used to *predict*. The agent extracts only. It never forecasts.

## 9. Outputs

- `reports/<ticker>.md`, a one-page note:
  1. **Call:** "Q3 revenue growth likely **accelerates**; nowcast +14% YoY vs guidance-implied +11%."
  2. **Evidence:** one chart of pageview YoY vs revenue YoY over time, plus the current-quarter nowcast.
  3. **Reliability:** historical hit rate and skill vs naive for this ticker.
  4. **Caveats:** coverage, article ambiguity, structural breaks.
- `reports/summary.md`: a cross-ticker results table and an honest overall conclusion (where the signal works, where it doesn't, and why).

## 10. Repository layout (created as each milestone needs it, no empty placeholder files)

```
alt-data-kpi-nowcast/
├── config/universe.yaml        # tickers, CIKs, Wikipedia articles, rationale
├── src/nowcast/
│   ├── ingest/{sec_facts,sec_filings,wiki_pageviews}.py
│   ├── quarters.py  features.py  models.py  evaluate.py
│   ├── agent/extract.py
│   └── report.py
├── tests/                      # leakage tests first
├── notebooks/                  # exploration only; logic lives in src/
├── reports/
└── docs/DESIGN.md
```

## 11. Milestones

| # | Milestone | Done when | Est. |
|---|---|---|---|
| 1 | **Ingest** SEC facts + pageviews into DuckDB for the universe | Tables populated; universe.yaml finalized with ≥8 tickers passing the criteria | 1 day |
| 2 | **Align** quarters (Q4 derivation, first-filed) + features | `quarters` and `features` tables; spot-check 3 tickers by hand against 10-Qs | 1 day |
| 3 | **Evaluate** M0–M3 walk-forward | Leakage tests pass; results table produced | 1 day |
| 4 | **Agent** extraction + eval | Revenue accuracy vs XBRL reported; guidance table populated | 1–2 days |
| 5 | **Report** per-ticker notes + summary + README results | A PM could read `summary.md` and act on it | 0.5 day |

Two weekends total. **M1–M3 alone are a complete, resume-worthy project.** M4 adds the agentic angle.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pageviews don't predict revenue | Report it honestly; analyze which ticker characteristics (consumer share, single product) predict where the signal works |
| Short history (~30–40 quarters) → overfitting | Few parameters, pooled model, out-of-sample only, report confidence intervals |
| XBRL concept switches / missing Q4 | Concept fallback order; Q4 derivation; manual spot checks |
| Wikipedia article ≠ company demand | Selection criteria; document the article choice; drop tickers that fail sanity checks |
| Bot spikes / viral news | `agent=user`, winsorization, robustness checks |
| LLM extraction errors | Evidence quotes verified against source; accuracy measured vs XBRL |
| Scope creep | Non-goals list; M1–M3 first |

## 13. Compliance and data ethics

Public data only, no PII, and no MNPI. Respect the SEC fair-access policy (≤10 req/s, declared User-Agent) and Wikimedia's API policy (User-Agent, reasonable request rates). Cache everything locally to avoid repeated requests.

## 14. Open questions

- Final universe: run the data checks in milestone 1, then lock it.
- Is Google Trends worth adding for triangulation, given how brittle access is?
- Should the 30-day-early cut be the headline (more actionable) or the full-quarter cut (more accurate)?
- Guidance coverage: how many of the chosen companies give quarterly revenue guidance? (Netflix does; many restaurants guide annually or on comps only.)

## 15. Interview framing

This project is designed to answer the questions a sector data science interviewer asks:
- *"How do you map a dataset to a KPI?"* → §7.2, exact fiscal-period alignment.
- *"How do you avoid look-ahead bias?"* → §7.4, the leakage rules, tested in code.
- *"How do you know it's a real signal?"* → skill vs naive, binomial and DM tests, robustness, and every ticker reported.
- *"How would you use LLMs?"* → §8, extraction with verified evidence and measured accuracy, and never forecasting.
- *"What would you tell the PM?"* → §9, the call, the evidence, the reliability, the caveats.
