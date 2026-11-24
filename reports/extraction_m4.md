# Milestone 4: LLM extraction (claude-haiku-5-5)

Releases extracted: 30 (API calls this run: 0; the rest came from the cache). Errors/refusals: 0.

## Revenue accuracy vs XBRL (first-filed)
- Outcomes over 30 releases: match 28, no_revenue_extracted 2
- Accuracy where the model extracted a revenue figure: **28/28 (100%)**
- `no_revenue_extracted` can be correct: some Item 2.02 8-Ks are preliminary updates or deal announcements that state no quarterly revenue. These are listed below for review.
- Exact to the dollar: 26/30
- Revenue evidence quotes found in source: 93%

## Evidence verification
- KPI quotes found verbatim in source: 98% of 191
- Guidance quotes found verbatim in source: 96% of 134
- Releases with verified next-quarter dollar revenue guidance: 13/30; by ticker: ETSY 2, LULU 2, NFLX 1, PINS 3, PTON 2, RBLX 1, ZM 2

## Cost (actual token usage)
- Input tokens 460,020, output tokens 55,818 → **$0.0739** ($0.00246/release)
- Projected for all 496 releases at this rate: **$1.22**

## Releases not matched
| ticker   | period_end   | outcome              |   revenue_reported_usd |   xbrl_revenue |   rel_error |
|:---------|:-------------|:---------------------|-----------------------:|---------------:|------------:|
| PLNT     | 2021-12-31   | no_revenue_extracted |                    nan |      183640000 |         nan |
| WING     | 2020-03-28   | no_revenue_extracted |                    nan |       55436000 |         nan |

Not measured: guidance/KPI precision and recall against hand labels (hand-labeled set skipped by decision).
