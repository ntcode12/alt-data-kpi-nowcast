"""Run the extractor over a sample (pilot) or all releases; write tables and reports/extraction_m4.md."""

from __future__ import annotations

import datetime as dt
import json
import logging

import pandas as pd

from nowcast.agent.extract import EarningsExtract, Extractor, estimate_cost, guidance_rows, revenue_accuracy, verify

log = logging.getLogger("nowcast")


def pilot_sample(releases: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    """Stratified by ticker (round-robin), deterministic."""
    r = releases.dropna(subset=["text", "period_end"]).sample(frac=1.0, random_state=seed)
    r["_k"] = r.groupby("ticker").cumcount()
    return r.sort_values(["_k", "ticker"]).head(n).drop(columns="_k")


def run(con, client, model: str, cache_dir, n: int | None):
    rel = con.execute("SELECT * FROM raw_releases").df()
    for c in ["filed", "period_end"]:
        rel[c] = pd.to_datetime(rel[c]).dt.date
    rel = rel.where(rel.notna(), None)
    todo = pilot_sample(rel, n) if n else rel.dropna(subset=["text", "period_end"])
    ex = Extractor(client, model=model, cache_dir=cache_dir)
    q = con.execute('SELECT ticker, fiscal_q, "end", revenue FROM quarters').df()
    q["end"] = pd.to_datetime(q["end"]).dt.date
    ext_rows, guide = [], []
    for i, r in enumerate(todo.itertuples(), 1):
        rec = ex.run(r.ticker, r.accn, r.period_end, r.text)
        e = EarningsExtract(**rec["extract"]) if rec["extract"] else None
        v = verify(e, r.text) if e else None
        ext_rows.append(dict(
            ticker=r.ticker, accn=r.accn, period_end=r.period_end, model=model,
            revenue_reported_usd=e.revenue_reported_usd if e else None,
            revenue_verified=v.revenue_verified if v else False,
            n_kpis=len(e.kpis) if e else 0, n_kpis_verified=sum(v.kpis_verified) if v else 0,
            n_guidance=len(e.guidance) if e else 0, n_guidance_verified=sum(v.guidance_verified) if v else 0,
            error=rec["error"], input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
            json=json.dumps(rec["extract"])))
        if e:
            for g in guidance_rows(r.ticker, r.accn, r.filed, r.period_end, e, v, model):
                tq = q[(q.ticker == r.ticker) & q["end"].map(lambda d: 80 <= (d - r.period_end).days <= 100)]
                g["target_end"] = tq["end"].iloc[0] if len(tq) else None
                g["fiscal_q"] = tq["fiscal_q"].iloc[0] if len(tq) else None
                guide.append(g)
        if i % 10 == 0:
            log.info("extracted %d/%d (API calls so far: %d)", i, len(todo), ex.api_calls)
    ext = pd.DataFrame(ext_rows)
    return ext, pd.DataFrame(guide), q, ex.api_calls


def report(ext: pd.DataFrame, guide: pd.DataFrame, q: pd.DataFrame, model: str, n_all: int, api_calls: int) -> str:
    acc = revenue_accuracy(ext, q.rename(columns={}).drop(columns="fiscal_q"))
    tin, tout = int(ext["input_tokens"].sum()), int(ext["output_tokens"].sum())
    cost = estimate_cost(model, tin, tout)
    per = cost / max(len(ext), 1)
    kv = ext["n_kpis_verified"].sum() / max(ext["n_kpis"].sum(), 1)
    gv = ext["n_guidance_verified"].sum() / max(ext["n_guidance"].sum(), 1)
    miss = acc[~acc["match"]][["ticker", "period_end", "outcome", "revenue_reported_usd", "xbrl_revenue",
                              "rel_error"]]
    lines = [
        f"# Milestone 4: LLM extraction ({model})",
        "",
        f"Releases extracted: {len(ext)} (API calls this run: {api_calls}; the rest came from the cache). "
        f"Errors/refusals: {int(ext['error'].notna().sum())}.",
        "",
        "## Revenue accuracy vs XBRL (first-filed)",
        f"- Outcomes over {len(acc)} releases: "
        + ", ".join(f"{k} {v}" for k, v in acc["outcome"].value_counts().items()),
        f"- Accuracy where the model extracted a revenue figure: **{int(acc['match'].sum())}/"
        f"{int(acc['revenue_reported_usd'].notna().sum())} "
        f"({acc['match'].sum() / max(acc['revenue_reported_usd'].notna().sum(), 1):.0%})**",
        "- `no_revenue_extracted` can be correct: some Item 2.02 8-Ks are preliminary updates or deal "
        "announcements that state no quarterly revenue. These are listed below for review.",
        f"- Exact to the dollar: {int((acc['rel_error'] == 0).sum())}/{len(acc)}",
        f"- Revenue evidence quotes found in source: {ext['revenue_verified'].mean():.0%}",
        "",
        "## Evidence verification",
        f"- KPI quotes found verbatim in source: {kv:.0%} of {int(ext['n_kpis'].sum())}",
        f"- Guidance quotes found verbatim in source: {gv:.0%} of {int(ext['n_guidance'].sum())}",
        f"- Releases with verified next-quarter dollar revenue guidance: {guide['source_accn'].nunique() if len(guide) else 0}"
        f"/{len(ext)}; by ticker: "
        + (", ".join(f"{t} {k}" for t, k in guide.groupby('ticker').size().items()) if len(guide) else "none"),
        "",
        "## Cost (actual token usage)",
        f"- Input tokens {tin:,}, output tokens {tout:,} → **${cost:.4f}** (${per:.5f}/release)",
        f"- Projected for all {n_all} releases at this rate: **${per * n_all:.2f}**",
        "",
        "## Releases not matched",
        miss.to_markdown(index=False) if len(miss) else "None.",
        "",
        "Not measured: guidance/KPI precision and recall against hand labels (hand-labeled set skipped by decision).",
    ]
    return "\n".join(lines) + "\n"
